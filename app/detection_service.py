from ultralytics import YOLO
import cv2
import numpy as np
import asyncio
import asyncpg
import os
import json
import traceback
from datetime import datetime
from collections import deque
from typing import Optional, Dict, List
import threading
import time
import torch
import requests
import ultralytics.utils.loss as _loss_mod
if not hasattr(_loss_mod, 'DFLoss'):
    class DFLoss(torch.nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
    _loss_mod.DFLoss = DFLoss


class DetectionService:
    # Thresholds
    PERSON_THRESH  = 0.70
    CIG_THRESH     = 0.50
    DISPLAY_THRESH = 0.50
    SMOKE_THRESH   = 0.50
    # Timing
    BUFFER_SECS    = 5
    FPS            = 30
    COOLDOWN       = 20
    SMOKE_BUF      = 15   # ~0.5 s at 30 fps

    def __init__(self, database):
        self.db    = database
        self.model = self.smoke_model = None
        self.device = 'cpu'
        self.active_detections:  Dict[int, threading.Thread]       = {}
        self.video_buffers:      Dict[int, deque]                  = {}
        self.frame_times:        Dict[int, deque]                  = {}
        self.recording_states:   Dict[int, dict]                   = {}
        self.latest_frames:      Dict[int, Optional[np.ndarray]]   = {}
        self.frame_locks:        Dict[int, threading.Lock]         = {}
        self.latest_detections:  Dict[int, List[dict]]             = {}
        self.smoke_buffer:       Dict[int, deque]                  = {}
        self.save_semaphore = threading.Semaphore(2)
        self._root      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._clips_dir = os.path.join(self._root, "clips")
        self._weights   = os.path.join(self._root, "weights")
        os.makedirs(self._clips_dir, exist_ok=True)
        self.model       = self._load_model("best.pt", ["smoke_best.pt", "best (1).pt", "best(1).pt"])
        self.smoke_model = self._load_model("11k.pt")
        if self.model is None:
            raise RuntimeError("Main model not found in weights/")

    # ── Device / model loading ──────────────────────────────────────────

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            try:
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                return "cuda:0"
            except Exception:
                pass
        return "cpu"

    def _load_model(self, primary: str, fallbacks: list = None) -> Optional[YOLO]:
        for name in [primary] + (fallbacks or []):
            path = os.path.join(self._weights, name)
            if not os.path.exists(path):
                continue
            try:
                m = YOLO(path)
                if self.device == 'cpu':
                    self.device = self._get_device()
                if self.device.startswith("cuda"):
                    try:
                        m.to(self.device)
                    except Exception:
                        self.device = "cpu"
                print(f"Loaded {name} | classes: {list(m.names.values())}")
                return m
            except Exception as e:
                print(f"Error loading {name}: {e}")
        print(f"Model not found: {primary}")
        return None

    # ── Camera lifecycle ────────────────────────────────────────────────

    async def start_detection(self, camera_id: int, camera_info: Dict, ws_manager) -> bool:
        if camera_id in self.active_detections:
            return False
        buf = int(self.BUFFER_SECS * self.FPS)
        self.video_buffers[camera_id]     = deque(maxlen=buf)
        self.frame_times[camera_id]       = deque(maxlen=buf)
        self.recording_states[camera_id]  = {"is_recording": False, "start_time": None}
        self.latest_frames[camera_id]     = None
        self.frame_locks[camera_id]       = threading.Lock()
        self.latest_detections[camera_id] = []
        self.smoke_buffer[camera_id]      = deque(maxlen=self.SMOKE_BUF)
        t = threading.Thread(
            target=self._detection_loop,
            args=(camera_id, camera_info, ws_manager),
            daemon=True,
        )
        t.start()
        self.active_detections[camera_id] = t
        return True

    async def stop_detection(self, camera_id: int) -> bool:
        if camera_id not in self.active_detections:
            return False
        self.recording_states.get(camera_id, {})["stop"] = True
        await asyncio.sleep(1)
        for store in (self.active_detections, self.video_buffers, self.frame_times,
                      self.recording_states, self.latest_frames, self.frame_locks,
                      self.latest_detections, self.smoke_buffer):
            store.pop(camera_id, None)
        return True

    def is_detection_running(self, camera_id: int) -> bool:
        return camera_id in self.active_detections

    # ── Camera open ─────────────────────────────────────────────────────

    def _open_camera(self, source) -> Optional[cv2.VideoCapture]:
        is_rtsp = isinstance(source, str) and source.startswith("rtsp://")
        if is_rtsp:
            for attempt in range(3):
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, self.FPS)
                    time.sleep(0.5)
                    for _ in range(5):
                        ret, f = cap.read()
                        if ret and f is not None:
                            print(f"RTSP opened (attempt {attempt+1}): {f.shape[1]}x{f.shape[0]}")
                            return cap
                        time.sleep(0.2)
                cap.release()
                if attempt < 2:
                    time.sleep(2)
            print(f"Failed to open RTSP: {source}")
            return None

        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                ret, f = cap.read()
                if ret and f is not None:
                    return cap
                cap.release()

        for alt in (1, 2):
            cap = cv2.VideoCapture(alt, cv2.CAP_ANY)
            if cap.isOpened():
                ret, f = cap.read()
                if ret and f is not None:
                    print(f"Using fallback camera index {alt}")
                    return cap
                cap.release()

        print(f"Cannot open camera: {source}")
        return None

    # ── Detection loop ──────────────────────────────────────────────────

    def _set_latest(self, camera_id: int, frame: np.ndarray):
        try:
            with self.frame_locks[camera_id]:
                self.latest_frames[camera_id] = frame
        except Exception:
            pass

    def _detection_loop(self, camera_id: int, camera_info: Dict, ws_manager):
        source  = camera_info["source"]
        is_rtsp = isinstance(source, str) and source.startswith("rtsp://")
        cap     = self._open_camera(source)
        if cap is None:
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, self.FPS)
        print(f"Detection started for camera {camera_id}")

        frame_count        = 0
        last_detected_time = 0
        is_processing      = False
        read_errors        = 0
        reconnects         = 0
        max_errors         = 200 if is_rtsp else 30

        try:
            while camera_id in self.active_detections:
                ret, frame = cap.read()
                if not ret or frame is None:
                    read_errors += 1
                    # RTSP reconnect logic
                    if is_rtsp and read_errors % 100 == 0 and reconnects < 5:
                        reconnects += 1
                        cap.release()
                        time.sleep(1)
                        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        time.sleep(0.3)
                        ret2, f2 = cap.read()
                        if ret2 and f2 is not None:
                            read_errors = reconnects = 0
                            self._set_latest(camera_id, f2)
                            continue
                    if read_errors > max_errors:
                        break
                    wait = min(0.05 * (1 + read_errors // 10), 0.5) if is_rtsp else 0.1
                    time.sleep(wait)
                    continue

                read_errors = reconnects = 0
                frame_count += 1
                current_time = time.time()
                frame_copy   = frame.copy()

                # Store raw frame immediately for streaming continuity
                self._set_latest(camera_id, frame_copy)
                self.video_buffers[camera_id].append(frame_copy)
                self.frame_times[camera_id].append(current_time)

                # Skip every other frame when RTSP buffer fills up (reduces lag)
                if is_rtsp and len(self.video_buffers[camera_id]) > 30 and frame_count % 2:
                    continue

                # ── Model inference ──
                bbox_display, p_det, p_conf, p_bbox, c_det, c_conf, c_bbox, comb_conf = \
                    self._process_main_model(frame)
                s_det, s_conf, s_bbox = self._process_smoke(
                    frame, camera_id, p_det, p_bbox, c_det, c_bbox)

                # ── Build annotated display frame ──
                display = frame_copy.copy()
                if bbox_display:
                    self._draw_boxes(display, bbox_display)
                if s_det and s_bbox:
                    self._draw_boxes(display, [{"class": "Smoke", "confidence": s_conf, "bbox": s_bbox}])

                all_display = bbox_display + ([{"class": "Smoke", "confidence": s_conf, "bbox": s_bbox}] if s_det and s_bbox else [])
                self._set_latest(camera_id, display)
                with self.frame_locks[camera_id]:
                    self.latest_detections[camera_id] = all_display

                # ── Violation trigger ──
                if (p_det and c_det and s_det
                        and comb_conf > self.PERSON_THRESH
                        and (current_time - last_detected_time) > self.COOLDOWN
                        and not is_processing
                        and not self.recording_states.get(camera_id, {}).get("is_recording")):

                    is_processing = True
                    last_detected_time = current_time
                    self.recording_states[camera_id].update({"is_recording": True, "start_time": current_time})

                    final_conf  = (p_conf + c_conf + s_conf) / 3
                    det_class   = "Person+Cigarette+Smoke"
                    bbox_for_db = bbox_display + ([{"class": "Smoke", "confidence": s_conf, "bbox": s_bbox}] if s_bbox else [])
                    ts          = datetime.now()
                    print(f"VIOLATION: {det_class} cam={camera_id} conf={final_conf:.2f}")
                    self._trigger_violation(camera_id, det_class, final_conf, bbox_for_db, ws_manager, ts)

                    def _reset(cid=camera_id):
                        time.sleep(self.COOLDOWN)
                        nonlocal is_processing
                        is_processing = False
                        self.recording_states.get(cid, {}).update({"is_recording": False})
                    threading.Thread(target=_reset, daemon=True).start()

                time.sleep(1.0 / self.FPS)

        except Exception as e:
            print(f"Detection loop error camera {camera_id}: {e}")
            traceback.print_exc()
        finally:
            cap.release()
            print(f"Detection stopped for camera {camera_id}")

    # ── Model inference helpers ─────────────────────────────────────────

    def _process_main_model(self, frame):
        """Run main YOLO, return (bbox_display, p_det, p_conf, p_bbox, c_det, c_conf, c_bbox, comb_conf)."""
        empty = ([], False, 0, None, False, 0, None, 0)
        if self.model is None:
            return empty
        try:
            results = self.model(frame, device=self.device, verbose=False, iou=0.3, conf=0.20, max_det=20)
        except Exception:
            return empty

        all_dets = []
        for r in results:
            for box in r.boxes:
                cls  = int(box.cls[0])
                name = self.model.names[cls].lower()
                if name not in ('person', 'cigarette'):
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                all_dets.append({"class": name, "conf": conf, "bbox": [x1, y1, x2, y2],
                                  "area": (x2 - x1) * (y2 - y1)})

        all_dets = self._apply_nms(all_dets)
        persons  = [(d["conf"], d["bbox"]) for d in all_dets if d["class"] == "person"]
        cigs     = [(d["conf"], d["bbox"]) for d in all_dets if d["class"] == "cigarette"]
        pairs    = self._match_pairs(persons, cigs)

        bbox_display: List[dict] = []
        if pairs:
            pairs.sort(key=lambda x: (x[0] + x[2]) / 2, reverse=True)
            p_conf, p_bbox, c_conf, c_bbox = pairs[0]
            comb = (p_conf + c_conf) / 2
            for pc, pb, cc, cb in pairs:
                bbox_display += [
                    {"class": "Person",    "confidence": pc, "bbox": pb},
                    {"class": "Cigarette", "confidence": cc, "bbox": cb},
                ]
            bbox_display = self._deduplicate(bbox_display)
            # Also show unpaired detections above display threshold
            paired_p = {tuple(pb) for _, pb, _, _ in pairs}
            paired_c = {tuple(cb) for _, _, _, cb in pairs}
            for conf, bbox in persons:
                if conf >= self.DISPLAY_THRESH and tuple(bbox) not in paired_p:
                    bbox_display.append({"class": "Person", "confidence": conf, "bbox": bbox})
            for conf, bbox in cigs:
                if conf >= self.DISPLAY_THRESH and tuple(bbox) not in paired_c:
                    bbox_display.append({"class": "Cigarette", "confidence": conf, "bbox": bbox})
            return bbox_display, True, p_conf, p_bbox, True, c_conf, c_bbox, comb

        # No valid pairs — show anything above display threshold for debugging
        for conf, bbox in persons:
            if conf >= self.DISPLAY_THRESH:
                bbox_display.append({"class": "Person", "confidence": conf, "bbox": bbox})
        for conf, bbox in cigs:
            if conf >= self.DISPLAY_THRESH:
                bbox_display.append({"class": "Cigarette", "confidence": conf, "bbox": bbox})
        return bbox_display, False, 0, None, False, 0, None, 0

    def _match_pairs(self, persons, cigs):
        """Spatially match (conf, bbox) persons to nearest cigarettes."""
        pairs, used = [], set()
        for p_conf, p_bbox in persons:
            if p_conf < self.PERSON_THRESH:
                continue
            px = (p_bbox[0] + p_bbox[2]) / 2
            py = (p_bbox[1] + p_bbox[3]) / 2
            pw = p_bbox[2] - p_bbox[0]
            ph = p_bbox[3] - p_bbox[1]
            radius = max(pw, ph) * 2.5
            best_i = best_cc = best_cb = None
            for i, (c_conf, c_bbox) in enumerate(cigs):
                if i in used or c_conf < self.CIG_THRESH:
                    continue
                cx = (c_bbox[0] + c_bbox[2]) / 2
                cy = (c_bbox[1] + c_bbox[3]) / 2
                dist   = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                in_box = (p_bbox[0] - pw * .3 <= cx <= p_bbox[2] + pw * .3 and
                          p_bbox[1] - ph * .3 <= cy <= p_bbox[3] + ph * .3)
                if (dist <= radius or in_box) and (best_i is None or c_conf > best_cc):
                    best_i, best_cc, best_cb = i, c_conf, c_bbox
            if best_i is not None:
                pairs.append((p_conf, p_bbox, best_cc, best_cb))
                used.add(best_i)
        return pairs

    def _process_smoke(self, frame, camera_id, p_det, p_bbox, c_det, c_bbox):
        """Run smoke model, update buffer, check spatial match. Returns (detected, conf, bbox)."""
        if self.smoke_model is None:
            return False, 0, None
        try:
            results = self.smoke_model(frame, device=self.device, verbose=False, iou=0.3, conf=0.20, max_det=20)
        except Exception:
            return False, 0, None

        now = time.time()
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                name = str(self.smoke_model.names[cls]).lower()
                is_smoke_class = 'smoke' in name or 'mouth' in name or 'exhale' in name or cls == 0
                if is_smoke_class and conf >= self.SMOKE_THRESH:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    self.smoke_buffer[camera_id].append({
                        "conf": conf, "bbox": [x1, y1, x2, y2],
                        "center": [(x1 + x2) / 2, (y1 + y2) / 2], "ts": now,
                    })

        if not (p_det and c_det and p_bbox and c_bbox):
            return False, 0, None

        cx1 = min(p_bbox[0], c_bbox[0]); cy1 = min(p_bbox[1], c_bbox[1])
        cx2 = max(p_bbox[2], c_bbox[2]); cy2 = max(p_bbox[3], c_bbox[3])
        w, h = cx2 - cx1, cy2 - cy1
        sx1, sy1 = cx1 - w * .30, cy1 - h * .30
        sx2, sy2 = cx2 + w * .30, cy2 + h * .30
        max_d    = max(w, h) * 1.5
        mcx, mcy = (cx1 + cx2) / 2, (cy1 + cy2) / 2

        best_conf, best_bbox = 0, None
        for det in self.smoke_buffer[camera_id]:
            if now - det["ts"] > 2.0:
                continue
            scx, scy = det["center"]
            dist = ((scx - mcx) ** 2 + (scy - mcy) ** 2) ** 0.5
            if sx1 <= scx <= sx2 and sy1 <= scy <= sy2 and dist <= max_d:
                if det["conf"] > best_conf:
                    best_conf, best_bbox = det["conf"], det["bbox"]

        return (best_conf > 0), best_conf, best_bbox

    def _draw_boxes(self, frame, detections):
        colors = {"Person": (0, 255, 0), "Cigarette": (255, 165, 0), "Smoke": (0, 0, 255)}
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = colors.get(det["class"], (0, 255, 0))
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - lh - 5), (x1 + lw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ── Violation handling ──────────────────────────────────────────────

    def _run_async(self, coro):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()


    def _trigger_violation(self, camera_id, det_class, conf, bbox_data, ws_manager, ts: datetime):
        video_path = os.path.join(
            self._clips_dir,
            f"detection_{ts.strftime('%Y-%m-%d_%H-%M-%S')}_cam{camera_id}_{det_class}_{conf:.2f}.mp4",
        )
        violation = {
            "camera_id": camera_id, "detection_class": det_class, "confidence": conf,
            "video_path": video_path, "timestamp": ts, "frame_count": 0, "bbox_data": bbox_data,
        }
        alert = {
            "type": "violation_alert", "camera_id": camera_id,
            "detection_class": det_class, "confidence": conf, "timestamp": ts.isoformat(),
        }
        threading.Thread(target=self._run_async, args=(self._save_to_db(violation, ws_manager),), daemon=True).start()
        threading.Thread(target=self._run_async, args=(ws_manager.broadcast(alert),), daemon=True).start()
        threading.Thread(target=self._save_clip, args=(camera_id, bbox_data, video_path), daemon=True).start()

    async def _save_to_db(self, violation: dict, ws_manager):
        try:
            conn = await asyncpg.connect(
                host=self.db.db_host, port=self.db.db_port,
                database=self.db.db_name, user=self.db.db_user, password=self.db.db_password,
            )
            try:
                location = await self._get_location(conn, violation["camera_id"])
                vid = await conn.fetchval("""
                    INSERT INTO violations
                    (camera_id, detection_class, confidence, video_path, timestamp, frame_count, bbox_data, location)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8) RETURNING id
                """,
                    violation["camera_id"], violation["detection_class"], violation["confidence"],
                    violation["video_path"], violation["timestamp"], 0,
                    json.dumps(violation.get("bbox_data", [])), location,
                )
                print(f"Violation saved to DB: ID={vid}")
            finally:
                await conn.close()
        except Exception as e:
            print(f"DB save error: {e}")
            traceback.print_exc()

    def _save_clip(self, camera_id: int, bbox_data: list, video_path: str):
        with self.save_semaphore:
            self._write_clip(camera_id, bbox_data, video_path)

    def _write_clip(self, camera_id: int, bbox_data: list, video_path: str):
        try:
            buf = list(self.video_buffers.get(camera_id, []))
            if not buf:
                return

            # Collect 10 s of post-detection frames (5 s during + 5 s after)
            post, target = [], int(10 * self.FPS)
            iv   = 1.0 / self.FPS
            end  = time.time() + 10
            prev = time.time()
            while time.time() < end:
                if camera_id not in self.active_detections:
                    break
                now = time.time()
                if now - prev >= iv:
                    f = self.latest_frames.get(camera_id)
                    if f is not None:
                        post.append(f.copy())
                        prev = now
                        if len(post) > target:
                            post = post[-target:]
                time.sleep(iv)

            # Pad if needed
            if not post:
                post = [buf[-1].copy()] * target
            elif len(post) < target:
                post += [post[-1].copy()] * (target - len(post))

            frames = buf + post
            h, w   = frames[0].shape[:2]
            out    = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.FPS, (w, h))
            if not out.isOpened():
                print(f"Cannot open VideoWriter: {video_path}")
                return
            stored = self.latest_detections.get(camera_id, [])
            try:
                for frame in frames:
                    f2 = frame.copy()
                    if stored:
                        self._draw_boxes(f2, stored)
                    out.write(f2)
            finally:
                out.release()

            size_mb = os.path.getsize(video_path) / 1_048_576 if os.path.exists(video_path) else 0
            print(f"Clip saved: {os.path.basename(video_path)} ({len(frames)} frames, {size_mb:.1f} MB)")
        except Exception as e:
            print(f"Clip save error: {e}")
            traceback.print_exc()

    # ── Location ────────────────────────────────────────────────────────

    async def _get_location(self, conn, camera_id: int) -> str:
        try:
            ip  = requests.get('https://api.ipify.org', timeout=2).text
            geo = requests.get(f'http://ip-api.com/json/{ip}', timeout=3).json()
            if geo.get('status') == 'success':
                lat, lon = geo.get('lat'), geo.get('lon')
                if lat and lon:
                    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
                    if key:
                        r = requests.get(
                            f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={key}",
                            timeout=5,
                        ).json()
                        if r.get('status') == 'OK' and r.get('results'):
                            return r['results'][0].get('formatted_address', '')
                city, region = geo.get('city', ''), geo.get('regionName', '')
                return f"{city}, {region}".strip(', ') or geo.get('country', 'Unknown')
        except Exception:
            pass
        try:
            row = await conn.fetchval("SELECT location FROM cameras WHERE id=$1", camera_id)
            if row:
                return str(row).strip()
        except Exception:
            pass
        return "Unknown"

    # ── Box / NMS utilities ─────────────────────────────────────────────

    def _iou(self, a, b) -> float:
        xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
        xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / union if union else 0.0

    def _boxes_overlap_enough(self, a, b, iou_thr=0.05, contain_thr=0.4) -> bool:
        if self._iou(a, b) > iou_thr:
            return True
        xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
        xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
        if xi2 <= xi1 or yi2 <= yi1:
            return False
        inter = (xi2 - xi1) * (yi2 - yi1)
        small = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
        return small > 0 and inter / small >= contain_thr

    def _apply_nms(self, dets, iou_thr=0.3, person_iou_thr=0.1):
        by_cls = {}
        for d in dets:
            by_cls.setdefault(d['class'], []).append(d)
        out = []
        for cls, items in by_cls.items():
            thr   = person_iou_thr if cls == 'person' else iou_thr
            items = sorted(items, key=lambda x: x['conf'], reverse=True)
            kept  = []
            while items:
                best = items.pop(0)
                kept.append(best)
                b    = best['bbox']
                bc   = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                avg_sz = ((b[2] - b[0]) + (b[3] - b[1])) / 2
                ct   = avg_sz * 0.3
                items = [
                    d for d in items
                    if self._iou(b, d['bbox']) < thr and
                    ((bc[0] - (d['bbox'][0]+d['bbox'][2])/2)**2 +
                     (bc[1] - (d['bbox'][1]+d['bbox'][3])/2)**2) ** 0.5 > ct
                ]
            out.extend(kept)
        return out

    def _deduplicate(self, dets: list) -> list:
        by_cls = {}
        for d in dets:
            by_cls.setdefault(d.get("class", ""), []).append(d)
        out = []
        for items in by_cls.values():
            items = sorted(items, key=lambda x: x.get("confidence", 0), reverse=True)
            kept  = []
            for d in items:
                if not any(self._boxes_overlap_enough(d["bbox"], k["bbox"]) for k in kept):
                    kept.append(d)
            out.extend(kept)
        return out

    # ── Video stream ────────────────────────────────────────────────────

    def get_video_stream(self, camera_id: int):
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Waiting for detection to start...", (50, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        _, ph_buf = cv2.imencode('.jpg', placeholder)
        ph_bytes  = ph_buf.tobytes()
        last_good = None

        while True:
            frame = None
            lock  = self.frame_locks.get(camera_id)
            if lock:
                try:
                    with lock:
                        frame = self.latest_frames.get(camera_id)
                        if frame is not None:
                            last_good = frame.copy()
                except Exception:
                    frame = last_good
            if frame is None:
                frame = last_good

            if frame is None:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + ph_bytes + b'\r\n'
                time.sleep(0.1)
                continue

            try:
                disp = cv2.resize(frame, (640, 360))
            except Exception:
                disp = frame
            ok, buf = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
            time.sleep(1.0 / 30)
