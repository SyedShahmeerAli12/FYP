from ultralytics import YOLO
import cv2
import numpy as np
import asyncio
import os
from datetime import datetime
from collections import deque
from typing import Optional, Dict
import threading
import time
import torch
import requests

# Workaround for DFLoss compatibility issue with models trained on newer Ultralytics versions
import ultralytics.utils.loss as loss_module
if not hasattr(loss_module, 'DFLoss'):
    class DFLoss(torch.nn.Module):
        """Dummy DFLoss class for loading models trained with custom loss"""
        def __init__(self, *args, **kwargs):
            super().__init__()
    loss_module.DFLoss = DFLoss
import json

class DetectionService:
    def __init__(self, database):
        self.database = database
        self.model = None  # Main model for Person + Cigarette (best.pt)
        self.smoke_model = None  # Smoke detection model (11k.pt)
        self.device = 'cpu'  # Will be set to 'cuda' if GPU is available
        self.active_detections = {}  # {camera_id: detection_thread}
        self.video_buffers = {}  # {camera_id: deque of frames}
        self.frame_times = {}  # {camera_id: deque of timestamps}
        self.recording_states = {}  # {camera_id: {"is_recording": bool, "start_time": float}}
        self.latest_frames = {}  # {camera_id: latest_frame} - for streaming
        self.frame_locks = {}  # {camera_id: threading.Lock} - for thread safety
        self.latest_detections = {}  # {camera_id: bbox_data} - store detection results (YOLO runs only once)
        self.smoke_detections_buffer = {}  # {camera_id: deque of smoke detections} - keep smoke detections across frames
        self.detection_threshold = 0.70  # For Person (Person avg: ~86.5%)
        self.cigarette_threshold = 0.50  # For Cigarette (lowered from 0.70 to catch cigarettes earlier)
        self.display_threshold = 0.50  # Only show detections above 50% on stream
        self.buffer_duration = 5  # seconds before detection
        self.clip_duration = 15  # total clip duration (5s before + 5s during + 5s after)
        self.fps = 30  # frames per second
        # Limit concurrent video saves to 2 (Solution 3)
        self.save_semaphore = threading.Semaphore(2)
        # Tiny epsilon to avoid float edge cases around threshold comparisons
        self._thr_eps = 1e-6
        
        # Project root and paths (run from project root)
        self._root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._clips_dir = os.path.join(self._root, "clips")
        self._weights_dir = os.path.join(self._root, "weights")
        os.makedirs(self._clips_dir, exist_ok=True)
        
        # Load models
        self.load_model()
        self.load_smoke_model()

    def load_smoke_model(self):
        """Load smoke detection model (11k.pt)"""
        try:
            smoke_model_path = os.path.join(self._weights_dir, "11k.pt")
            if not os.path.exists(smoke_model_path):
                print(f"⚠️ Smoke model not found: {smoke_model_path} - smoke detection will be disabled")
                return
            
            self.smoke_model = YOLO(smoke_model_path)
            print(f"✅ Loaded smoke model: {smoke_model_path}")
            print(f"   Classes: {list(self.smoke_model.names.values())}")
            print(f"   Smoke model will run on every frame independently")
            
            # Move to same device as main model
            if self.device.startswith("cuda"):
                try:
                    self.smoke_model.to(self.device)
                    print(f"✅ Smoke model on GPU")
                except Exception as e:
                    print(f"⚠️ Smoke model GPU error, using CPU: {e}")
            else:
                print("⚠️ Smoke model using CPU")
        except Exception as e:
            print(f"⚠️ Error loading smoke model: {e}")
            self.smoke_model = None
    
    def _get_device(self):
        """Choose device: prefer GPU if available or USE_GPU=1 in .env. Print diagnostics if CPU."""
        use_gpu_env = os.environ.get("USE_GPU", "").strip().lower() in ("1", "true", "yes")
        device_env = os.environ.get("PYTORCH_DEVICE", "").strip().lower()
        if device_env in ("cuda", "cuda:0", "gpu"):
            use_gpu_env = True
        # Diagnose PyTorch CUDA
        cuda_available = torch.cuda.is_available()
        print(f"   PyTorch {torch.__version__} | CUDA available: {cuda_available}")
        if cuda_available:
            try:
                name = torch.cuda.get_device_name(0)
                mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"   GPU: {name} ({mem_gb:.2f} GB)")
                return "cuda:0"
            except Exception as e:
                print(f"   GPU init failed: {e}")
        if use_gpu_env and not cuda_available:
            print("   USE_GPU or PYTORCH_DEVICE=cuda set but PyTorch has no CUDA. Install GPU build:")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return "cpu"

    def load_model(self):
        """Load YOLO model"""
        try:
            # Try best.pt first (new trained model)
            model_path = os.path.join(self._weights_dir, "best.pt")
            if not os.path.exists(model_path):
                # Try alternative paths
                if os.path.exists(os.path.join(self._weights_dir, "smoke_best.pt")):
                    model_path = os.path.join(self._weights_dir, "smoke_best.pt")
                elif os.path.exists(os.path.join(self._weights_dir, "best (1).pt")):
                    model_path = os.path.join(self._weights_dir, "best (1).pt")
                elif os.path.exists(os.path.join(self._weights_dir, "best(1).pt")):
                    model_path = os.path.join(self._weights_dir, "best(1).pt")
                else:
                    raise FileNotFoundError("Model file not found: best.pt or alternatives in weights/")
            
            self.model = YOLO(model_path)
            print(f"✅ Loaded model: {model_path}")
            print(f"   Classes: {list(self.model.names.values())}")
            
            # Set device (GPU if PyTorch has CUDA)
            self.device = self._get_device()
            if self.device.startswith("cuda"):
                try:
                    self.model.to(self.device)
                    device_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                    print(f"✅ Model loaded on GPU: {device_name} ({gpu_memory:.2f} GB)")
                    _ = torch.zeros(1, device=self.device)  # verify
                except Exception as e:
                    self.device = "cpu"
                    print(f"⚠️ GPU error, using CPU: {e}")
            else:
                print("⚠️ GPU not available, using CPU (video may lag). For GPU: install PyTorch with CUDA.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    # (Smoke model loader removed)
    
    async def start_detection(self, camera_id: int, camera_info: Dict, websocket_manager):
        """Start detection for a camera"""
        if camera_id in self.active_detections:
            return False
        
        # Initialize buffers
        buffer_size = int(self.buffer_duration * self.fps)
        self.video_buffers[camera_id] = deque(maxlen=buffer_size)
        self.frame_times[camera_id] = deque(maxlen=buffer_size)
        self.recording_states[camera_id] = {"is_recording": False, "start_time": None}
        self.latest_frames[camera_id] = None
        self.frame_locks[camera_id] = threading.Lock()
        
        
        # Start detection thread
        thread = threading.Thread(
            target=self._detection_loop,
            args=(camera_id, camera_info, websocket_manager),
            daemon=True
        )
        thread.start()
        
        self.active_detections[camera_id] = thread
        return True
    
    async def stop_detection(self, camera_id: int):
        """Stop detection for a camera"""
        if camera_id not in self.active_detections:
            return False
        
        # Signal to stop (will be checked in detection loop)
        if camera_id in self.recording_states:
            self.recording_states[camera_id]["stop"] = True
        
        # Wait a bit for thread to finish
        await asyncio.sleep(1)
        
        # Cleanup
        if camera_id in self.active_detections:
            del self.active_detections[camera_id]
        if camera_id in self.video_buffers:
            del self.video_buffers[camera_id]
        if camera_id in self.frame_times:
            del self.frame_times[camera_id]
        if camera_id in self.recording_states:
            del self.recording_states[camera_id]
        if camera_id in self.latest_frames:
            del self.latest_frames[camera_id]
        if camera_id in self.frame_locks:
            del self.frame_locks[camera_id]
        
        return True
    
    def is_detection_running(self, camera_id: int) -> bool:
        """Check if detection is running for a camera"""
        return camera_id in self.active_detections
    
    def _detection_loop(self, camera_id: int, camera_info: Dict, websocket_manager):
        """Main detection loop running in separate thread"""
        camera_source = camera_info["source"]
        
        # Check if camera_source is RTSP stream (starts with "rtsp://")
        cap = None
        is_rtsp = isinstance(camera_source, str) and camera_source.startswith("rtsp://")
        
        if is_rtsp:
            # RTSP stream - use FFMPEG backend with optimized settings
            print(f"📹 Opening RTSP stream: {camera_source}")
            # Retry RTSP connection up to 3 times (RTSP can be slow to connect)
            max_initial_retries = 3
            for retry in range(max_initial_retries):
                try:
                    # For RTSP, we need to use FFMPEG backend
                    cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
                    if cap.isOpened():
                        # CRITICAL: Set buffer size to 1 to reduce latency (prevents lag)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        # Set FPS to match stream (helps with sync)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        # Give RTSP a moment to initialize (RTSP streams need time to connect)
                        time.sleep(0.5)
                        # Test if we can read a frame (try multiple times for RTSP)
                        test_ret = False
                        test_frame = None
                        for test_attempt in range(5):  # Try reading frame up to 5 times
                            test_ret, test_frame = cap.read()
                            if test_ret and test_frame is not None:
                                break
                            time.sleep(0.2)  # Wait a bit between attempts
                        
                        if test_ret and test_frame is not None:
                            print(f"✅ RTSP stream opened successfully (attempt {retry + 1}/{max_initial_retries})")
                            print(f"   Resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
                            break  # Success, exit retry loop
                        else:
                            print(f"⚠️ RTSP stream opened but can't read frames (attempt {retry + 1}/{max_initial_retries})")
                            if cap:
                                cap.release()
                            cap = None
                            if retry < max_initial_retries - 1:
                                print(f"   Retrying in 2 seconds...")
                                time.sleep(2)
                    else:
                        print(f"❌ Failed to open RTSP stream (attempt {retry + 1}/{max_initial_retries})")
                        if retry < max_initial_retries - 1:
                            print(f"   Retrying in 2 seconds...")
                            time.sleep(2)
                except Exception as e:
                    print(f"❌ Error opening RTSP stream (attempt {retry + 1}/{max_initial_retries}): {e}")
                    if cap:
                        try:
                            cap.release()
                        except:
                            pass
                    cap = None
                    if retry < max_initial_retries - 1:
                        print(f"   Retrying in 2 seconds...")
                        time.sleep(2)
        else:
            # Local camera (webcam) - try different backends
            # DirectShow works on this system (tested), so try it first
            backends_to_try = [
                (camera_source, cv2.CAP_DSHOW),  # DirectShow (works on this system)
                (camera_source, cv2.CAP_MSMF),   # Media Foundation
                (camera_source, cv2.CAP_ANY),    # Default/Any
            ]
            
            # Try different backends for camera
            for source, backend in backends_to_try:
                try:
                    cap = cv2.VideoCapture(source, backend)
                    if cap.isOpened():
                        # Test if we can actually read a frame (like test_camera.py does)
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            print(f"✅ Camera {source} opened with backend {backend}")
                            break
                        else:
                            # Can't read frames, try next backend
                            if cap:
                                cap.release()
                            cap = None
                except Exception as e:
                    # Backend failed, try next one
                    if cap:
                        try:
                            cap.release()
                        except:
                            pass
                    cap = None
        
        # If RTSP stream failed or local camera not opened, try alternatives
        if not cap or not cap.isOpened():
            if isinstance(camera_source, str) and camera_source.startswith("rtsp://"):
                print(f"❌ Error: Cannot access RTSP stream for camera {camera_id}")
                print("   Troubleshooting:")
                print("   1. Check RTSP URL is correct")
                print("   2. Check network connection")
                print("   3. Verify camera credentials")
            else:
                print(f"⚠️ Camera {camera_source} not available, trying alternatives...")
                for alt_index in [1, 2]:
                    for source, backend in [(alt_index, cv2.CAP_MSMF), (alt_index, cv2.CAP_ANY)]:
                        try:
                            cap = cv2.VideoCapture(source, backend)
                            if cap.isOpened():
                                ret, test_frame = cap.read()
                                if ret and test_frame is not None:
                                    print(f"✅ Using camera index {alt_index} with backend {backend}")
                                    camera_source = alt_index
                                    break
                            else:
                                if cap:
                                    cap.release()
                                    cap = None
                        except:
                            if cap:
                                cap.release()
                                cap = None
                    if cap and cap.isOpened():
                        break
                
                if not cap or not cap.isOpened():
                    print(f"❌ Error: Cannot access any camera for camera {camera_id}")
                    print("   Troubleshooting:")
                    print("   1. Check if camera is connected")
                    print("   2. Close other apps using the camera (Zoom, Teams, etc.)")
                    print("   3. Check Windows camera permissions")
                return
        
        # Set camera properties (with error handling)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except:
            pass  # Some cameras don't support setting resolution
        
        try:
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        except:
            pass  # Some cameras don't support setting FPS - that's OK
        
        print(f"✅ Detection started for camera {camera_id}")
        
        frame_count = 0
        last_detection_time = 0
        cooldown_period = 20  # seconds between detections to avoid duplicates (increased to prevent same event duplicates)
        is_processing_detection = False  # Flag to prevent duplicate processing
        
        try:
            frame_read_errors = 0
            max_errors = 200 if is_rtsp else 30  # RTSP streams are less reliable, allow more errors (increased)
            frame_skip_count = 0  # For RTSP: skip frames if buffer is full (reduce lag)
            reconnect_attempts = 0
            max_reconnect_attempts = 5  # Maximum reconnection attempts
            
            while camera_id in self.active_detections:
                ret, frame = cap.read()
                if not ret or frame is None:
                    frame_read_errors += 1
                    
                    # For RTSP: Try to reconnect if connection lost (less aggressive - wait longer)
                    if is_rtsp and frame_read_errors > 100 and reconnect_attempts < max_reconnect_attempts:
                        if frame_read_errors % 100 == 0:  # Try reconnect every 100 failed frames (~3 seconds at 30fps)
                            reconnect_attempts += 1
                            print(f"🔄 Camera {camera_id} - RTSP connection lost, attempting reconnect {reconnect_attempts}/{max_reconnect_attempts}...")
                            try:
                                if cap:
                                    cap.release()
                                # Keep last good frame visible during reconnection (don't clear it)
                                time.sleep(1)  # Reduced wait time (was 2s) for faster recovery
                                cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
                                if cap.isOpened():
                                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                    cap.set(cv2.CAP_PROP_FPS, 30)
                                    time.sleep(0.3)  # Reduced initialization wait (was 0.5s)
                                    # Try reading frame multiple times (but faster)
                                    test_ret = False
                                    test_frame = None
                                    for test_attempt in range(3):  # Reduced attempts (was 5)
                                        test_ret, test_frame = cap.read()
                                        if test_ret and test_frame is not None:
                                            # Immediately update UI with new frame
                                            try:
                                                if camera_id in self.frame_locks:
                                                    with self.frame_locks[camera_id]:
                                                        self.latest_frames[camera_id] = test_frame.copy()
                                            except:
                                                pass
                                            break
                                        time.sleep(0.1)  # Reduced wait (was 0.2s)
                                    
                                    if test_ret and test_frame is not None:
                                        print(f"✅ Camera {camera_id} - RTSP reconnected successfully!")
                                        frame_read_errors = 0  # Reset error count
                                        reconnect_attempts = 0  # Reset reconnect attempts
                                        continue
                                    else:
                                        print(f"⚠️ Camera {camera_id} - Reconnected but can't read frames")
                                else:
                                    print(f"⚠️ Camera {camera_id} - Failed to reconnect")
                            except Exception as reconnect_err:
                                print(f"❌ Camera {camera_id} - Reconnection error: {reconnect_err}")
                    
                    if frame_read_errors > max_errors:
                        if is_rtsp:
                            print(f"❌ Camera {camera_id} stopped responding after {frame_read_errors} attempts")
                            print(f"   RTSP stream may be down or network issue")
                            print(f"   Attempted {reconnect_attempts} reconnections")
                        else:
                            print(f"❌ Camera {camera_id} stopped responding after {frame_read_errors} attempts")
                        break
                    
                    # Progressive backoff: wait longer as errors accumulate
                    wait_time = min(0.05 * (1 + frame_read_errors // 10), 0.5) if is_rtsp else 0.1
                    time.sleep(wait_time)
                    continue
                
                frame_read_errors = 0  # Reset on successful read
                reconnect_attempts = 0  # Reset reconnect attempts on successful read
                
                # For RTSP: Skip frames if buffer is getting full (reduce lag)
                if is_rtsp and camera_id in self.video_buffers:
                    buffer_size = len(self.video_buffers[camera_id])
                    if buffer_size > 30:  # If buffer has more than 1 second of frames
                        frame_skip_count += 1
                        if frame_skip_count < 2:  # Skip every other frame when buffer is full
                            continue
                        frame_skip_count = 0
                
                current_time = time.time()
                frame_count += 1
                
                # Add frame to buffer (single copy for efficiency)
                frame_copy = frame.copy()
                
                # CRITICAL: Update latest frame IMMEDIATELY after successful read (before processing)
                # This ensures UI always has fresh frames even during heavy processing
                try:
                    if camera_id in self.frame_locks:
                        with self.frame_locks[camera_id]:
                            self.latest_frames[camera_id] = frame_copy
                except:
                    pass  # Skip if lock unavailable
                self.video_buffers[camera_id].append(frame_copy)
                self.frame_times[camera_id].append(current_time)
                
                # Use same frame for display (no extra copy needed)
                display_frame = frame_copy
                
                # Run detection on EVERY frame (like simple script for smooth video)
                bbox_data = []
                results = []
                smoke_results = []
                
                if self.model is not None:
                    # Run YOLO detection on GPU (if available) - every frame for smoothness
                    try:
                        # Use stricter NMS parameters to reduce duplicates
                        # iou: IoU threshold for NMS (lower = more aggressive)
                        # conf: confidence threshold (lowered to 0.20 to catch more detections, we filter at 70% later)
                        # max_det: maximum detections per image (increased to catch more)
                        # GPU is used automatically via self.device
                        results = self.model(frame, device=self.device, verbose=False, 
                                           iou=0.3, conf=0.20, max_det=20)
                    except Exception as det_err:
                        print(f"⚠️ Detection error: {det_err}")
                        results = []
                
                # Run smoke detection if smoke model is available (runs on EVERY frame independently)
                if self.smoke_model is not None:
                    try:
                        # Match test_smoke_model.py: iou=0.3, conf=0.20, max_det=20 (low threshold to get all detections)
                        smoke_results = self.smoke_model(frame, device=self.device, verbose=False,
                                                         iou=0.3, conf=0.20, max_det=20)
                        # Debug: Show ALL smoke detections from model (even low confidence)
                        if smoke_results:
                            has_detections = False
                            for r in smoke_results:
                                if len(r.boxes) > 0:
                                    has_detections = True
                                    smoke_detections_raw = []
                                    for box in r.boxes:
                                        conf = float(box.conf[0])
                                        cls = int(box.cls[0])
                                        class_name = self.smoke_model.names[cls]
                                        smoke_detections_raw.append(f"{class_name}:{conf:.2f}")
                                    if smoke_detections_raw:
                                        # Log ALL smoke detections from 11k.pt model (not just every second) - match test_smoke_model.py
                                        print(f"💨 Camera {camera_id} - Smoke model (11k.pt) RAW output: {', '.join(smoke_detections_raw)}")
                            # Log if smoke model returns empty results
                            if not has_detections and frame_count % 300 == 0:  # Every 10 seconds
                                print(f"💨 Camera {camera_id} - Smoke model (11k.pt) running but NO detections found (threshold: 0.20)")
                        else:
                            if frame_count % 300 == 0:  # Every 10 seconds
                                print(f"💨 Camera {camera_id} - Smoke model returned empty results")
                        
                        # Debug: log if smoke model is running (first few frames only)
                        if frame_count < 5:
                            print(f"💨 Smoke model running on frame {frame_count}...")
                        elif frame_count == 30:  # After 1 second, confirm it's still running
                            print(f"💨 Smoke model confirmed running (frame {frame_count})")
                    except Exception as smoke_err:
                        print(f"⚠️ Smoke detection error: {smoke_err}")
                        import traceback
                        traceback.print_exc()
                        smoke_results = []
                else:
                    smoke_results = []
                    if frame_count % 300 == 0:  # Log every 10 seconds
                        print(f"⚠️ Camera {camera_id} - Smoke model is None (not loaded)")
                
                # Process results: Apply NMS, filter duplicates, and check Person + Cigarette
                # Reset all detection variables for this frame
                person_detected = False
                person_conf = 0
                person_bbox = None
                cigarette_detected = False
                cigarette_conf = 0
                cigarette_bbox = None
                
                if results:
                    # Collect all detections first
                    all_detections = []
                    raw_detections_log = []  # For logging all raw detections
                    for r in results:
                        for box in r.boxes:
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls = int(box.cls[0])
                            class_name = self.model.names[cls]
                            all_detections.append({
                                'class': class_name,
                                'conf': conf,
                                'bbox': [x1, y1, x2, y2],
                                'area': (x2 - x1) * (y2 - y1)
                            })
                            raw_detections_log.append(f"{class_name}:{conf:.2f}")
                    
                    # Log all raw detections from model (before NMS) - but filter out Smoke (we use 11k.pt for smoke)
                    filtered_log = [d for d in raw_detections_log if not d.lower().startswith('smoke:')]
                    if len(filtered_log) > 0 and frame_count % 30 == 0:  # Log every 30 frames (1 second at 30 FPS)
                        print(f"📊 Camera {camera_id} - Raw detections (main model, Smoke ignored): {', '.join(filtered_log)}")
                    
                    # Apply NMS (Non-Maximum Suppression) to filter duplicates
                    # Use lower IoU threshold (0.3) to catch more overlapping boxes
                    # Apply very aggressive NMS for Person class (0.1) to eliminate duplicates
                    filtered_detections = self._apply_nms(all_detections, iou_threshold=0.3, person_iou_threshold=0.1)
                    
                    # Log filtered detections
                    if len(filtered_detections) > 0 and frame_count % 30 == 0:
                        filtered_log = [f"{d['class']}:{d['conf']:.2f}" for d in filtered_detections]
                        print(f"✅ Camera {camera_id} - After NMS: {', '.join(filtered_log)}")
                    
                    # Process filtered detections - track Person and Cigarette separately
                    person_candidates = []
                    cigarette_candidates = []
                    # Detections to DISPLAY on stream (independent of violation logic)
                    bbox_data_display = []
                    
                    # First pass: collect all Person and Cigarette detections
                    for det in filtered_detections:
                        class_name = det['class']
                        conf = det['conf']
                        x1, y1, x2, y2 = det['bbox']
                        class_lower = class_name.lower()
                        
                        # ONLY process Person and Cigarette - ignore all other classes (including Smoke from main model)
                        # CRITICAL: Ignore "Smoke" from main model - we ONLY use 11k.pt for smoke detection
                        if class_lower not in ['person', 'cigarette']:
                            continue  # Skip all other classes (including Smoke, Vape, etc.)
                        
                        if class_lower == 'person':
                            person_candidates.append((conf, [x1, y1, x2, y2]))
                        elif class_lower == 'cigarette':
                            cigarette_candidates.append((conf, [x1, y1, x2, y2]))

                    # Show Person/Cigarette boxes when they form a valid pair (even without smoke)
                    # This allows users to see detections, but violations only happen when smoke is also detected
                    
                    # NEW LOGIC: Match Person + Cigarette pairs spatially (cigarette must be near person)
                    # This allows multiple people with cigarettes to be detected
                    valid_pairs = []  # List of (person_conf, person_bbox, cig_conf, cig_bbox)
                    used_cigarettes = set()  # Track which cigarettes have been matched
                    
                    # For each Person, find nearby Cigarettes
                    for person_conf, person_bbox in person_candidates:
                        if person_conf + self._thr_eps < self.detection_threshold:
                            continue  # Skip low confidence persons
                        
                        # Find cigarettes near this person
                        person_center_x = (person_bbox[0] + person_bbox[2]) / 2
                        person_center_y = (person_bbox[1] + person_bbox[3]) / 2
                        person_width = person_bbox[2] - person_bbox[0]
                        person_height = person_bbox[3] - person_bbox[1]
                        person_size = max(person_width, person_height)
                        
                        # INCREASED search radius to catch cigarettes better (2.5x person size)
                        # This ensures cigarettes are detected even if slightly further from person
                        search_radius = person_size * 2.5
                        
                        # Also check if cigarette overlaps with person bbox (expanded)
                        person_bbox_expanded = [
                            person_bbox[0] - person_width * 0.3,  # Expand left
                            person_bbox[1] - person_height * 0.3,  # Expand top
                            person_bbox[2] + person_width * 0.3,  # Expand right
                            person_bbox[3] + person_height * 0.3  # Expand bottom
                        ]
                        
                        best_cig_match = None
                        best_cig_conf = 0
                        best_cig_bbox = None
                        
                        for cig_idx, (cig_conf, cig_bbox) in enumerate(cigarette_candidates):
                            if cig_conf + self._thr_eps < self.cigarette_threshold:
                                continue  # Skip low confidence cigarettes
                            
                            if cig_idx in used_cigarettes:
                                continue  # Skip already matched cigarettes
                            
                            # Check if cigarette is near this person (center distance)
                            cig_center_x = (cig_bbox[0] + cig_bbox[2]) / 2
                            cig_center_y = (cig_bbox[1] + cig_bbox[3]) / 2
                            distance = ((person_center_x - cig_center_x)**2 + (person_center_y - cig_center_y)**2)**0.5
                            
                            # Check if cigarette overlaps with expanded person bbox
                            cig_overlaps = (cig_center_x >= person_bbox_expanded[0] and 
                                          cig_center_x <= person_bbox_expanded[2] and
                                          cig_center_y >= person_bbox_expanded[1] and 
                                          cig_center_y <= person_bbox_expanded[3])
                            
                            # If cigarette is within search radius OR overlaps with person, it's a match
                            if distance <= search_radius or cig_overlaps:
                                # Keep the best matching cigarette for this person
                                if cig_conf > best_cig_conf:
                                    best_cig_match = cig_idx
                                    best_cig_conf = cig_conf
                                    best_cig_bbox = cig_bbox
                        
                        # If we found a matching cigarette for this person, add the pair
                        if best_cig_match is not None:
                            valid_pairs.append((person_conf, person_bbox, best_cig_conf, best_cig_bbox))
                            used_cigarettes.add(best_cig_match)  # Mark cigarette as used
                    
                    # Process valid pairs - can have multiple people with cigarettes
                    if valid_pairs:
                        # Sort by combined confidence (highest first)
                        valid_pairs.sort(key=lambda x: (x[0] + x[2]) / 2.0, reverse=True)
                        
                        # Take the best pair for violation detection (but store all for display)
                        best_pair = valid_pairs[0]
                        person_conf, person_bbox, cigarette_conf, cigarette_bbox = best_pair
                        person_detected = True
                        cigarette_detected = True
                        
                        # Calculate combined confidence for best pair
                        combined_conf = (person_conf + cigarette_conf) / 2.0
                        
                        # Store ALL valid pairs for display (multiple people with cigarettes)
                        bbox_data = []
                        # Add Person and Cigarette boxes to display (even without smoke)
                        for p_conf, p_bbox, c_conf, c_bbox in valid_pairs:
                            bbox_data.append({"class": "Person", "confidence": p_conf, "bbox": p_bbox})
                            bbox_data.append({"class": "Cigarette", "confidence": c_conf, "bbox": c_bbox})
                            # Also add to display so boxes show on stream
                            bbox_data_display.append({"class": "Person", "confidence": p_conf, "bbox": p_bbox})
                            bbox_data_display.append({"class": "Cigarette", "confidence": c_conf, "bbox": c_bbox})
                    else:
                        # No valid pairs found
                        person_detected = False
                        cigarette_detected = False
                        person_conf = 0
                        person_bbox = None
                        cigarette_conf = 0
                        cigarette_bbox = None
                        combined_conf = 0
                        bbox_data = []
                    
                    # ALSO show Person and Cigarette boxes separately if they're above display_threshold
                    # This allows boxes to appear even when they don't form a valid pair (for violation detection)
                    for p_conf, p_bbox in person_candidates:
                        if p_conf >= self.display_threshold:  # Show if above 50%
                            # Check if this person box is already in display (from valid_pairs)
                            person_already_added = any(
                                d.get("class") == "Person" and 
                                abs(d.get("bbox", [0,0,0,0])[0] - p_bbox[0]) < 5 and
                                abs(d.get("bbox", [0,0,0,0])[1] - p_bbox[1]) < 5
                                for d in bbox_data_display
                            )
                            if not person_already_added:
                                bbox_data_display.append({"class": "Person", "confidence": p_conf, "bbox": p_bbox})
                    
                    for c_conf, c_bbox in cigarette_candidates:
                        if c_conf >= self.display_threshold:  # Show if above 50%
                            # Check if this cigarette box is already in display (from valid_pairs)
                            cig_already_added = any(
                                d.get("class") == "Cigarette" and 
                                abs(d.get("bbox", [0,0,0,0])[0] - c_bbox[0]) < 5 and
                                abs(d.get("bbox", [0,0,0,0])[1] - c_bbox[1]) < 5
                                for d in bbox_data_display
                            )
                            if not cig_already_added:
                                bbox_data_display.append({"class": "Cigarette", "confidence": c_conf, "bbox": c_bbox})
                    
                    # Detailed logging for detection status
                    if frame_count % 30 == 0:  # Log every 1 second
                        log_parts = []
                        if person_candidates:
                            max_person = max(person_candidates, key=lambda x: x[0])
                            person_above_threshold = (max_person[0] + self._thr_eps) >= self.detection_threshold
                            if person_detected:
                                log_parts.append(f"Person: {max_person[0]:.3f} ✅")
                            elif person_above_threshold:
                                log_parts.append(f"Person: {max_person[0]:.3f} ✅ (no pair)")
                            else:
                                log_parts.append(f"Person: {max_person[0]:.3f} ❌ (<{self.detection_threshold:.2f})")
                        else:
                            log_parts.append("Person: NOT DETECTED")
                        
                        if cigarette_candidates:
                            max_cig = max(cigarette_candidates, key=lambda x: x[0])
                            cig_above_threshold = (max_cig[0] + self._thr_eps) >= self.cigarette_threshold
                            if cigarette_detected:
                                log_parts.append(f"Cigarette: {max_cig[0]:.3f} ✅")
                            elif cig_above_threshold:
                                log_parts.append(f"Cigarette: {max_cig[0]:.3f} ✅ (no pair)")
                            else:
                                log_parts.append(f"Cigarette: {max_cig[0]:.3f} ❌ (<{self.cigarette_threshold:.2f})")
                        else:
                            log_parts.append("Cigarette: NOT DETECTED")
                        
                        print(f"🔍 Camera {camera_id} Detection Status: {' | '.join(log_parts)}")
                        
                        # Log why violation is/isn't triggered (all three required)
                        if person_detected and cigarette_detected and smoke_detected:
                            num_pairs = len(valid_pairs) if 'valid_pairs' in locals() and valid_pairs else 0
                            print(f"✅ Camera {camera_id} - Person + Cigarette + Smoke DETECTED! Combined: {combined_conf:.2f} ({num_pairs} pair(s))")
                        elif person_detected and cigarette_detected:
                            print(f"⚠️ Camera {camera_id} - Missing: SMOKE (Person✅ Cigarette✅ Smoke❌) - No violation")
                        elif person_candidates:
                            print(f"⚠️ Camera {camera_id} - Missing: CIGARETTE or SMOKE (Person✅)")
                        elif cigarette_candidates:
                            print(f"⚠️ Camera {camera_id} - Missing: PERSON or SMOKE (Cigarette✅)")
                        else:
                            print(f"❌ Camera {camera_id} - No valid detections")
                
                # Check for smoke detection (run independently, not conditional on person+cigarette)
                smoke_detected = False
                smoke_conf = 0
                smoke_bbox = None
                all_smoke_detections = []  # Store smoke detections from THIS frame only
                
                # Initialize smoke detections buffer for this camera if not exists (keep last 0.5 seconds = 15 frames)
                if camera_id not in self.smoke_detections_buffer:
                    self.smoke_detections_buffer[camera_id] = deque(maxlen=15)  # 0.5 seconds at 30 FPS
                
                # Process smoke results independently (smoke model runs on every frame)
                # Match test_smoke_model.py logic EXACTLY (lines 181-209)
                if self.smoke_model is not None and smoke_results:
                    for r in smoke_results:
                        for box in r.boxes:
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls = int(box.cls[0])
                            class_name = self.smoke_model.names[cls]
                            
                            # Match test_smoke_model.py line 190: Check if it's smoke
                            class_lower = str(class_name).lower()
                            is_smoke = 'smoke' in class_lower or 'mouth' in class_lower or str(class_name) == '1' or str(class_name) == '0' or str(cls) == '0'
                            
                            # Lowered threshold to catch smoke earlier: 0.50 (50% threshold)
                            if is_smoke and conf >= 0.50:
                                # Store all smoke detections
                                smoke_det = {
                                    'conf': conf,
                                    'bbox': [x1, y1, x2, y2],
                                    'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                                    'class_name': class_name,
                                    'frame_count': frame_count,
                                    'timestamp': time.time()
                                }
                                all_smoke_detections.append(smoke_det)
                                
                                # Also add to buffer for temporal matching (store for 0.5 seconds = 15 frames)
                                # Reduced buffer time to prevent matching stale smoke with new Person+Cigarette
                                if camera_id not in self.smoke_detections_buffer:
                                    self.smoke_detections_buffer[camera_id] = deque(maxlen=15)  # 0.5 seconds at 30 FPS
                                self.smoke_detections_buffer[camera_id].append(smoke_det)
                    
                    # Log smoke detections for debugging - show ALL detections (not just every second)
                    if len(all_smoke_detections) > 0:
                        smoke_log = [f"Smoke:{d['conf']:.2f}" for d in all_smoke_detections]
                        print(f"💨 Camera {camera_id} - Smoke detected (11k.pt, ≥0.50): {', '.join(smoke_log)}")
                    elif frame_count % 300 == 0:  # Log every 10 seconds if no smoke
                        print(f"💨 Camera {camera_id} - No smoke detections found (11k.pt, threshold: 0.50)")
                
                # TEMPORAL MATCHING: Check smoke from CURRENT frame AND very recent buffer (last 0.5 seconds only)
                # Only match if smoke is in current frame OR was detected very recently (within last 0.5 seconds)
                # This prevents matching stale smoke with new Person+Cigarette
                if person_detected and cigarette_detected:
                    # Get smoke from buffer (last 0.5 seconds = 15 frames) - MUCH SHORTER to prevent stale matches
                    all_smoke_from_buffer = []
                    if camera_id in self.smoke_detections_buffer:
                        current_time = time.time()
                        for buffered_smoke in self.smoke_detections_buffer[camera_id]:
                            # Only use smoke from last 0.5 seconds (very recent)
                            if current_time - buffered_smoke['timestamp'] <= 0.5:
                                all_smoke_from_buffer.append(buffered_smoke)
                    
                    # Combine current frame smoke and buffer smoke
                    all_smoke_to_check = all_smoke_detections + all_smoke_from_buffer
                    
                    # Only log once per second to reduce spam
                    if frame_count % 30 == 0:
                        print(f"🔍 Camera {camera_id} - Person+Cigarette detected! Checking smoke: {len(all_smoke_detections)} current + {len(all_smoke_from_buffer)} from buffer = {len(all_smoke_to_check)} total")
                    
                    # Check smoke from current frame AND buffer
                    if all_smoke_to_check:
                        # Get combined area of person and cigarette
                        combined_x1 = min(person_bbox[0], cigarette_bbox[0])
                        combined_y1 = min(person_bbox[1], cigarette_bbox[1])
                        combined_x2 = max(person_bbox[2], cigarette_bbox[2])
                        combined_y2 = max(person_bbox[3], cigarette_bbox[3])
                        
                        # Expand search area very tightly (0.15x - extremely strict to reduce false matches)
                        width = combined_x2 - combined_x1
                        height = combined_y2 - combined_y1
                        search_x1 = combined_x1 - width * 0.15  # Reduced to 0.15x for very tight matching
                        search_y1 = combined_y1 - height * 0.15
                        search_x2 = combined_x2 + width * 0.15
                        search_y2 = combined_y2 + height * 0.15
                        
                        # Calculate person+cigarette center for distance check
                        person_cig_center_x = (combined_x1 + combined_x2) / 2
                        person_cig_center_y = (combined_y1 + combined_y2) / 2
                        max_distance = max(width, height) * 0.5  # Smoke must be within 50% of person+cigarette size (reduced from 80%)
                        
                        # Find smoke that overlaps with person+cigarette area (check current frame + buffer)
                        print(f"🔍 Camera {camera_id} - Checking {len(all_smoke_to_check)} smoke detections (current + buffer) against Person+Cigarette area")
                        print(f"   Person bbox: [{person_bbox[0]:.0f}, {person_bbox[1]:.0f}] to [{person_bbox[2]:.0f}, {person_bbox[3]:.0f}]")
                        print(f"   Cigarette bbox: [{cigarette_bbox[0]:.0f}, {cigarette_bbox[1]:.0f}] to [{cigarette_bbox[2]:.0f}, {cigarette_bbox[3]:.0f}]")
                        print(f"   Search area: [{search_x1:.0f}, {search_y1:.0f}] to [{search_x2:.0f}, {search_y2:.0f}]")
                        for smoke_det in all_smoke_to_check:
                            smoke_center_x, smoke_center_y = smoke_det['center']
                            
                            # Check if smoke is within search area
                            in_search_area = (search_x1 <= smoke_center_x <= search_x2 and 
                                            search_y1 <= smoke_center_y <= search_y2)
                            
                            # Calculate distance from person+cigarette center
                            distance = ((smoke_center_x - person_cig_center_x)**2 + 
                                       (smoke_center_y - person_cig_center_y)**2)**0.5
                            within_distance = distance <= max_distance
                            
                            print(f"   🔍 Smoke at [{smoke_center_x:.0f}, {smoke_center_y:.0f}] conf:{smoke_det['conf']:.2f} - Distance: {distance:.0f}px (max: {max_distance:.0f}px)")
                            
                            if in_search_area and within_distance:
                                print(f"   ✅ Smoke INSIDE search area AND within distance! Matching...")
                                if smoke_det['conf'] > smoke_conf:
                                    smoke_detected = True
                                    smoke_conf = smoke_det['conf']
                                    smoke_bbox = smoke_det['bbox']
                                    
                                    # Add Smoke box when smoke is detected (Person+Cigarette boxes already added above)
                                    # Check if smoke box already exists to avoid duplicates
                                    smoke_box_exists = any(d.get("class") == "Smoke" for d in bbox_data_display)
                                    if not smoke_box_exists:
                                        bbox_data_display.append({
                                            "class": "Smoke", 
                                            "confidence": smoke_conf, 
                                            "bbox": smoke_bbox
                                        })
                                        print(f"   ✅✅✅ Smoke box ADDED to display! bbox_data_display now has {len(bbox_data_display)} boxes")
                                    
                                    if frame_count % 30 == 0:
                                        print(f"✅ Camera {camera_id} - ALL THREE in SAME frame! Person:{person_conf:.2f} + Cigarette:{cigarette_conf:.2f} + Smoke:{smoke_conf:.2f}")
                            else:
                                if not in_search_area:
                                    print(f"   ❌ Smoke OUTSIDE search area")
                                elif not within_distance:
                                    print(f"   ❌ Smoke TOO FAR from Person+Cigarette (distance: {distance:.0f}px > max: {max_distance:.0f}px)")
                                if frame_count % 30 == 0:
                                    print(f"⚠️ Camera {camera_id} - Smoke detected but NOT matched (smoke center: [{smoke_center_x:.0f}, {smoke_center_y:.0f}], distance: {distance:.0f}px)")
                    else:
                        if frame_count % 30 == 0:
                            print(f"⚠️ Camera {camera_id} - Person+Cigarette detected but NO smoke detections (current frame or buffer)")
                
                # Also show smoke detections on stream even if person/cigarette aren't detected (for debugging)
                elif all_smoke_detections and frame_count % 30 == 0:
                    print(f"💨 Camera {camera_id} - Smoke detected but Person/Cigarette missing")
                
                # Production behavior: ONLY create violation when Person + Cigarette + Smoke ALL detected
                # If smoke is missing, NO violation should be created
                has_valid_smoking = person_detected and cigarette_detected and smoke_detected
                has_smoke_violation = has_valid_smoking  # All three must be present
                
                # Log detection status
                if frame_count % 30 == 0:  # Log every 1 second
                    if person_detected and cigarette_detected and smoke_detected:
                        print(f"🎯 Camera {camera_id} - Person✅ + Cigarette✅ + Smoke✅ (Combined: {combined_conf:.2f})")
                    elif person_detected or cigarette_detected or smoke_detected:
                        missing = []
                        if not person_detected: missing.append("Person")
                        if not cigarette_detected: missing.append("Cigarette")
                        if not smoke_detected: missing.append("Smoke")
                        print(f"⚠️ Camera {camera_id} - Missing: {', '.join(missing)} (No violation - all three required)")
                
                # bbox_data and combined_conf are already set above from valid_pairs logic
                # No need to recalculate - they're already correct
                
                # CRITICAL: Store latest detections (thread-safe) - YOLO runs ONLY ONCE per frame
                # Everything else (streaming, video saving) will reuse these results
                # Boxes disappear when detection stops (100% accurate, no persistence)
                try:
                    with self.frame_locks[camera_id]:
                        # Use display detections so Person boxes show even without a valid pair
                        if 'bbox_data_display' in locals():
                            self.latest_detections[camera_id] = bbox_data_display
                        else:
                            self.latest_detections[camera_id] = bbox_data  # Fallback
                except:
                    pass  # Skip if lock unavailable
                
                # Draw detections (Person/Cigarette)
                # Boxes will persist as long as detections continue
                if 'bbox_data_display' in locals() and bbox_data_display:
                    # Draw all boxes (Person, Cigarette)
                    for det in bbox_data_display:
                        x1, y1, x2, y2 = det["bbox"]
                        class_name = det['class']
                        conf = det['confidence']
                        label = f"{class_name} {conf:.2f}"
                        
                        # Different colors for different classes
                        if class_name == "Person":
                            color = (0, 255, 0)  # Green
                        elif class_name == "Cigarette":
                            color = (255, 165, 0)  # Orange
                        elif class_name == "Smoke":
                            color = (0, 0, 255)  # Red
                        else:
                            color = (0, 255, 0)  # Default green
                        
                        # Draw box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        # Draw label with background for better visibility
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(display_frame,
                                    (x1, y1 - label_size[1] - 5),
                                    (x1 + label_size[0], y1),
                                    color, -1)
                        cv2.putText(display_frame, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Update latest frame for streaming (thread-safe) - use display_frame with boxes
                # NOTE: We already update raw frame above, but update with boxes here for display
                # Use try/except for faster lock acquisition
                try:
                    if camera_id in self.frame_locks:
                        with self.frame_locks[camera_id]:
                            self.latest_frames[camera_id] = display_frame
                except:
                    pass  # Skip if lock unavailable (shouldn't happen but safe)
                
                # If Person + Cigarette + Smoke ALL detected, trigger IMMEDIATELY (no sustained requirement)
                # CRITICAL: Check flag to prevent duplicate processing AND recording state
                # For smoke violations, we want immediate detection when all three are present
                if (has_valid_smoking and  # All three detected (Person+Cigarette+Smoke)
                    combined_conf > self.detection_threshold and 
                    (current_time - last_detection_time) > cooldown_period and
                    not is_processing_detection and  # Prevent duplicate processing
                    not self.recording_states.get(camera_id, {}).get("is_recording", False)):  # Prevent duplicate recording
                    
                    # Set flag IMMEDIATELY to prevent duplicate triggers
                    is_processing_detection = True
                    last_detection_time = current_time
                    
                    # Detection class is always Person+Cigarette+Smoke (all three required for violation)
                    detection_class = "Person+Cigarette+Smoke"
                    # Combined confidence includes all three
                    combined_conf = (person_conf + cigarette_conf + smoke_conf) / 3.0
                    # Add smoke to bbox_data for saving
                    if smoke_bbox:
                        bbox_data.append({"class": "Smoke", "confidence": smoke_conf, "bbox": smoke_bbox})
                    
                    # Lock recording state IMMEDIATELY to prevent duplicate saves
                    if camera_id not in self.recording_states:
                        self.recording_states[camera_id] = {"is_recording": False, "start_time": 0}
                    self.recording_states[camera_id]["is_recording"] = True
                    self.recording_states[camera_id]["start_time"] = current_time
                    
                    # Lock recording state IMMEDIATELY to prevent duplicate saves
                    if camera_id not in self.recording_states:
                        self.recording_states[camera_id] = {"is_recording": False, "start_time": 0}
                    self.recording_states[camera_id]["is_recording"] = True
                    self.recording_states[camera_id]["start_time"] = current_time
                    
                    # CRITICAL: Capture timestamp at EXACT moment of detection (not when saving)
                    detection_timestamp = datetime.now()
                    
                    print(f"🚨🚨🚨 VIOLATION DETECTED: {detection_class} 🚨🚨🚨")
                    print(f"   📍 Camera: {camera_id}")
                    print(f"   👤 Person: {person_conf:.2f}")
                    print(f"   🚬 Cigarette: {cigarette_conf:.2f}")
                    if smoke_detected:
                        print(f"   💨 Smoke: {smoke_conf:.2f}")
                    print(f"   📊 Combined: {combined_conf:.2f}")
                    print(f"⏰ Detection timestamp: {detection_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                    print(f"💾 Saving video clip and inserting into database...")
                    
                    # Save to DB IMMEDIATELY (before video processing) - ONLY ONCE
                    violation_data = {
                        "camera_id": camera_id,
                        "detection_class": detection_class,
                        "confidence": combined_conf,
                        "video_path": os.path.join(self._clips_dir, f"detection_{detection_timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_cam{camera_id}_{detection_class}_{combined_conf:.2f}.mp4"),  # Placeholder, will be updated after video save
                        "timestamp": detection_timestamp,
                        "frame_count": 0,  # Will be updated after video save
                        "bbox_data": bbox_data
                    }
                    
                    # Save to DB immediately in separate thread
                    def save_db_immediately():
                        try:
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                new_loop.run_until_complete(self._save_violation_to_db_thread(violation_data, websocket_manager))
                                print(f"✅ Violation saved to DB immediately: {detection_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                            finally:
                                new_loop.close()
                        except Exception as e:
                            print(f"⚠️ Error saving to DB immediately: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    threading.Thread(target=save_db_immediately, daemon=True).start()
                    
                    # Send immediate alert to UI via WebSocket (from thread) - ONLY ONCE
                    try:
                        alert_message = "🚨 SMOKE + CIGARETTE + PERSON VIOLATION DETECTED! 🚨"
                        alert_notification = {
                            "type": "violation_alert",
                            "message": alert_message,
                            "camera_id": camera_id,
                            "detection_class": detection_class,
                            "confidence": combined_conf,
                            "timestamp": detection_timestamp.isoformat()
                        }
                        print(f"📢 Sending alert to UI: {alert_notification['message']}")
                        # Send alert in background thread (create new event loop)
                        threading.Thread(
                            target=self._send_alert_thread,
                            args=(alert_notification, websocket_manager),
                            daemon=True
                        ).start()
                    except Exception as e:
                        print(f"⚠️ Error sending alert: {e}")
                    
                    # Save video to disk IMMEDIATELY (not delayed) - start right away
                    print(f"💾 Starting immediate video save to disk...")
                    self._save_detection_clip(
                        camera_id,
                        detection_class,
                        combined_conf,
                        bbox_data,
                        websocket_manager,
                        detection_timestamp  # Pass exact detection timestamp
                    )
                    print(f"✅ Video save process started immediately")
                    
                    # Reset flag and recording state after cooldown period
                    def reset_flag_after_cooldown():
                        time.sleep(cooldown_period)  # Wait for cooldown period
                        nonlocal is_processing_detection
                        is_processing_detection = False
                        # Unlock recording state
                        if camera_id in self.recording_states:
                            self.recording_states[camera_id]["is_recording"] = False
                        print(f"✅ Cooldown complete for camera {camera_id}, ready for next detection")
                    
                    threading.Thread(target=reset_flag_after_cooldown, daemon=True).start()
                
                # Small delay to maintain FPS
                time.sleep(1.0 / self.fps)
        
        except Exception as e:
            print(f"❌ Error in detection loop for camera {camera_id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                cap.release()
            except:
                pass
            print(f"🟢 Detection stopped for camera {camera_id}")
    
    def _save_detection_clip(
        self,
        camera_id: int,
        detection_class: str,
        confidence: float,
        bbox_data: list,
        websocket_manager,
        detection_timestamp: datetime = None
    ):
        """Save 15-second video clip (5s before + 5s during + 5s after)"""
        # Use current time if timestamp not provided (backward compatibility)
        if detection_timestamp is None:
            detection_timestamp = datetime.now()
        
        # Run in separate thread to avoid blocking detection
        thread = threading.Thread(
            target=self._save_clip_thread,
            args=(camera_id, detection_class, confidence, bbox_data, websocket_manager, detection_timestamp),
            daemon=True
        )
        thread.start()
    
    def _save_clip_thread(
        self,
        camera_id: int,
        detection_class: str,
        confidence: float,
        bbox_data: list,
        websocket_manager,
        detection_timestamp: datetime
    ):
        """Thread function to save video clip - limited to 2 concurrent saves - STARTS IMMEDIATELY"""
        print(f"💾 Video save thread started for camera {camera_id} at {detection_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        # Acquire semaphore to limit concurrent saves (Solution 3)
        with self.save_semaphore:
            print(f"📹 Starting video file creation for camera {camera_id}...")
            self._save_clip_internal(camera_id, detection_class, confidence, bbox_data, websocket_manager, detection_timestamp)
            print(f"✅ Video file saved to disk for camera {camera_id}")
    
    def _save_clip_internal(
        self,
        camera_id: int,
        detection_class: str,
        confidence: float,
        bbox_data: list,
        websocket_manager,
        detection_timestamp: datetime
    ):
        """Internal function to save video clip"""
        try:
            if camera_id not in self.video_buffers or len(self.video_buffers[camera_id]) == 0:
                return
            
            # Get buffer frames (5 seconds before detection)
            buffer_frames = list(self.video_buffers[camera_id])
            
            # Collect additional frames from the ongoing detection loop (5s during + 5s after)
            # Use a more reliable method: record frames directly from latest_frames for fixed duration
            recording_frames = []
            recording_duration = 10  # 5s during + 5s after
            frames_to_record = int(recording_duration * self.fps)
            
            start_collection_time = time.time()
            last_frame_time = start_collection_time
            frame_interval = 1.0 / self.fps  # Time between frames at 30 FPS
            
            # Collect frames for exactly recording_duration seconds
            while (time.time() - start_collection_time) < recording_duration:
                # Check if detection is still running
                if camera_id not in self.active_detections:
                    break
                
                # Get latest frame from detection loop (thread-safe)
                if camera_id in self.latest_frames and camera_id in self.frame_locks:
                    try:
                        with self.frame_locks[camera_id]:
                            latest = self.latest_frames.get(camera_id)
                            if latest is not None:
                                current_time = time.time()
                                # Only add frame if enough time has passed (to maintain FPS)
                                if current_time - last_frame_time >= frame_interval:
                                    recording_frames.append(latest.copy())
                                    last_frame_time = current_time
                                    
                                    # Limit to prevent excessive memory usage
                                    if len(recording_frames) > frames_to_record:
                                        recording_frames = recording_frames[-frames_to_record:]
                    except:
                        pass  # Skip if lock unavailable
                
                # Small sleep to avoid busy waiting (but maintain frame rate)
                time.sleep(frame_interval)
            
            # If we don't have enough frames, pad with last frame
            if len(recording_frames) < frames_to_record:
                if len(recording_frames) > 0:
                    last_frame = recording_frames[-1]
                    while len(recording_frames) < frames_to_record:
                        recording_frames.append(last_frame.copy())
                elif len(buffer_frames) > 0:
                    # Use last buffer frame
                    last_frame = buffer_frames[-1]
                    recording_frames = [last_frame.copy()] * frames_to_record
                else:
                    # No frames available, skip saving
                    print(f"⚠️ No frames available for video save on camera {camera_id}")
                    return
            
            # Combine buffer + recording frames
            all_frames = buffer_frames + recording_frames
            
            if len(all_frames) == 0:
                return
            
            # Draw green boxes on all frames before saving - USE STORED DETECTIONS (NO YOLO)
            # CRITICAL: NO YOLO HERE - Use stored detections from _detection_loop (YOLO runs only once)
            frames_with_boxes = []
            # Get stored detections (thread-safe)
            try:
                with self.frame_locks[camera_id]:
                    stored_detections = self.latest_detections.get(camera_id, [])
            except:
                stored_detections = []
            
            for frame in all_frames:
                frame_with_box = frame.copy()
                
                # Draw stored detections ONLY (no YOLO, no duplicates, matches live stream exactly)
                for det in stored_detections:
                    x1, y1, x2, y2 = det["bbox"]
                    label = f"{det['class']} {det['confidence']:.2f}"
                    # Draw box
                    cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw label with background
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame_with_box,
                                (x1, y1 - label_size[1] - 5),
                                (x1 + label_size[0], y1),
                                (0, 255, 0), -1)
                    cv2.putText(frame_with_box, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                frames_with_boxes.append(frame_with_box)
            
            # Create video filename using EXACT detection timestamp (not current time)
            filename = f"detection_{detection_timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_cam{camera_id}_{detection_class}_{confidence:.2f}.mp4"
            video_path = os.path.join(self._clips_dir, filename)
            
            # Get video properties from first frame
            height, width = frames_with_boxes[0].shape[:2]
            
            # Write video using mp4v codec (more compatible, doesn't need OpenH264)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            out = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
            
            if not out.isOpened():
                print(f"❌ Error: Could not open video writer for {video_path}")
                return
            
            # Write frames with green boxes
            for frame in frames_with_boxes:
                out.write(frame)
            
            out.release()
            
            # Verify file exists on disk
            if os.path.exists(video_path):
                file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                print(f"✅ Video saved to disk: {video_path}")
                print(f"📁 File size: {file_size_mb:.2f} MB | Frames: {len(frames_with_boxes)}")
            else:
                print(f"❌ ERROR: Video file not found on disk: {video_path}")
            
            # Update DB entry with actual video path and frame count (DB entry was created immediately)
            # The DB entry was already created with placeholder path, now update it with actual path
            # Note: This is optional - the DB entry already exists with the correct timestamp
            # We could update it, but it's not critical since the placeholder path matches the actual path format
        
        except Exception as e:
            print(f"❌ Error saving detection clip: {e}")
            import traceback
            traceback.print_exc()
    
    def _send_alert_thread(self, alert_notification: Dict, websocket_manager):
        """Send alert notification via WebSocket from thread"""
        try:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(websocket_manager.broadcast(alert_notification))
                print(f"✅ Alert sent successfully: {alert_notification.get('message', 'Unknown')}")
            finally:
                new_loop.close()
        except Exception as e:
            print(f"⚠️ Error sending alert in thread: {e}")
            import traceback
            traceback.print_exc()
    
    async def _save_violation_to_db_thread(self, violation_data: Dict, websocket_manager):
        """Save violation to database from thread - uses new connection"""
        import asyncpg
        try:
            # Create a new connection for this thread (don't use pool from main loop)
            conn = await asyncpg.connect(
                host=self.database.db_host,
                port=self.database.db_port,
                database=self.database.db_name,
                user=self.database.db_user,
                password=self.database.db_password
            )
            
            try:
                # Get location for this camera
                location = await self._get_location_for_camera_db(conn, violation_data["camera_id"])
                violation_data["location"] = location
                
                # Insert violation
                violation_id = await conn.fetchval("""
                    INSERT INTO violations 
                    (camera_id, detection_class, confidence, video_path, timestamp, frame_count, bbox_data, location)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id
                """,
                    violation_data["camera_id"],
                    violation_data["detection_class"],
                    violation_data["confidence"],
                    violation_data["video_path"],
                    violation_data["timestamp"],
                    violation_data.get("frame_count", 0),
                    json.dumps(violation_data.get("bbox_data", {})),
                    violation_data.get("location", "Unknown")
                )
                
                print(f"✅ Violation saved to DB: ID {violation_id}")
                
                # Notify WebSocket clients (this will be handled by main loop)
                # We can't use websocket_manager from here, so we'll skip it
                # The frontend can poll or we can use a queue
                
            finally:
                await conn.close()
        
        except Exception as e:
            print(f"❌ Error saving violation to DB: {e}")
            import traceback
            traceback.print_exc()
    
    async def _get_location_for_camera_db(self, conn, camera_id: int) -> str:
        """Get location for a camera using existing connection"""
        try:
            # Priority: Use Google Maps API for detailed area-level location
            # This gives locations like "Tipu Sultan, Karachi" or "Korangi, Karachi"
            location = await self._get_location_from_ip()
            if location and location != "Unknown":
                return location
            
            # Fallback: If Google Maps fails, check cameras table
            db_location = await conn.fetchval("""
                SELECT location FROM cameras WHERE id = $1
            """, camera_id)
            
            if db_location:
                return db_location
            
            return "Unknown"
        except Exception as e:
            print(f"⚠️ Error getting location: {e}")
            return "Unknown"
    
    async def _get_location_from_ip(self) -> str:
        """Get detailed location from IP using Google Maps Geocoding API"""
        try:
            # Get public IP
            public_ip = requests.get('https://api.ipify.org', timeout=2).text
            
            # Get coordinates from IP (using ip-api.com - free, no key needed)
            ip_response = requests.get(f'http://ip-api.com/json/{public_ip}', timeout=3)
            if ip_response.status_code == 200:
                ip_data = ip_response.json()
                if ip_data.get('status') == 'success':
                    lat = ip_data.get('lat')
                    lon = ip_data.get('lon')
                    
                    # If we have coordinates, use Google Maps Geocoding API for detailed location
                    if lat and lon:
                        location = await self._get_google_location(lat, lon)
                        if location and location != "Unknown":
                            return location
                    
                    # Fallback: use city and region from IP service
                    city = ip_data.get('city', '')
                    region = ip_data.get('regionName', '')
                    if city and region:
                        return f"{city}, {region}"
                    elif city:
                        return city
                    elif region:
                        return region
                    else:
                        return ip_data.get('country', 'Unknown')
        except Exception as e:
            print(f"⚠️ IP geolocation failed: {e}")
        
        return "Unknown"
    
    async def _get_google_location(self, lat: float, lon: float) -> str:
        """Get detailed area-level location using Google Maps Geocoding API"""
        try:
            # Hardcoded Google Maps API key
            google_api_key = "AIzaSyCU3GrxAZ6SrlYidlod1P6taIQgs3rQytQ"
            
            # Use Google Maps Reverse Geocoding API
            url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={google_api_key}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'OK' and len(data.get('results', [])) > 0:
                    # Get the first result (most specific)
                    result = data['results'][0]
                    address_components = result.get('address_components', [])
                    
                    # Extract area/neighborhood, sublocality, or locality
                    area = None
                    sublocality = None
                    locality = None
                    city = None
                    
                    for component in address_components:
                        types = component.get('types', [])
                        name = component.get('long_name', '')
                        
                        if 'sublocality_level_1' in types or 'sublocality' in types:
                            sublocality = name
                        elif 'neighborhood' in types:
                            area = name
                        elif 'locality' in types:
                            locality = name
                        elif 'administrative_area_level_1' in types:
                            city = name
                    
                    # Build location string: Area, City (e.g., "Tipu Sultan, Karachi")
                    location_parts = []
                    if area:
                        location_parts.append(area)
                    elif sublocality:
                        location_parts.append(sublocality)
                    
                    if locality:
                        location_parts.append(locality)
                    elif city:
                        location_parts.append(city)
                    
                    if location_parts:
                        return ", ".join(location_parts)
                    
                    # Fallback: use formatted address
                    formatted_address = result.get('formatted_address', '')
                    if formatted_address:
                        # Extract area and city from formatted address
                        parts = formatted_address.split(',')
                        if len(parts) >= 2:
                            return f"{parts[0].strip()}, {parts[-2].strip()}"
                        return formatted_address
            
            elif data.get('status') == 'REQUEST_DENIED':
                print("⚠️ Google Maps API key invalid or quota exceeded")
            elif data.get('status') == 'OVER_QUERY_LIMIT':
                print("⚠️ Google Maps API quota exceeded")
                
        except Exception as e:
            print(f"⚠️ Google Maps geocoding failed: {e}")
        
        return "Unknown"
    
    def _apply_nms(self, detections, iou_threshold=0.3, person_iou_threshold=0.1):
        """Apply Non-Maximum Suppression to filter duplicate/overlapping boxes"""
        if not detections:
            return []
        
        # Group detections by class first
        by_class = {}
        for det in detections:
            class_name = det['class'].lower()
            if class_name not in by_class:
                by_class[class_name] = []
            by_class[class_name].append(det)
        
        filtered = []
        
        # Apply NMS per class
        for class_name, class_dets in by_class.items():
            # Use more aggressive threshold for Person class
            class_iou_threshold = person_iou_threshold if class_name == 'person' else iou_threshold
            
            # Sort by confidence (highest first)
            sorted_dets = sorted(class_dets, key=lambda x: x['conf'], reverse=True)
            
            # For Person class, use aggressive NMS but allow multiple if they're far apart
            # This allows multiple people to be detected (not just one)
            if class_name == 'person':
                # Use aggressive NMS but keep multiple persons if they're far enough apart
                while sorted_dets:
                    best = sorted_dets.pop(0)
                    filtered.append(best)
                    
                    # Remove overlapping persons (but keep if far apart - multiple people)
                    remaining = []
                    for det in sorted_dets:
                        iou = self._calculate_iou(best['bbox'], det['bbox'])
                        center1 = ((best['bbox'][0] + best['bbox'][2]) / 2, (best['bbox'][1] + best['bbox'][3]) / 2)
                        center2 = ((det['bbox'][0] + det['bbox'][2]) / 2, (det['bbox'][1] + det['bbox'][3]) / 2)
                        center_dist = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
                        avg_size = ((best['bbox'][2] - best['bbox'][0]) + (best['bbox'][3] - best['bbox'][1]) + 
                                   (det['bbox'][2] - det['bbox'][0]) + (det['bbox'][3] - det['bbox'][1])) / 4
                        center_threshold = avg_size * 0.3  # Allow multiple people if far apart
                        
                        if iou < class_iou_threshold and center_dist > center_threshold:
                            remaining.append(det)
                    
                    sorted_dets = remaining
                continue  # Skip the rest of the NMS logic for Person
            
            while sorted_dets:
                # Take the highest confidence detection
                best = sorted_dets.pop(0)
                filtered.append(best)
                
                # Remove overlapping detections of the same class
                remaining = []
                for det in sorted_dets:
                    # Calculate IoU (Intersection over Union)
                    iou = self._calculate_iou(best['bbox'], det['bbox'])
                    
                    # Also check if boxes are very close (center distance)
                    center1 = ((best['bbox'][0] + best['bbox'][2]) / 2, (best['bbox'][1] + best['bbox'][3]) / 2)
                    center2 = ((det['bbox'][0] + det['bbox'][2]) / 2, (det['bbox'][1] + det['bbox'][3]) / 2)
                    center_dist = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
                    
                    # Get average box size for distance threshold
                    avg_size = ((best['bbox'][2] - best['bbox'][0]) + (best['bbox'][3] - best['bbox'][1]) + 
                               (det['bbox'][2] - det['bbox'][0]) + (det['bbox'][3] - det['bbox'][1])) / 4
                    
                    # For Person class, use very strict center distance (0.1 instead of 0.3)
                    # For other classes, use normal threshold
                    center_threshold = (avg_size * 0.1) if class_name == 'person' else (avg_size * 0.3)
                    
                    # Keep if IoU is below threshold AND centers are far enough apart
                    # This catches both overlapping and nearby duplicate boxes
                    if iou < class_iou_threshold and center_dist > center_threshold:
                        remaining.append(det)
                
                sorted_dets = remaining
        
        return filtered
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    async def _save_violation_to_db(self, violation_data: Dict, websocket_manager):
        """Save violation to database and notify via WebSocket (from main loop)"""
        try:
            # Get location if not already set
            if "location" not in violation_data:
                # Get location from database or IP
                pool = await self.database.get_pool()
                async with pool.acquire() as conn:
                    violation_data["location"] = await self._get_location_for_camera_db(conn, violation_data["camera_id"])
            
            violation_id = await self.database.create_violation(violation_data)
            print(f"✅ Violation saved to DB: ID {violation_id}")
            
            # Notify WebSocket clients
            notification = {
                "type": "violation_detected",
                "violation": {
                    "id": violation_id,
                    "camera_id": violation_data["camera_id"],
                    "detection_class": violation_data["detection_class"],
                    "confidence": violation_data["confidence"],
                    "timestamp": violation_data["timestamp"].isoformat(),
                    "video_path": violation_data["video_path"]
                }
            }
            
            await websocket_manager.broadcast(notification)
        
        except Exception as e:
            print(f"❌ Error saving violation to DB: {e}")
    
    def get_video_stream(self, camera_id: int):
        """Generator for video streaming - reads from detection loop's latest frame"""
        
        print(f"📹 Starting video stream for camera {camera_id} (using detection loop frames)")
        
        # Create placeholder for when no frames available
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Waiting for detection to start...", (50, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', placeholder)
        placeholder_bytes = buffer.tobytes()
        
        last_frame_time = 0
        frame_timeout = 5.0  # Increased timeout (was 2s) - RTSP can have longer gaps
        last_good_frame = None  # Keep last good frame during reconnections
        
        try:
            while True:
                frame = None
                
                # Get latest frame from detection loop (thread-safe)
                if camera_id in self.latest_frames and camera_id in self.frame_locks:
                    try:
                        with self.frame_locks[camera_id]:
                            frame = self.latest_frames[camera_id]
                            if frame is not None:
                                last_frame_time = time.time()
                                last_good_frame = frame.copy()  # Keep copy of good frame
                    except:
                        # If lock fails, use last good frame
                        frame = last_good_frame
                
                # Use last good frame if current is None (during reconnections)
                if frame is None and last_good_frame is not None:
                    frame = last_good_frame
                
                # Check if frame is too old (only show placeholder if we never had a frame)
                if frame is None:
                    # Send placeholder only if we never got a frame
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + placeholder_bytes + b'\r\n')
                    time.sleep(0.1)
                    continue
                elif (time.time() - last_frame_time) > frame_timeout and last_good_frame is None:
                    # Only show placeholder if we never had a good frame
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + placeholder_bytes + b'\r\n')
                    time.sleep(0.1)
                    continue
                
                # CRITICAL: NO YOLO HERE - Use stored detections from _detection_loop (YOLO runs only once)
                display_frame = frame.copy()
                # Draw stored detections ONLY (thread-safe, no YOLO, no duplicates)
                try:
                    with self.frame_locks[camera_id]:
                        detections = self.latest_detections.get(camera_id, [])
                    
                    # Draw stored detections (no YOLO, no duplicates, matches detection loop exactly)
                    for det in detections:
                        x1, y1, x2, y2 = det["bbox"]
                        label = f"{det['class']} {det['confidence']:.2f}"
                        # Draw box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Draw label with background
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(display_frame, 
                                    (x1, y1 - label_size[1] - 5),
                                    (x1 + label_size[0], y1),
                                    (0, 255, 0), -1)
                        cv2.putText(display_frame, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                except Exception as e:
                    # If drawing fails, just show frame without boxes
                    pass
                
                # Resize for streaming
                try:
                    display_frame = cv2.resize(display_frame, (640, 360))
                except:
                    pass
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Small delay to maintain ~30 FPS
                time.sleep(1.0 / 30)
        
        except Exception as e:
            print(f"❌ Error in video stream: {e}")
            import traceback
            traceback.print_exc()

