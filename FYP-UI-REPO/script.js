'use strict';

const API = 'http://localhost:8000';

// ── Auth helpers ─────────────────────────────────────────────────────────────

const getToken    = ()        => localStorage.getItem('auth_token');
const setToken    = t         => localStorage.setItem('auth_token', t);
const clearToken  = ()        => { localStorage.removeItem('auth_token'); localStorage.removeItem('user_info'); };
const getUser     = ()        => { try { return JSON.parse(localStorage.getItem('user_info')); } catch { return null; } };
const saveUser    = u         => localStorage.setItem('user_info', JSON.stringify(u));

// ── API wrapper ───────────────────────────────────────────────────────────────

async function api(path, options = {}) {
    const token = getToken();
    const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const res = await fetch(API + path, { ...options, headers });

    if (res.status === 401) {
        clearToken();
        showLogin();
        throw new Error('Session expired. Please log in again.');
    }
    return res;
}

// ── Toast notifications ───────────────────────────────────────────────────────

function toast(message, type = 'info', duration = 4000) {
    const colors = { info: '#3b82f6', success: '#10b981', error: '#ef4444', warning: '#f59e0b', violation: '#dc2626' };
    const t = document.createElement('div');
    t.className = 'toast';
    t.style.background = colors[type] || colors.info;
    t.innerHTML = `
        <span>${message}</span>
        <button onclick="this.parentElement.remove()" style="background:none;border:none;color:#fff;cursor:pointer;font-size:16px;padding:0 0 0 12px;">&times;</button>
    `;
    document.getElementById('toastContainer').prepend(t);
    setTimeout(() => t.classList.add('toast-show'), 10);
    setTimeout(() => { t.classList.remove('toast-show'); setTimeout(() => t.remove(), 300); }, duration);
}

// ── Page routing ──────────────────────────────────────────────────────────────

const VIEWS = ['dashboard', 'camera', 'violations', 'users'];

function navigate(view) {
    VIEWS.forEach(v => {
        const el = document.getElementById(`${v}-view`);
        if (el) el.style.display = 'none';
    });
    const target = document.getElementById(`${view}-view`);
    if (target) target.style.display = 'block';

    document.querySelectorAll('.nav-link').forEach(a => {
        a.classList.toggle('active', a.dataset.view === view);
    });

    window._currentView = view;

    if (view === 'dashboard')  refreshDashboard();
    if (view === 'violations') loadViolations();
    if (view === 'users')      loadUsers();
    if (view === 'camera')     initCameraView();
}

function showLogin() {
    document.getElementById('login-section').style.display = 'flex';
    document.getElementById('app-section').style.display   = 'none';
}

function showApp(user) {
    document.getElementById('login-section').style.display = 'none';
    document.getElementById('app-section').style.display   = 'flex';
    populateUserUI(user);
    navigate('dashboard');
}

function populateUserUI(user) {
    if (!user) return;
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    set('headerUserName',  user.full_name || user.email);
    set('headerUserRole',  user.role || 'Admin');
    set('dropUserName',    user.full_name || user.email);
    set('dropUserEmail',   user.email || '');
}

// ── Invite flow ───────────────────────────────────────────────────────────────

async function handleInvitePage(token) {
    document.getElementById('login-section').style.display  = 'none';
    document.getElementById('app-section').style.display    = 'none';
    document.getElementById('invite-section').style.display = 'flex';

    try {
        const res  = await fetch(`${API}/api/auth/invite/${token}`);
        const data = await res.json();
        if (!res.ok) {
            document.getElementById('inviteError').textContent   = data.detail || 'Invalid or expired invite link.';
            document.getElementById('inviteError').style.display = 'block';
            document.getElementById('inviteForm').style.display  = 'none';
            return;
        }
        document.getElementById('inviteWelcome').textContent = `Hi ${data.full_name}! Set a password for ${data.email}`;
    } catch {
        document.getElementById('inviteError').textContent   = 'Cannot reach server.';
        document.getElementById('inviteError').style.display = 'block';
        document.getElementById('inviteForm').style.display  = 'none';
        return;
    }

    document.getElementById('inviteForm').addEventListener('submit', async e => {
        e.preventDefault();
        const pw  = document.getElementById('invitePassword').value;
        const pw2 = document.getElementById('inviteConfirm').value;
        const errEl = document.getElementById('inviteError');
        errEl.style.display = 'none';

        if (pw !== pw2) {
            errEl.textContent = 'Passwords do not match.'; errEl.style.display = 'block'; return;
        }
        if (pw.length < 6) {
            errEl.textContent = 'Password must be at least 6 characters.'; errEl.style.display = 'block'; return;
        }

        const btn = document.getElementById('inviteBtn');
        document.getElementById('inviteBtnText').style.display = 'none';
        document.getElementById('inviteSpinner').style.display = 'inline-block';
        btn.disabled = true;

        try {
            const res  = await fetch(`${API}/api/auth/invite`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ token, password: pw }),
            });
            const data = await res.json();
            if (res.ok) {
                setToken(data.token); saveUser(data.user);
                document.getElementById('inviteForm').style.display   = 'none';
                document.getElementById('inviteSuccess').textContent  = 'Account activated! Redirecting…';
                document.getElementById('inviteSuccess').style.display = 'block';
                setTimeout(() => {
                    window.history.replaceState({}, '', '/');
                    showApp(data.user);
                    document.getElementById('invite-section').style.display = 'none';
                }, 1500);
            } else {
                errEl.textContent = data.detail || 'Failed to activate.'; errEl.style.display = 'block';
            }
        } catch {
            errEl.textContent = 'Cannot reach server.'; errEl.style.display = 'block';
        } finally {
            document.getElementById('inviteBtnText').style.display = 'inline';
            document.getElementById('inviteSpinner').style.display = 'none';
            btn.disabled = false;
        }
    });
}

// ── Google OAuth ──────────────────────────────────────────────────────────────

async function handleGoogleCredential(response) {
    const errEl = document.getElementById('loginError');
    errEl.style.display = 'none';
    try {
        const res  = await fetch(`${API}/api/auth/google`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ credential: response.credential }),
        });
        const data = await res.json();
        if (res.ok) {
            setToken(data.token);
            saveUser(data.user);
            showApp(data.user);
        } else {
            errEl.textContent   = data.detail || 'Google sign-in failed.';
            errEl.style.display = 'block';
        }
    } catch {
        errEl.textContent   = 'Cannot reach server. Is the backend running?';
        errEl.style.display = 'block';
    }
}

async function initGoogleAuth() {
    try {
        const res = await fetch(`${API}/api/config`);
        const { google_client_id } = await res.json();
        if (!google_client_id) return;

        const waitForGoogle = (resolve) => {
            if (window.google && window.google.accounts) return resolve();
            setTimeout(() => waitForGoogle(resolve), 100);
        };
        await new Promise(waitForGoogle);

        google.accounts.id.initialize({
            client_id: google_client_id,
            callback: handleGoogleCredential,
            auto_select: false,
        });
        google.accounts.id.renderButton(
            document.getElementById('googleSignInDiv'),
            { theme: 'outline', size: 'large', width: 320, text: 'signin_with', shape: 'rectangular' }
        );
    } catch (e) {
        console.warn('Google auth unavailable:', e);
    }
}

// ── App init ──────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    // Handle invite link (?invite=TOKEN in URL)
    const urlParams   = new URLSearchParams(window.location.search);
    const inviteToken = urlParams.get('invite');
    if (inviteToken) {
        handleInvitePage(inviteToken);
        return;
    }

    initGoogleAuth();

    const token = getToken();
    if (token) {
        api('/api/auth/me')
            .then(r => r.json())
            .then(user => { saveUser(user); showApp(user); })
            .catch(() => { clearToken(); showLogin(); });
    } else {
        showLogin();
    }

    // Login form
    document.getElementById('loginForm').addEventListener('submit', async e => {
        e.preventDefault();
        const email    = document.getElementById('loginEmail').value.trim();
        const password = document.getElementById('loginPassword').value;
        const remember = document.getElementById('rememberMe').checked;

        const errEl  = document.getElementById('loginError');
        const btn    = document.getElementById('loginBtn');
        const btnTxt = document.getElementById('loginBtnText');
        const spin   = document.getElementById('loginSpinner');

        errEl.style.display = 'none';
        btnTxt.style.display = 'none';
        spin.style.display   = 'inline-block';
        btn.disabled = true;

        try {
            const res  = await fetch(`${API}/api/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password, remember_me: remember }),
            });
            const data = await res.json();
            if (res.ok) {
                setToken(data.token);
                saveUser(data.user);
                showApp(data.user);
            } else {
                errEl.textContent    = data.detail || 'Incorrect email or password.';
                errEl.style.display  = 'block';
            }
        } catch {
            errEl.textContent   = 'Cannot reach server. Is the backend running?';
            errEl.style.display = 'block';
        } finally {
            btnTxt.style.display = 'inline';
            spin.style.display   = 'none';
            btn.disabled = false;
        }
    });

    // Forgot password link
    document.getElementById('forgotLink').addEventListener('click', e => {
        e.preventDefault();
        document.getElementById('resetEmail').value  = '';
        document.getElementById('resetMsg').style.display = 'none';
        document.getElementById('forgotModal').style.display = 'flex';
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', e => {
        const dd  = document.getElementById('userDropdown');
        const pill = document.getElementById('userPill');
        if (!dd.contains(e.target) && !pill.contains(e.target)) {
            dd.classList.remove('open');
            document.getElementById('pillChevron').classList.remove('rotated');
        }
    });

    // Auto-refresh dashboard every 30 s
    setInterval(() => { if (window._currentView === 'dashboard') refreshDashboard(); }, 30000);
});

// ── Auth actions ──────────────────────────────────────────────────────────────

function togglePassword() {
    const inp  = document.getElementById('loginPassword');
    const icon = document.getElementById('pwEyeIcon');
    if (inp.type === 'password') {
        inp.type = 'text';
        icon.className = 'far fa-eye-slash';
    } else {
        inp.type = 'password';
        icon.className = 'far fa-eye';
    }
}

async function logout() {
    try { await api('/api/auth/logout', { method: 'POST' }); } catch {}
    clearToken();
    stopWebSocket();
    showLogin();
    document.getElementById('loginForm').reset();
}

async function requestReset() {
    const email = document.getElementById('resetEmail').value.trim();
    const msgEl = document.getElementById('resetMsg');
    const btn   = document.getElementById('resetBtn');
    if (!email) { msgEl.className = 'alert alert-danger'; msgEl.textContent = 'Please enter your email.'; msgEl.style.display = 'block'; return; }

    btn.disabled = true;
    btn.textContent = 'Sending…';
    try {
        const res  = await fetch(`${API}/api/auth/forgot-password`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email }),
        });
        const data = await res.json();
        msgEl.className    = 'alert alert-success';
        msgEl.textContent  = `Token: ${data.reset_token || '—'} (in production this would be emailed)`;
        msgEl.style.display = 'block';
    } catch {
        msgEl.className    = 'alert alert-danger';
        msgEl.textContent  = 'Request failed. Please try again.';
        msgEl.style.display = 'block';
    } finally {
        btn.disabled    = false;
        btn.textContent = 'Send Reset Link';
    }
}

function closeForgotModal() {
    document.getElementById('forgotModal').style.display = 'none';
}

// ── Sidebar & dropdown ────────────────────────────────────────────────────────

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('collapsed');
}

function toggleDropdown() {
    const dd = document.getElementById('userDropdown');
    dd.classList.toggle('open');
    document.getElementById('pillChevron').classList.toggle('rotated');
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

let chartMonths, chartWeeks, chartSummary;

async function refreshDashboard() {
    try {
        const res   = await api('/api/dashboard/stats');
        const stats = await res.json();
        renderKPIs(stats);
        populateFilters(stats.detection_classes || []);
        await renderCharts(stats);
        renderOutlets(stats.top_outlets || []);
        renderRecentDates(stats.top_violations || []);
    } catch (err) {
        toast('Failed to load dashboard: ' + err.message, 'error');
    }
}

function renderKPIs(s) {
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v ?? '—'; };
    set('statTotal',  s.total_violations);
    set('statHigh',   s.high_priority);
    set('statMedium', s.medium_priority);
    set('statLow',    s.low_priority);
    set('statToday',  s.today_violations);
}

function populateFilters(classes) {
    ['monthsFilter', 'weeksFilter'].forEach(id => {
        const sel = document.getElementById(id);
        if (!sel) return;
        const cur = sel.value;
        sel.innerHTML = '<option value="__all__">All Types</option>';
        classes.forEach(c => { const o = document.createElement('option'); o.value = c; o.textContent = c; sel.appendChild(o); });
        if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
    });
}

async function reloadCharts() {
    const cls = (() => {
        const m = document.getElementById('monthsFilter')?.value;
        return m && m !== '__all__' ? m : null;
    })();
    const qs = cls ? `&detection_class=${encodeURIComponent(cls)}` : '';
    try {
        const [mRes, wRes] = await Promise.all([
            api(`/api/violations/stats?period=month${qs}`),
            api(`/api/violations/stats?period=week${qs}`),
        ]);
        const [mData, wData] = await Promise.all([mRes.json(), wRes.json()]);
        buildChartMonths(mData.stats || []);
        buildChartWeeks(wData.stats  || []);
    } catch {}
}

async function renderCharts(stats) {
    await reloadCharts();
    buildChartSummary(stats.hourly_summary || []);
}

function buildChartMonths(data) {
    const ctx = document.getElementById('chartMonths');
    if (!ctx) return;
    if (chartMonths) chartMonths.destroy();
    chartMonths = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.period).reverse(),
            datasets: [{ label: 'Violations', data: data.map(d => d.count).reverse(), backgroundColor: '#6d28d9', borderRadius: 4, barThickness: 40 }],
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } } },
    });
}

function buildChartWeeks(data) {
    const ctx = document.getElementById('chartWeeks');
    if (!ctx) return;
    if (chartWeeks) chartWeeks.destroy();
    chartWeeks = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.period).reverse(),
            datasets: [{ label: 'Violations', data: data.map(d => d.count).reverse(), backgroundColor: '#0ea5e9', borderRadius: 4, barThickness: 40 }],
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } } },
    });
}

function buildChartSummary(data) {
    const ctx = document.getElementById('chartSummary');
    if (!ctx) return;
    if (chartSummary) chartSummary.destroy();
    const now = new Date();
    let labels = [], values = [];
    if (data.length > 0) {
        labels = data.map(d => { try { return new Date(d.hour).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }); } catch { return d.hour; } });
        values = data.map(d => d.count || 0);
    } else {
        for (let i = 23; i >= 0; i--) {
            labels.push(new Date(now - i * 3600000).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }));
            values.push(0);
        }
    }
    chartSummary = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{ label: 'Violations', data: values, borderColor: '#6d28d9', backgroundColor: 'rgba(109,40,217,0.1)', tension: 0.4, fill: true, pointRadius: 3 }],
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } }, x: { grid: { display: false } } } },
    });
}

function renderOutlets(outlets) {
    const el = document.getElementById('outletsList');
    if (!el) return;
    el.innerHTML = outlets.length ? outlets.map(o => `
        <div class="list-row">
            <span>${o.location || 'Unknown'}</span>
            <span class="badge">${o.count}</span>
        </div>`).join('') : '<div class="empty-msg">No data yet</div>';
}

function renderRecentDates(dates) {
    const el = document.getElementById('recentDatesList');
    if (!el) return;
    el.innerHTML = dates.length ? dates.map(d => {
        const dateStr = d.date ? new Date(d.date + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : '—';
        return `<div class="list-row"><span>${dateStr}</span><span class="badge">${d.count}</span></div>`;
    }).join('') : '<div class="empty-msg">No data yet</div>';
}

// ── Camera view ───────────────────────────────────────────────────────────────

let currentCameraId = 1;
let ws = null;
let detectionRunning = false;

function initCameraView() {
    switchCamera(currentCameraId);
}

async function switchCamera(id) {
    currentCameraId = id;
    document.querySelectorAll('.cam-btn').forEach((b, i) => b.classList.toggle('active', i + 1 === id));
    document.getElementById('cameraTitle').textContent = `Live Camera — Cam ${id}`;
    document.getElementById('camName').textContent = `Camera 0${id}`;

    const feed   = document.getElementById('liveVideoFeed');
    const badge  = document.getElementById('streamBadge');
    const holder = document.getElementById('streamPlaceholder');

    feed.src = '';
    badge.className = 'stream-badge offline';
    badge.innerHTML = '<i class="fas fa-circle"></i> OFFLINE';
    holder.style.display = 'flex';

    connectWebSocket(id);

    try {
        const res  = await api(`/api/cameras/${id}/status`);
        const data = await res.json();
        detectionRunning = data.is_running;
        updateDetectionBtn();
        if (detectionRunning) startStream(id);
    } catch (err) {
        toast('Cannot reach server', 'error');
    }
}

function startStream(id) {
    const feed   = document.getElementById('liveVideoFeed');
    const badge  = document.getElementById('streamBadge');
    const holder = document.getElementById('streamPlaceholder');
    const token  = getToken();

    feed.src = `${API}/api/cameras/${id}/stream${token ? '?token=' + token : ''}`;
    feed.onload = () => {
        badge.className = 'stream-badge live';
        badge.innerHTML = '<i class="fas fa-circle"></i> LIVE';
        holder.style.display = 'none';
    };
    feed.onerror = () => {
        badge.className = 'stream-badge offline';
        badge.innerHTML = '<i class="fas fa-circle"></i> ERROR';
    };
}

function stopStream() {
    const feed  = document.getElementById('liveVideoFeed');
    const badge = document.getElementById('streamBadge');
    const holder = document.getElementById('streamPlaceholder');
    feed.src = '';
    badge.className = 'stream-badge offline';
    badge.innerHTML = '<i class="fas fa-circle"></i> OFFLINE';
    holder.style.display = 'flex';
}

function updateDetectionBtn() {
    const btn  = document.getElementById('detectionToggleBtn');
    const stat = document.getElementById('camDetectionStatus');
    if (detectionRunning) {
        btn.className  = 'btn btn-danger';
        btn.innerHTML  = '<i class="fas fa-stop"></i> Stop Detection';
        stat.textContent = 'Active';
        stat.className = 'status-dot active';
    } else {
        btn.className  = 'btn btn-success';
        btn.innerHTML  = '<i class="fas fa-play"></i> Start Detection';
        stat.textContent = 'Inactive';
        stat.className = 'status-dot';
    }
}

async function toggleDetection() {
    const btn = document.getElementById('detectionToggleBtn');
    btn.disabled = true;
    const action = detectionRunning ? 'stop' : 'start';
    try {
        const res = await api(`/api/cameras/${currentCameraId}/${action}`, { method: 'POST' });
        if (res.ok) {
            detectionRunning = !detectionRunning;
            updateDetectionBtn();
            if (detectionRunning) {
                startStream(currentCameraId);
                toast('Detection started', 'success');
            } else {
                stopStream();
                toast('Detection stopped', 'info');
            }
        } else {
            const d = await res.json();
            toast(d.detail || 'Action failed', 'error');
        }
    } catch (err) {
        toast('Request failed: ' + err.message, 'error');
    } finally {
        btn.disabled = false;
    }
}

function connectWebSocket(cameraId) {
    if (ws) ws.close();
    ws = new WebSocket(`ws://localhost:8000/ws/${cameraId}`);
    ws.onmessage = e => {
        const data = JSON.parse(e.data);
        if (data.type === 'violation_alert' || data.type === 'violation_detected') {
            const msg = `🚨 Violation on Camera ${data.camera_id} — conf: ${(data.confidence * 100).toFixed(0)}%`;
            toast(msg, 'violation', 8000);
            pushLiveAlert(data);
            if (window._currentView === 'violations') loadViolations();
            if (window._currentView === 'dashboard')  refreshDashboard();
        }
    };
    ws.onerror  = () => {};
    ws.onclose  = () => {};
}

function stopWebSocket() { if (ws) { ws.close(); ws = null; } }

let alertCount = 0;
function pushLiveAlert(data) {
    alertCount++;
    const feed = document.getElementById('liveAlertFeed');
    if (!feed) return;
    const first = feed.querySelector('.empty-msg');
    if (first) first.remove();
    const ts  = new Date().toLocaleTimeString();
    const row = document.createElement('div');
    row.className = 'alert-row';
    row.innerHTML = `
        <span class="alert-dot"></span>
        <span class="alert-text">${data.detection_class || 'Violation'} — Cam ${data.camera_id}</span>
        <span class="alert-time">${ts}</span>
    `;
    feed.prepend(row);
    // Keep last 20 alerts
    while (feed.children.length > 20) feed.removeChild(feed.lastChild);

    // Bell badge
    const bell  = document.getElementById('alertBell');
    const badge = document.getElementById('bellBadge');
    bell.style.display  = 'flex';
    badge.textContent   = alertCount;
}

// ── Violations view ───────────────────────────────────────────────────────────

async function loadViolations() {
    const container = document.getElementById('violationsList');
    if (!container) return;
    container.innerHTML = '<div class="loading-msg"><span class="spinner dark"></span> Loading…</div>';

    const period   = document.getElementById('violationPeriod')?.value || 'all';
    const cameraId = document.getElementById('violationCamera')?.value || 'all';

    let url = '/api/violations?limit=100';
    if (cameraId !== 'all') url += `&camera_id=${cameraId}`;

    const now = new Date();
    if (period === 'today') {
        url += `&start_date=${now.toISOString().split('T')[0]}`;
    } else if (period === 'week') {
        const d = new Date(now); d.setDate(d.getDate() - 7);
        url += `&start_date=${d.toISOString().split('T')[0]}`;
    } else if (period === 'month') {
        const d = new Date(now); d.setMonth(d.getMonth() - 1);
        url += `&start_date=${d.toISOString().split('T')[0]}`;
    }

    try {
        const res  = await api(url);
        const data = await res.json();
        const list = data.violations || [];

        if (!list.length) {
            container.innerHTML = '<div class="empty-msg" style="padding:40px;">No violations found</div>';
            return;
        }
        container.innerHTML = list.map(v => {
            const prio  = v.priority || 'low';
            const ts    = new Date(v.timestamp).toLocaleString();
            const conf  = (v.confidence * 100).toFixed(1);
            return `
            <div class="violation-row">
                <div class="v-left">
                    <span class="priority-badge ${prio}">${prio.toUpperCase()}</span>
                    <div class="v-info">
                        <span class="v-class">${v.detection_class}</span>
                        <span class="v-meta">Camera ${v.camera_id} &bull; ${ts} &bull; Confidence: ${conf}%</span>
                        ${v.location ? `<span class="v-meta"><i class="fas fa-map-marker-alt"></i> ${v.location}</span>` : ''}
                    </div>
                </div>
                ${v.has_video
                    ? `<button class="btn btn-ghost btn-sm" onclick="downloadVideo(${v.id})" title="Download clip">
                           <i class="fas fa-download"></i> Video
                       </button>`
                    : `<button class="btn btn-ghost btn-sm" disabled title="Clip not saved — detection may have been stopped before recording finished" style="opacity:.4;cursor:not-allowed;">
                           <i class="fas fa-video-slash"></i> No Clip
                       </button>`
                }
            </div>`;
        }).join('');
    } catch (err) {
        container.innerHTML = `<div class="empty-msg" style="padding:40px;color:#ef4444;">Error: ${err.message}</div>`;
    }
}

async function downloadVideo(id) {
    try {
        const res = await api(`/api/violations/${id}/video`);
        if (!res.ok) { toast('Video not found', 'error'); return; }
        const blob = await res.blob();
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href = url; a.download = `violation_${id}.mp4`;
        document.body.appendChild(a); a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (err) {
        toast('Download failed: ' + err.message, 'error');
    }
}

// ── Users view ────────────────────────────────────────────────────────────────

async function loadUsers() {
    const container = document.getElementById('usersList');
    if (!container) return;
    container.innerHTML = '<div class="loading-msg"><span class="spinner dark"></span> Loading…</div>';
    try {
        const res   = await api('/api/admin/users');
        const data  = await res.json();
        const users = data.users || [];
        const me    = getUser();
        if (!users.length) {
            container.innerHTML = '<div class="empty-msg" style="padding:30px;">No users found</div>';
            return;
        }
        container.innerHTML = users.map(u => {
            const isMe       = me && u.id === me.id;
            const isGoogle   = u.auth_provider === 'google';
            const isInactive = !u.is_active;
            const providerBadge = isGoogle
                ? `<span class="provider-badge google"><i class="fab fa-google"></i> Google</span>`
                : `<span class="provider-badge local"><i class="fas fa-lock"></i> Email</span>`;
            const youBadge = isMe ? `<span class="you-badge">You</span>` : '';
            const lastLogin = u.last_login
                ? 'Last login: ' + new Date(u.last_login).toLocaleString()
                : 'Never logged in';
            const toggleBtn = !isMe ? `
                <button onclick="toggleUserActive(${u.id}, ${u.is_active})"
                    class="btn btn-sm ${isInactive ? 'btn-success' : 'btn-warning'}"
                    title="${isInactive ? 'Activate' : 'Deactivate'} account">
                    <i class="fas fa-${isInactive ? 'check' : 'ban'}"></i> ${isInactive ? 'Activate' : 'Deactivate'}
                </button>` : '';
            const removeBtn = !isMe ? `
                <button onclick="removeAdmin(${u.id})" class="btn btn-danger btn-sm" title="Permanently delete">
                    <i class="fas fa-trash-alt"></i>
                </button>` : '';
            return `
            <div class="user-row ${isInactive ? 'user-inactive' : ''}">
                <div class="user-avatar-sm" style="${isInactive ? 'opacity:.4;' : ''}">
                    <i class="fas ${isGoogle ? 'fa-google' : 'fa-user'}"></i>
                </div>
                <div class="user-details">
                    <span class="user-full-name">${u.full_name} ${youBadge} ${isInactive ? '<span class="inactive-badge">Inactive</span>' : ''}</span>
                    <span class="user-email">${u.email}</span>
                    <span class="user-meta">${lastLogin}</span>
                </div>
                ${providerBadge}
                <span class="role-badge">${u.role || 'admin'}</span>
                <div class="user-actions">${toggleBtn}${removeBtn}</div>
            </div>`;
        }).join('');
    } catch (err) {
        container.innerHTML = `<div class="empty-msg" style="padding:30px;color:#ef4444;">${err.message}</div>`;
    }
}

async function toggleUserActive(userId, currentlyActive) {
    const action = currentlyActive ? 'deactivate' : 'activate';
    if (!confirm(`${action.charAt(0).toUpperCase() + action.slice(1)} this user?`)) return;
    try {
        const res  = await api(`/api/admin/users/${userId}/toggle`, { method: 'PATCH' });
        const data = await res.json();
        if (res.ok) {
            toast(`User ${data.is_active ? 'activated' : 'deactivated'}`, data.is_active ? 'success' : 'warning');
            loadUsers();
        } else {
            toast(data.detail || 'Failed', 'error');
        }
    } catch {
        toast('Request failed', 'error');
    }
}

async function addAdmin() {
    const name  = document.getElementById('newName').value.trim();
    const email = document.getElementById('newEmail').value.trim();
    const msgEl = document.getElementById('addUserMsg');
    const btn   = document.getElementById('addAdminBtn');

    msgEl.style.display = 'none';
    if (!name || !email) {
        msgEl.className = 'alert alert-danger'; msgEl.textContent = 'Name and email are required.'; msgEl.style.display = 'block'; return;
    }

    btn.disabled = true;
    try {
        const res  = await api('/api/admin/users', { method: 'POST', body: JSON.stringify({ email, full_name: name }) });
        const data = await res.json();
        if (res.ok) {
            const sentMsg = data.email_sent
                ? `Invitation sent to <strong>${email}</strong>. They must click the link in their email to activate.`
                : `Email not configured — share this invite link manually:<br><a href="${data.invite_link}" target="_blank" style="word-break:break-all;font-size:12px;">${data.invite_link}</a>`;
            msgEl.className = 'alert alert-success'; msgEl.innerHTML = sentMsg; msgEl.style.display = 'block';
            document.getElementById('newName').value  = '';
            document.getElementById('newEmail').value = '';
            await loadUsers();
        } else {
            msgEl.className = 'alert alert-danger'; msgEl.textContent = data.detail || 'Failed to invite.'; msgEl.style.display = 'block';
        }
    } catch (err) {
        msgEl.className = 'alert alert-danger'; msgEl.textContent = err.message; msgEl.style.display = 'block';
    } finally {
        btn.disabled = false;
    }
}

async function removeAdmin(userId) {
    if (!confirm('Remove this admin? They will no longer be able to log in.')) return;
    try {
        const res  = await api(`/api/admin/users/${userId}`, { method: 'DELETE' });
        const data = await res.json();
        if (res.ok) {
            toast('Admin removed', 'success');
            loadUsers();
        } else {
            toast(data.detail || 'Failed to remove admin', 'error');
        }
    } catch (err) {
        toast(err.message, 'error');
    }
}
