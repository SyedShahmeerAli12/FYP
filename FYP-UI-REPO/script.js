// API Configuration
const API_BASE_URL = 'http://localhost:8000';
let currentCameraId = null;
let websocketConnection = null;
let detectionStatus = {};

// Authentication helpers
function getAuthToken() {
    return localStorage.getItem('auth_token');
}

function setAuthToken(token) {
    localStorage.setItem('auth_token', token);
}

function removeAuthToken() {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_info');
}

function getUserInfo() {
    const userInfo = localStorage.getItem('user_info');
    return userInfo ? JSON.parse(userInfo) : null;
}

function setUserInfo(user) {
    localStorage.setItem('user_info', JSON.stringify(user));
}

// API request helper with authentication
async function apiRequest(url, options = {}) {
    const token = getAuthToken();
    const headers = {
        'Content-Type': 'application/json',
        ...options.headers
    };
    
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}${url}`, {
            ...options,
            headers
        });
        
        // If unauthorized, redirect to login
        if (response.status === 401) {
            removeAuthToken();
            showLoginPage();
            throw new Error('Unauthorized - Please login again');
        }
        
        return response;
    } catch (error) {
        if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
            console.error('Network error: Backend may not be running. Please start the backend server.');
            throw new Error('Cannot connect to backend server. Please ensure the backend is running on http://localhost:8000');
        }
        throw error;
    }
}

function showLoginPage() {
    document.getElementById('login-section').style.display = 'flex';
    document.getElementById('dashboard-section').style.display = 'none';
}

function showDashboardPage() {
    document.getElementById('login-section').style.display = 'none';
    document.getElementById('dashboard-section').style.display = 'flex';
}

function showErrorMessage(message) {
    // Remove existing error messages
    const existingError = document.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
    
    // Create error message element
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.style.cssText = 'color: #dc2626; font-size: 14px; margin-top: 10px; padding: 10px; background: #fee2e2; border-radius: 6px;';
    errorDiv.textContent = message;
    
    // Insert after login form
    const loginForm = document.getElementById('loginForm');
    loginForm.parentNode.insertBefore(errorDiv, loginForm.nextSibling);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentNode) {
            errorDiv.remove();
        }
    }, 5000);
}

// Check token immediately (before DOM loads) to prevent flash
(function() {
    const token = localStorage.getItem('auth_token');
    if (token) {
        // Hide login, show dashboard immediately
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('login-section').style.display = 'none';
            document.getElementById('dashboard-section').style.display = 'flex';
        });
    }
})();

document.addEventListener("DOMContentLoaded", function() {
    
    // Check if user is already logged in
    const token = getAuthToken();
    if (token) {
        // Validate token by calling /api/auth/me
        apiRequest('/api/auth/me')
            .then(response => {
                if (response.ok) {
                    return response.json();
                } else {
                    throw new Error('Invalid token');
                }
            })
            .then(user => {
                setUserInfo(user);
                showDashboardPage();
                loadDashboardData();
                renderCharts([], [], null);
                updateUserDisplay(user);
            })
            .catch(() => {
                removeAuthToken();
                showLoginPage();
            });
    } else {
        // No token, show login page
        showLoginPage();
    }
    
    // Forgot Password Link
    const forgotPasswordLink = document.getElementById('forgotPasswordLink');
    if (forgotPasswordLink) {
        forgotPasswordLink.addEventListener('click', function(e) {
            e.preventDefault();
            showForgotPasswordModal();
        });
    }
    
    // 1. LOGIN LOGIC
    const loginForm = document.getElementById('loginForm');
    if(loginForm) {
        loginForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const emailInput = loginForm.querySelector('input[name="username"]') || loginForm.querySelector('input[type="text"]');
            const passwordInput = loginForm.querySelector('input[name="password"]') || loginForm.querySelector('input[type="password"]');
            
            if (!emailInput || !passwordInput) {
                showErrorMessage('Login form fields not found. Please refresh the page.');
                return;
            }
            
            const email = emailInput.value.trim();
            const password = passwordInput.value.trim(); // Trim password too to remove any accidental spaces
            const rememberMe = document.getElementById('rememberMe').checked;
            
            // Remove existing error messages
            const existingError = document.querySelector('.error-message');
            if (existingError) {
                existingError.remove();
            }
            
            // Show loading state
            const submitBtn = loginForm.querySelector('.btn-submit');
            const originalText = submitBtn.textContent;
            submitBtn.textContent = 'Logging in...';
            submitBtn.disabled = true;
            
            try {
                console.log('Attempting login:', { email, password_length: password.length, remember_me: rememberMe });
                
                const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        email: email,
                        password: password,
                        remember_me: rememberMe
                    })
                });
                
                const data = await response.json();
                console.log('Login response:', { status: response.status, data });
                
                if (response.ok) {
                    // Store token and user info
                    setAuthToken(data.token);
                    setUserInfo(data.user);
                    
                    // Show dashboard
                    showDashboardPage();
                    loadDashboardData();
                    renderCharts([], [], null);
                    updateUserDisplay(data.user);
                } else {
                    // Show error message
                    console.error('Login failed:', data);
                    showErrorMessage(data.detail || 'Login failed. Please check your credentials.');
                }
            } catch (error) {
                showErrorMessage('Network error. Please try again.');
                console.error('Login error:', error);
            } finally {
                submitBtn.textContent = originalText;
                submitBtn.disabled = false;
            }
        });
    }

    // 2. SIDEBAR TOGGLE
    const toggleBtn = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('sidebar');
    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', function() {
            sidebar.classList.toggle('collapsed');
            setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 300);
        });
    }

    // 3. LIVE CAMERA SUBMENU DROPDOWN
    // Live Cameras link - now directly loads camera view (no dropdown)
    const cameraLink = document.getElementById('cameraLink');
    if(cameraLink) {
        cameraLink.addEventListener('click', function(e) {
            e.preventDefault();
            loadCamera(1); // Default to camera 1
        });
    }

    // 4. PROFILE DROPDOWN
    const userAvatar = document.getElementById('userAvatar');
    const userDropdown = document.getElementById('userDropdown');
    if (userAvatar && userDropdown) {
        userAvatar.addEventListener('click', function(e) {
            e.stopPropagation();
            userDropdown.classList.toggle('show');
        });
        document.addEventListener('click', function(e) {
            if (!userDropdown.contains(e.target) && !userAvatar.contains(e.target)) {
                userDropdown.classList.remove('show');
            }
        });
    }

    // 5. LOGOUT LOGIC
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', async function(e) {
            e.preventDefault();
            try {
                await apiRequest('/api/auth/logout', { method: 'POST' });
            } catch (error) {
                console.error('Logout error:', error);
            } finally {
                removeAuthToken();
            userDropdown.classList.remove('show');
                showLoginPage();
            sidebar.classList.remove('collapsed');
                // Clear form
                document.getElementById('loginForm').reset();
            }
        });
    }
    
    // Update user display in header
    function updateUserDisplay(user) {
        const userDetails = document.querySelector('.user-details');
        if (userDetails && user) {
            const h4 = userDetails.querySelector('h4');
            const span = userDetails.querySelector('span');
            if (h4) h4.textContent = user.email || 'Admin';
            if (span) span.textContent = user.full_name || 'Admin';
        }
    }

    // 6. LOAD DASHBOARD DATA
    async function loadDashboardData() {
        try {
            // Load dashboard stats
            const statsResponse = await apiRequest('/api/dashboard/stats');
            const stats = await statsResponse.json();
            
            // Update KPI cards
            updateKPICards(stats);
            
            // Load violation charts (pass stats for summary chart)
            await loadViolationCharts(stats);
        } catch (error) {
            console.error('Error loading dashboard data:', error);
        }
    }
    
    // Auto-refresh dashboard every 30 seconds
    setInterval(() => {
        if (document.getElementById('dashboard-view') && 
            document.getElementById('dashboard-view').style.display !== 'none') {
            loadDashboardData();
        }
    }, 30000); // 30 seconds
    
    function updateKPICards(stats) {
        // Update violations count
        const violationsCountCard = document.querySelector('.stat-card.bg-blue h3');
        if (violationsCountCard) {
            violationsCountCard.textContent = stats.total_violations || 0;
        }
        
        // Update high priority violations
        const highPriorityCard = document.querySelector('.stat-card.bg-bright-red h3');
        if (highPriorityCard) {
            highPriorityCard.textContent = stats.high_priority || 0;
        }
        
        // Update medium priority violations
        const mediumPriorityCard = document.querySelector('.stat-card.bg-yellow h3');
        if (mediumPriorityCard) {
            mediumPriorityCard.textContent = stats.medium_priority || 0;
        }
        
        // Update low priority violations
        const lowPriorityCard = document.querySelector('.stat-card.bg-orange h3');
        if (lowPriorityCard) {
            lowPriorityCard.textContent = stats.low_priority || 0;
        }
        
        // Update Top 7 Outlets list
        updateTopOutletsList(stats.top_outlets || []);
    }
    
    function updateTopViolationsList(violations) {
        // Find the card with "Top 7 Violations" title
        const cards = document.querySelectorAll('.card');
        let topViolationsCard = null;
        for (let card of cards) {
            const title = card.querySelector('.card-title');
            if (title && (title.textContent.includes('Top 7 Violations') || title.textContent.includes('Top 7 Violations by Date'))) {
                topViolationsCard = card;
                break;
            }
        }
        
        if (topViolationsCard) {
            const cardBody = topViolationsCard.querySelector('.card-body');
            if (cardBody) {
                if (violations.length === 0) {
                    cardBody.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">No violations yet</div>';
                } else {
                    cardBody.innerHTML = violations.map((v) => {
                        // Format date: "2025-12-21" -> "December 21, 2025"
                        let formattedDate = 'Unknown Date';
                        const dateValue = v.date;
                        
                        if (dateValue) {
                            try {
                                const dateStr = String(dateValue).trim();
                                const dateMatch = dateStr.match(/(\d{4}-\d{2}-\d{2})/);
                                
                                if (dateMatch) {
                                    const dateParts = dateMatch[1].split('-');
                                    const year = parseInt(dateParts[0], 10);
                                    const month = parseInt(dateParts[1], 10) - 1;
                                    const day = parseInt(dateParts[2], 10);
                                    
                                    if (!isNaN(year) && !isNaN(month) && !isNaN(day)) {
                                        const monthNames = ['January', 'February', 'March', 'April', 'May', 'June',
                                                           'July', 'August', 'September', 'October', 'November', 'December'];
                                        formattedDate = `${monthNames[month]} ${day}, ${year}`;
                                    }
                                }
                            } catch (e) {
                                console.error('Date formatting error:', e);
                            }
                        }
                        
                        // EXACT COPY of outlets format - character by character identical structure
                        return `<div class="list-item" style="display: flex; justify-content: space-between; align-items: center; padding: 12px; border-bottom: 1px solid #eee;">
                            <span style="font-weight: 600; color: #333;">${formattedDate}</span>
                            <span style="color: #666; font-weight: 600; background: #f3f4f6; padding: 4px 10px; border-radius: 4px;">${v.count || 0}</span>
                        </div>`;
                    }).join('');
                }
            }
        }
    }
    
    function updateTopOutletsList(outlets) {
        // Find the card with "Top 7 Outlets" title
        const cards = document.querySelectorAll('.card');
        let outletsCard = null;
        for (let card of cards) {
            const title = card.querySelector('.card-title');
            if (title && title.textContent.includes('Top 7 Outlets')) {
                outletsCard = card;
                break;
            }
        }
        
        if (outletsCard) {
            const cardBody = outletsCard.querySelector('.card-body');
            if (cardBody) {
                if (outlets.length === 0) {
                    cardBody.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">No violations yet</div>';
                } else {
                    cardBody.innerHTML = outlets.map((o, idx) => 
                        `<div class="list-item" style="display: flex; justify-content: space-between; align-items: center; padding: 12px; border-bottom: 1px solid #eee;">
                            <span style="font-weight: 600; color: #333;">${o.location || 'Unknown Location'}</span>
                            <span style="color: #666; font-weight: 600; background: #f3f4f6; padding: 4px 10px; border-radius: 4px;">${o.count || 0}</span>
                        </div>`
                    ).join('');
                }
            }
        }
    }
    
    function _populateChartFiltersFromDetectionClasses(detectionClasses) {
        const monthsSel = document.getElementById('monthsFilter');
        const weeksSel = document.getElementById('weeksFilter');
        if (!monthsSel || !weeksSel) return;

        const currentMonths = monthsSel.value || "__all__";
        const currentWeeks = weeksSel.value || "__all__";

        const uniq = Array.from(new Set(detectionClasses || []));

        const fill = (sel, current) => {
            sel.innerHTML = '';
            const optAll = document.createElement('option');
            optAll.value = '__all__';
            optAll.textContent = 'All Violations';
            sel.appendChild(optAll);

            for (const c of uniq) {
                const opt = document.createElement('option');
                opt.value = c;
                opt.textContent = c;
                sel.appendChild(opt);
            }
            sel.value = current;
            if (!sel.value) sel.value = '__all__';
        };

        fill(monthsSel, currentMonths);
        fill(weeksSel, currentWeeks);
    }

    async function loadViolationCharts(dashboardStats = null) {
        try {
            // Get dashboard stats if not provided
            if (!dashboardStats) {
                try {
                    const statsResponse = await apiRequest('/api/dashboard/stats');
                    dashboardStats = await statsResponse.json();
                    _populateChartFiltersFromDetectionClasses(dashboardStats.detection_classes || []);
                } catch (e) {
                    // Non-fatal: charts can still load without filters
                }
            } else {
                _populateChartFiltersFromDetectionClasses(dashboardStats.detection_classes || []);
            }

            const monthsSel = document.getElementById('monthsFilter');
            const weeksSel = document.getElementById('weeksFilter');
            const selectedClass = (monthsSel && monthsSel.value && monthsSel.value !== '__all__')
                ? monthsSel.value
                : ((weeksSel && weeksSel.value && weeksSel.value !== '__all__') ? weeksSel.value : null);

            const filterQs = selectedClass ? `&detection_class=${encodeURIComponent(selectedClass)}` : '';

            // Load monthly stats
            const monthlyResponse = await apiRequest(`/api/violations/stats?period=month${filterQs}`);
            const monthlyStats = await monthlyResponse.json();
            
            // Load weekly stats
            const weeklyResponse = await apiRequest(`/api/violations/stats?period=week${filterQs}`);
            const weeklyStats = await weeklyResponse.json();
            
            renderCharts(monthlyStats.stats, weeklyStats.stats, dashboardStats);
        } catch (error) {
            console.error('Error loading violation charts:', error);
            renderCharts([], [], null);
        }
    }

    // Re-render charts on filter change
    const monthsSel = document.getElementById('monthsFilter');
    const weeksSel = document.getElementById('weeksFilter');
    if (monthsSel) {
        monthsSel.addEventListener('change', () => loadViolationCharts());
    }
    if (weeksSel) {
        weeksSel.addEventListener('change', () => loadViolationCharts());
    }
    
    // 7. CHARTS CONFIGURATION
    function renderCharts(monthlyData = [], weeklyData = [], dashboardStats = null) {
        const ctxMonths = document.getElementById('chartMonths');
        if (ctxMonths) {
            if (window.chartMonthsInstance) window.chartMonthsInstance.destroy();
            
            const labels = monthlyData.map(s => s.period).reverse() || ['', '', '', 'October 2025', '', '', ''];
            const data = monthlyData.map(s => s.count).reverse() || [0, 0, 0, 0, 0, 0, 0];
            
            window.chartMonthsInstance = new Chart(ctxMonths.getContext('2d'), {
                type: 'bar',
                data: { 
                    labels: labels, 
                    datasets: [{ 
                        label: 'Violations', 
                        data: data, 
                        backgroundColor: '#7b1fa2', 
                        barThickness: 80, 
                        borderRadius: 4 
                    }] 
                },
                options: { 
                    responsive: true, 
                    maintainAspectRatio: false, 
                    plugins: { legend: { display: false } }, 
                    scales: { y: { beginAtZero: true } } 
                }
            });
        }
        
        const ctxWeeks = document.getElementById('chartWeeks');
        if (ctxWeeks) {
            if (window.chartWeeksInstance) window.chartWeeksInstance.destroy();
            
            const labels = weeklyData.map(s => s.period).reverse() || ['Week: 41', '', '', '', '', '', 'Week: 43'];
            const data = weeklyData.map(s => s.count).reverse() || [0, 0, 0, 0, 0, 0, 0];
            
            window.chartWeeksInstance = new Chart(ctxWeeks.getContext('2d'), {
                data: { 
                    labels: labels, 
                    datasets: [{ 
                        type: 'bar', 
                        label: 'Violations', 
                        data: data, 
                        backgroundColor: '#7b1fa2', 
                        barThickness: 60, 
                        order: 2 
                    }] 
                },
                options: { 
                    responsive: true, 
                    maintainAspectRatio: false, 
                    plugins: { legend: { position: 'bottom', labels: { boxWidth: 10 } } }, 
                    scales: { y: { beginAtZero: true } } 
                }
            });
        }
        
        const ctxSummary = document.getElementById('chartSummary');
        if (ctxSummary) {
            if (window.chartSummaryInstance) window.chartSummaryInstance.destroy();
            
            // Get hourly summary data from dashboard stats
            const hourlyData = (dashboardStats && dashboardStats.hourly_summary) ? dashboardStats.hourly_summary : [];
            
            // Format labels (show hour in HH:MM format)
            let labels = [];
            let data = [];
            
            if (hourlyData.length > 0) {
                labels = hourlyData.map(item => {
                    try {
                        const date = new Date(item.hour);
                        return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
                    } catch (e) {
                        return String(item.hour).substring(11, 16); // Extract HH:MM from ISO string
                    }
                });
                data = hourlyData.map(item => item.count || 0);
            } else {
                // If no hourly data, show last 24 hours with zeros
                const now = new Date();
                for (let i = 23; i >= 0; i--) {
                    const hour = new Date(now.getTime() - i * 60 * 60 * 1000);
                    labels.push(hour.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }));
                    data.push(0);
                }
            }
            
            window.chartSummaryInstance = new Chart(ctxSummary.getContext('2d'), {
                type: 'line', 
                data: { 
                    labels: labels,
                    datasets: [{
                        label: 'Violations',
                        data: data,
                        borderColor: '#7b1fa2',
                        backgroundColor: 'rgba(123, 31, 162, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }]
                },
                options: { 
                    responsive: true, 
                    maintainAspectRatio: false, 
                    plugins: { 
                        legend: { display: false },
                        tooltip: {
                            mode: 'index',
                            intersect: false
                        }
                    }, 
                    scales: { 
                        y: { 
                            beginAtZero: true,
                            ticks: {
                                stepSize: 1
                            }
                        },
                        x: { 
                            grid: { display: false },
                            ticks: {
                                maxRotation: 45,
                                minRotation: 0
                            }
                        } 
                    },
                    interaction: {
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    }
                }
            });
        }
    }
});

// 7. GLOBAL FUNCTIONS FOR HTML ONCLICK

function showDashboard() {
    document.getElementById('dashboard-view').style.display = 'block';
    document.getElementById('camera-view').style.display = 'none';
    // Reset active class
    document.querySelectorAll('.nav-item a').forEach(el => el.classList.remove('active'));
    // Set dashboard active
    document.querySelector('.nav-item a[onclick="showDashboard()"]').classList.add('active');
    // Refresh dashboard data when switching to dashboard
    loadDashboardData();
}

async function loadCamera(cameraId) {
    document.getElementById('dashboard-view').style.display = 'none';
    document.getElementById('camera-view').style.display = 'block';
    
    // Reset active class on all nav items
    document.querySelectorAll('.nav-item a').forEach(el => el.classList.remove('active'));
    // Set Live Cameras as active
    const cameraLink = document.getElementById('cameraLink');
    if (cameraLink) {
        cameraLink.classList.add('active');
    }

    document.getElementById('cameraTitle').innerText = `Live Stream: Camera 0${cameraId}`;
    document.getElementById('camIp').innerText = `192.168.1.10${cameraId}`;
    
    currentCameraId = cameraId;
    
    // Connect WebSocket for real-time alerts
    connectWebSocket(cameraId);
    
    // Load video stream
    const videoFeed = document.getElementById('liveVideoFeed');
    const streamStatus = document.getElementById('streamStatus');
    
    // Set up MJPEG stream with token in query parameter (img tags can't send headers)
    const token = getAuthToken();
    if (token) {
        videoFeed.src = `${API_BASE_URL}/api/cameras/${cameraId}/stream?token=${token}`;
    } else {
        videoFeed.src = `${API_BASE_URL}/api/cameras/${cameraId}/stream`;
    }
    videoFeed.onload = () => {
        if (streamStatus) {
            streamStatus.innerHTML = '<i class="fas fa-circle" style="font-size: 8px; margin-right: 4px;"></i> LIVE';
            streamStatus.style.background = 'rgba(220, 38, 38, 0.8)';
        }
        if (placeholder) placeholder.style.display = 'none';
    };
    videoFeed.onerror = () => {
        if (streamStatus) {
            streamStatus.innerHTML = '<i class="fas fa-circle" style="font-size: 8px; margin-right: 4px;"></i> ERROR';
            streamStatus.style.background = 'rgba(239, 68, 68, 0.8)';
        }
    };
    
    // Check detection status
    await checkDetectionStatus(cameraId);
    
    console.log(`Switching to Camera ${cameraId}`);
}

async function checkDetectionStatus(cameraId) {
    try {
        const response = await apiRequest(`/api/cameras/${cameraId}/status`);
        const data = await response.json();
        detectionStatus[cameraId] = data.is_running;
        updateDetectionButton(cameraId, data.is_running);
    } catch (error) {
        console.error('Error checking detection status:', error);
    }
}

function updateDetectionButton(cameraId, isRunning) {
    let btnContainer = document.getElementById('detectionButtonContainer');
    if (!btnContainer) {
        // Create button container if it doesn't exist
        const cameraCard = document.querySelector('#camera-view .card');
        const cardHeader = cameraCard.querySelector('.card-header');
        btnContainer = document.createElement('div');
        btnContainer.id = 'detectionButtonContainer';
        btnContainer.style.marginTop = '10px';
        cardHeader.appendChild(btnContainer);
    }
    
    btnContainer.innerHTML = `
        <button id="detectionToggleBtn" class="detection-btn ${isRunning ? 'stop' : 'start'}" 
                onclick="toggleDetection(${cameraId})">
            <i class="fas ${isRunning ? 'fa-stop' : 'fa-play'}"></i>
            ${isRunning ? 'Stop Detection' : 'Start Detection'}
        </button>
        <span id="detectionStatus" style="margin-left: 10px; color: ${isRunning ? 'green' : 'gray'};">
            ${isRunning ? '● Detection Active' : '○ Detection Inactive'}
        </span>
    `;
}

async function toggleDetection(cameraId) {
    const isRunning = detectionStatus[cameraId] || false;
    
    try {
        const endpoint = isRunning ? 'stop' : 'start';
        const response = await apiRequest(`/api/cameras/${cameraId}/${endpoint}`, {
            method: 'POST'
        });
        
        if (response.ok) {
            detectionStatus[cameraId] = !isRunning;
            updateDetectionButton(cameraId, !isRunning);
            
            if (!isRunning) {
                // Show notification
                showNotification('Detection started successfully!', 'success');
            } else {
                showNotification('Detection stopped.', 'info');
            }
        } else {
            throw new Error('Failed to toggle detection');
        }
    } catch (error) {
        console.error('Error toggling detection:', error);
        showNotification('Error toggling detection. Please try again.', 'error');
    }
}

function connectWebSocket(cameraId) {
    // Close existing connection
    if (websocketConnection) {
        websocketConnection.close();
    }
    
    // Connect to WebSocket
    const ws = new WebSocket(`ws://localhost:8000/ws/${cameraId}`);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'violation_alert') {
            // Show immediate prominent alert
            showNotification(
                data.message || 'Violation detected',
                'violation',
                true  // Make it prominent
            );
            
            // Refresh violations list if on violations page
            if (window.currentView === 'violations') {
                loadViolations();
            }
            
            // Update dashboard stats
            loadDashboardData();
        } else if (data.type === 'violation_detected') {
            // Show notification
            showNotification(
                'Violation detected',
                'violation'
            );
            
            // Refresh violations list if on violations page
            if (window.currentView === 'violations') {
                loadViolations();
            }
            
            // Update dashboard stats
            loadDashboardData();
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
    };
    
    websocketConnection = ws;
}

function showNotification(message, type = 'info', prominent = false) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Make violation alerts more prominent
    const isProminent = prominent || type === 'violation';
    const padding = isProminent ? '20px 25px' : '15px 20px';
    const fontSize = isProminent ? '18px' : '14px';
    const fontWeight = isProminent ? 'bold' : 'normal';
    const duration = isProminent ? 8000 : 5000; // Show longer for violations
    
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: ${padding};
        background: ${type === 'violation' ? '#dc2626' : type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
        border-radius: 6px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
        font-size: ${fontSize};
        font-weight: ${fontWeight};
        ${isProminent ? 'border: 3px solid #ff0000;' : ''}
    `;
    
    document.body.appendChild(notification);
    
    // Remove after duration
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, duration);
}

// Violations viewing functions
async function showViolationsView() {
    document.getElementById('dashboard-view').style.display = 'none';
    document.getElementById('camera-view').style.display = 'none';
    
    let violationsView = document.getElementById('violations-view');
    if (!violationsView) {
        violationsView = createViolationsView();
        document.querySelector('.content-area').appendChild(violationsView);
    }
    
    violationsView.style.display = 'block';
    window.currentView = 'violations';
    
    await loadViolations();
}

function createViolationsView() {
    const view = document.createElement('div');
    view.id = 'violations-view';
    view.innerHTML = `
        <div class="filter-container">
            <div class="filter-box">
                <span class="filter-label">Period</span>
                <select id="violationPeriod" class="filter-display" onchange="loadViolations()">
                    <option value="all">All Time</option>
                    <option value="month">This Month</option>
                    <option value="week">This Week</option>
                    <option value="today">Today</option>
                </select>
            </div>
            <div class="filter-box">
                <span class="filter-label">Camera</span>
                <select id="violationCamera" class="filter-display" onchange="loadViolations()">
                    <option value="all">All Cameras</option>
                    <option value="1">Camera 01</option>
                </select>
            </div>
        </div>
        <div class="card">
            <div class="card-header">
                <div class="card-title">Violations Log</div>
            </div>
            <div class="card-body" id="violationsList">
                <div style="text-align: center; padding: 40px;">Loading violations...</div>
            </div>
        </div>
    `;
    return view;
}

async function loadViolations() {
    try {
        const period = document.getElementById('violationPeriod')?.value || 'all';
        const cameraId = document.getElementById('violationCamera')?.value || '';
        
        let url = '/api/violations?limit=100';
        if (cameraId && cameraId !== 'all') {
            url += `&camera_id=${cameraId}`;
        }
        
        const response = await apiRequest(url);
        const data = await response.json();
        
        const violationsList = document.getElementById('violationsList');
        if (!violationsList) return;
        
        if (data.violations && data.violations.length > 0) {
            violationsList.innerHTML = data.violations.map(v => `
                <div class="violation-item">
                    <div class="violation-info">
                        <h4>${v.detection_class} - ${(v.confidence * 100).toFixed(1)}%</h4>
                        <p>Camera ${v.camera_id} • ${new Date(v.timestamp).toLocaleString()}</p>
                        ${v.location ? `<p>Location: ${v.location}</p>` : ''}
                    </div>
                    <button onclick="downloadViolationVideo(${v.id})" class="btn-download">Download Video</button>
                </div>
            `).join('');
        } else {
            violationsList.innerHTML = '<p style="text-align: center; padding: 20px; color: #6b7280;">No violations found</p>';
        }
    } catch (error) {
        console.error('Error loading violations:', error);
        const violationsList = document.getElementById('violationsList');
        if (violationsList) {
            violationsList.innerHTML = '<p style="text-align: center; padding: 20px; color: #dc2626;">Error loading violations</p>';
        }
    }
}

// Forgot Password Functions
function showForgotPasswordModal() {
    document.getElementById('forgotPasswordModal').style.display = 'flex';
}

function closeForgotPasswordModal() {
    document.getElementById('forgotPasswordModal').style.display = 'none';
    document.getElementById('resetEmail').value = '';
    document.getElementById('resetTokenDisplay').style.display = 'none';
}

async function requestPasswordReset() {
    const email = document.getElementById('resetEmail').value.trim();
    if (!email) {
        alert('Please enter your email');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/auth/forgot-password`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email: email })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Show reset token (for testing - in production, this would be sent via email)
            const tokenDisplay = document.getElementById('resetTokenDisplay');
            tokenDisplay.innerHTML = `<strong>Reset Token:</strong> ${data.reset_token}<br><small>Use this token to reset your password. In production, this would be sent via email.</small>`;
            tokenDisplay.style.display = 'block';
            alert('Password reset token generated. Check the token below or your email.');
        } else {
            alert(data.detail || 'Failed to generate reset token');
        }
    } catch (error) {
        console.error('Error requesting password reset:', error);
        alert('Error requesting password reset. Please try again.');
    }
}

// Users Management Functions
async function showUsersView() {
    document.getElementById('dashboard-view').style.display = 'none';
    document.getElementById('camera-view').style.display = 'none';
    document.getElementById('violations-view').style.display = 'none';
    document.getElementById('users-view').style.display = 'block';
    
    // Reset active class
    document.querySelectorAll('.nav-item a').forEach(el => el.classList.remove('active'));
    
    await loadUsers();
}

async function loadUsers() {
    try {
        const response = await apiRequest('/api/admin/users');
        const data = await response.json();
        
        const usersList = document.getElementById('usersList');
        if (!usersList) return;
        
        if (data.users && data.users.length > 0) {
            usersList.innerHTML = data.users.map(user => `
                <div class="user-item" style="display: flex; justify-content: space-between; align-items: center; padding: 15px; border-bottom: 1px solid #e5e7eb;">
                    <div>
                        <h4>${user.full_name}</h4>
                        <p style="color: #6b7280; font-size: 14px;">${user.email}</p>
                        <p style="color: #9ca3af; font-size: 12px;">
                            ${user.last_login ? `Last login: ${new Date(user.last_login).toLocaleString()}` : 'Never logged in'}
                        </p>
                    </div>
                    <button onclick="removeAdmin(${user.id})" class="btn-remove" style="background: #dc2626; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer;">
                        Remove
                    </button>
                </div>
            `).join('');
        } else {
            usersList.innerHTML = '<p style="text-align: center; padding: 20px; color: #6b7280;">No users found</p>';
        }
    } catch (error) {
        console.error('Error loading users:', error);
        alert('Error loading users. Please try again.');
    }
}

async function addAdmin() {
    const email = document.getElementById('newAdminEmail')?.value.trim();
    const password = document.getElementById('newAdminPassword')?.value;
    const fullName = document.getElementById('newAdminName')?.value.trim();
    
    if (!email || !password || !fullName) {
        alert('Please fill in all fields');
        return;
    }
    
    if (password.length < 6) {
        alert('Password must be at least 6 characters long');
        return;
    }
    
    try {
        const response = await apiRequest('/api/admin/users', {
            method: 'POST',
            body: JSON.stringify({
                email: email,
                password: password,
                full_name: fullName
            })
        });
        
        if (response.ok) {
            alert('Admin added successfully!');
            document.getElementById('newAdminEmail').value = '';
            document.getElementById('newAdminPassword').value = '';
            document.getElementById('newAdminName').value = '';
            await loadUsers();
        } else {
            const data = await response.json();
            alert(data.detail || 'Failed to add admin');
        }
    } catch (error) {
        console.error('Error adding admin:', error);
        alert('Error adding admin. Please try again.');
    }
}

async function removeAdmin(userId) {
    if (!confirm('Are you sure you want to remove this admin? They will no longer be able to login.')) {
        return;
    }
    
    try {
        const response = await apiRequest(`/api/admin/users/${userId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            alert('Admin removed successfully!');
            await loadUsers();
        } else {
            const data = await response.json();
            alert(data.detail || 'Failed to remove admin');
        }
    } catch (error) {
        console.error('Error removing admin:', error);
        alert('Error removing admin. Please try again.');
    }
}

// Original loadViolations function (if it exists)
async function loadViolationsOriginal() {
    const period = document.getElementById('violationPeriod')?.value || 'all';
    const cameraId = document.getElementById('violationCamera')?.value === 'all' ? null : 
                     parseInt(document.getElementById('violationCamera')?.value);
    
    let startDate = null;
    let endDate = null;
    
    if (period === 'today') {
        startDate = new Date().toISOString().split('T')[0];
    } else if (period === 'week') {
        const weekAgo = new Date();
        weekAgo.setDate(weekAgo.getDate() - 7);
        startDate = weekAgo.toISOString().split('T')[0];
    } else if (period === 'month') {
        const monthAgo = new Date();
        monthAgo.setMonth(monthAgo.getMonth() - 1);
        startDate = monthAgo.toISOString().split('T')[0];
    }
    
    try {
        let url = `${API_BASE_URL}/api/violations?limit=100`;
        if (startDate) url += `&start_date=${startDate}`;
        if (cameraId) url += `&camera_id=${cameraId}`;
        
        const response = await apiRequest(url.replace(API_BASE_URL, ''));
        const data = await response.json();
        
        displayViolations(data.violations);
    } catch (error) {
        console.error('Error loading violations:', error);
        document.getElementById('violationsList').innerHTML = 
            '<div style="text-align: center; padding: 40px; color: #ef4444;">Error loading violations</div>';
    }
}

function displayViolations(violations) {
    const container = document.getElementById('violationsList');
    
    if (violations.length === 0) {
        container.innerHTML = '<div style="text-align: center; padding: 40px;">No violations found</div>';
        return;
    }
    
    container.innerHTML = violations.map(v => `
        <div class="violation-item">
            <div class="violation-info">
                <h4>${v.detection_class} - ${(v.confidence * 100).toFixed(1)}%</h4>
                <p>Camera ${v.camera_id} • ${new Date(v.timestamp).toLocaleString()}</p>
                ${v.location ? `<p>Location: ${v.location}</p>` : ''}
            </div>
            <button onclick="downloadViolationVideo(${v.id})" class="btn-download">Download Video</button>
        </div>
    `).join('');
}

// Download violation video with authentication
async function downloadViolationVideo(violationId) {
    try {
        const token = getAuthToken();
        if (!token) {
            alert('Please login to download videos');
            showLoginPage();
            return;
        }
        
        // Fetch video with authentication
        const response = await apiRequest(`/api/violations/${violationId}/video`);
        
        if (!response.ok) {
            if (response.status === 401) {
                alert('Session expired. Please login again.');
                removeAuthToken();
                showLoginPage();
                return;
            }
            throw new Error('Failed to download video');
        }
        
        // Get video blob
        const blob = await response.blob();
        
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `violation_${violationId}.mp4`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Error downloading video:', error);
        alert('Failed to download video. Please try again.');
    }
}