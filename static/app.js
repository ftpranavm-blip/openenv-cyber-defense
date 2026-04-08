const API_URL = "http://localhost:5000/api";

const DOM = {
    stepBadge: document.getElementById('step-badge'),
    totalReward: document.getElementById('total-reward'),
    pSeverity: document.getElementById('p-severity'),
    pZone: document.getElementById('p-zone'),
    pScoreBar: document.getElementById('p-score-bar'),
    patientCard: document.getElementById('patient-card'),
    eventLog: document.getElementById('event-log'),
    pCondition: document.getElementById('p-condition')
};

// Map nodes dynamically to their elements
const hospDOM = {
    A: document.getElementById('hosp-A'),
    B: document.getElementById('hosp-B'),
    C: document.getElementById('hosp-C')
};

// --- LEAFLET MAP INTEGRATION ---
let map, ambuMarker, routeLine, currentState = null;

const COORDS = {
    center: [19.0760, 72.8777], // Mumbai (Central)
    A: [19.0800, 72.8800],      // City General
    B: [19.1200, 72.8500],      // Metro Trauma
    C: [18.9500, 72.8200]       // South District
};

const ZONES = {
    'central': COORDS.center,
    'north': [19.1500, 72.8500],
    'south': [18.9300, 72.8200],
    'west': [19.0500, 72.8300],
    'east': [19.0500, 72.9000]
};

function initMap() {
    map = L.map('map-container', { zoomControl: false }).setView(COORDS.center, 11);
    
    // Premium Dark CartoDB basemap matching the Antigravity theme
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(map);

    // Add Hospitals
    const createMarker = (coord, label) => {
        let nameMap = { 'A': 'Lilavati Hospital', 'B': 'KEM Hospital', 'C': 'Breach Candy' };
        const icon = L.divIcon({
            className: 'hosp-marker',
            html: `<span>${label}</span>`,
            iconSize: [24, 24]
        });
        L.marker(coord, { icon }).addTo(map).bindPopup(`<b>${nameMap[label]}</b>`);
    };

    createMarker(COORDS.A, 'A');
    createMarker(COORDS.B, 'B');
    createMarker(COORDS.C, 'C');

    // Add Ambulance
    const ambIcon = L.divIcon({
        className: 'ambulance-marker',
        html: `🚑`,
        iconSize: [24, 24],
        iconAnchor: [12, 12]
    });
    ambuMarker = L.marker(COORDS.center, { icon: ambIcon, zIndexOffset: 1000 }).addTo(map);

    // Line holding route
    routeLine = L.polyline([], { color: '#4F86F7', weight: 3, dashArray: '5, 10' }).addTo(map);
}

// Map Animation Engine
function animateRouting(targetHostpitalKey) {
    const start = currentState && currentState.patient_zone ? ZONES[currentState.patient_zone] || COORDS.center : COORDS.center;

    if (!['A', 'B', 'C'].includes(targetHostpitalKey)) {
        ambuMarker.setLatLng(start);
        routeLine.setLatLngs([]);
        return;
    }

    const end = COORDS[targetHostpitalKey];
    
    // Evaluate Real-World Traffic State
    let travelTime = 20;
    if (currentState) {
        travelTime = currentState[`hospital_${targetHostpitalKey}_travel_time`] || 20;
    }
    
    let routeColor = '#4F86F7';
    if (travelTime > 30) {
        routeColor = '#ff4d4d'; // Heavy traffic (Red)
    } else if (travelTime > 18) {
        routeColor = '#ffb84d'; // Moderate (Orange)
    } else {
        routeColor = '#4dff88'; // Clear (Green)
    }
    
    routeLine.setStyle({ color: routeColor });
    routeLine.setLatLngs([start, end]);
    
    // Dynamic interpolation matching traffic gridlock severity
    let startTs = null;
    const dur = Math.min(3000, Math.max(500, travelTime * 40)); // ms

    function anim(ts) {
        if (!startTs) startTs = ts;
        const prog = Math.min((ts - startTs) / dur, 1);
        
        // Easing out quint
        const ease = 1 - Math.pow(1 - prog, 5);
        
        const lat = start[0] + (end[0] - start[0]) * ease;
        const lng = start[1] + (end[1] - start[1]) * ease;
        
        ambuMarker.setLatLng([lat, lng]);
        
        if (prog < 1) requestAnimationFrame(anim);
        else {
             // Reset back to center after dispatch completes
             setTimeout(() => {
                 ambuMarker.setLatLng(start);
                 routeLine.setLatLngs([]);
             }, 500);
        }
    }
    requestAnimationFrame(anim);
}

// --- STATE MANAGEMENT ---

document.addEventListener("DOMContentLoaded", () => {
    initMap();
    refreshState();
});

async function refreshState() {
    try {
        const res = await fetch(`${API_URL}/state`);
        const data = await res.json();
        if (data.status === 'success') {
            updateUI(data.state);
        } else {
            showToast("Failed to fetch state", true);
        }
    } catch (err) {
        showToast("Server connection error", true);
    }
}

async function takeAction(actionName) {
    // Fire map animation concurrently with HTTP request
    if (actionName.startsWith('route_to_hospital_')) {
        animateRouting(actionName.split('_').pop());
    } else {
        animateRouting('reset');
    }

    try {
        const res = await fetch(`${API_URL}/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: actionName })
        });
        
        const data = await res.json();
        if (data.status === 'success') {
            updateUI(data.state);
            logEvent(data.info);
            // Flash reward
            const rwd = data.reward;
            if (rwd > 0) showToast(`+${rwd.toFixed(2)} Reward`, false);
        } else {
            showToast(data.message, true);
        }
    } catch (err) {
        showToast("Error communicating with server", true);
    }
}

async function resetEnv() {
    try {
        const res = await fetch(`${API_URL}/reset`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
            DOM.eventLog.innerHTML = "";
            showToast("Simulation Reset.");
            updateUI(data.state);
            ambuMarker.setLatLng(COORDS.center);
            routeLine.setLatLngs([]);
        }
    } catch (err) {
        showToast("Reset failed.", true);
    }
}

// --- UI UPDATERS ---

function updateUI(state) {
    if(!state) return;
    currentState = state;
    
    // Core details
    DOM.stepBadge.innerText = `${state.sim_step} / 20`;
    DOM.pSeverity.innerText = state.patient_severity.toUpperCase();
    DOM.pCondition.innerText = state.patient_condition || "UNKNOWN";
    DOM.pZone.innerText = state.patient_zone.toUpperCase();
    
    // Condition Score Bar
    const scorePct = Math.round(state.patient_condition_score * 100);
    DOM.pScoreBar.style.width = `${scorePct}%`;
    
    if (state.patient_severity === 'critical') {
        DOM.patientCard.classList.add('critical');
        DOM.pSeverity.classList.add('critical');
    } else {
        DOM.patientCard.classList.remove('critical');
        DOM.pSeverity.classList.remove('critical');
    }

    // Telemetry updates
    if (state.hospitals) {
        state.hospitals.forEach((h, i) => {
            const label = ["A", "B", "C"][i];
            const node = hospDOM[label];
            node.querySelector('.hosp-beds').innerText = state[`hospital_${label}_available_beds`];
            node.querySelector('.hosp-icu').innerText = state[`hospital_${label}_icu_available`];
            node.querySelector('.hosp-wait').innerText = state[`hospital_${label}_wait_time`];
            node.querySelector('.hosp-rating-val').innerText = state[`hospital_${label}_rating`].toFixed(1);
            node.querySelector('.hosp-spec').innerText = state[`hospital_${label}_specialty`];
        });
    }

    // Cumulative Reward Counter
    const totalRwd = state._episode_history ? 
        state._episode_history.reduce((sum, ev) => sum + ev.reward, 0) : 0;
    
    animateValue(DOM.totalReward, parseFloat(DOM.totalReward.innerText) || 0, totalRwd, 500);
}

function logEvent(info) {
    if (!info) return;
    const div = document.createElement('div');
    div.className = 'log-entry';
    
    const surv_icon = info.survived ? '🟢' : '🔴';
    const rClass = info.reward >= 0 ? "pos" : "neg";
    const rText = info.reward >= 0 ? `+${info.reward}` : info.reward;
    
    div.innerHTML = `
        <span>${surv_icon}</span> <span class="log-action">${info.action}</span> 
        &rarr; <span>${info.hospital || "N/A"}</span> 
        <span style="float:right;" class="log-rwd ${rClass}">${rText}</span>
    `;
    
    DOM.eventLog.prepend(div);
}

function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = (start + progress * (end - start)).toFixed(2);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

function showToast(msg, isError = false) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${isError ? 'error' : ''}`;
    toast.innerText = msg;
    
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

async function startLLMPass() {
    const btn = document.getElementById('llm-btn');
    const oldHTML = btn.innerHTML;
    btn.innerHTML = `Running AI Dispatch...`;
    btn.style.opacity = '0.6';
    btn.style.pointerEvents = 'none';

    showToast("Gemini AI evaluating conditions & traffic...");
    try {
        const res = await fetch(`${API_URL}/autopilot`, { method: 'POST' });
        const data = await res.json();
        
        if (data.status === 'success') {
            // Found action, trigger standard dispatch
            showToast(`Autopilot selected: ${data.action.toUpperCase()}`);
            takeAction(data.action);
        } else {
            showToast("Autopilot error: " + data.message, true);
        }
    } catch(err) {
        showToast("Autopilot connection failed", true);
    } finally {
        btn.innerHTML = oldHTML;
        btn.style.opacity = '1';
        btn.style.pointerEvents = 'auto';
    }
}
