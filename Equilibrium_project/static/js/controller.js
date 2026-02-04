// static/js/controller.js

// --- 1. API HELPER ---
async function api(endpoint, data={}) {
    if(!STATE.sessionId) {
        const res = await fetch(`${SERVER_URL}/new_session`, { method:'POST'});
        const d = await res.json();
        STATE.sessionId = d.session_id;
        localStorage.setItem('session_id', d.session_id);
    }
    const res = await fetch(`${SERVER_URL}${endpoint}`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({...data, session_id: STATE.sessionId})
    });
    return await res.json();
}

// --- 2. CONFIG & NAVIGATION ---
function assignConditions() {
    const pid = LOGS.prolificId || "test";
    let h = 0;
    for(let i=0; i<pid.length; i++) h = (h*31 + pid.charCodeAt(i)) >>> 0;
    const layouts = ["layout1", "layout2", "layout3", "layout4"];
    const models = ["model1", "model2", "model3", "model4"];
    
    STATE.assignment.task1 = `${layouts[h%4]}_${models[h%4]}`;
    STATE.assignment.task2 = `${layouts[h%4]}_${models[(h+1)%4]}`;
    LOGS.assignment = STATE.assignment;
}

function showPage(pageId) {
    const pages = ['page-intro','page-consent','page-instruction-1',
                   'page-instruction-2a','page-instruction-2b','page-instruction-2c',
                   'page-intermission','page-task','page-qs','page-end'];
    
    pages.forEach(id => {
        const el = document.getElementById(id);
        if(el) el.classList.add('hidden');
    });
    const target = document.getElementById(pageId);
    if(target) target.classList.remove('hidden');
    window.scrollTo(0,0);
    STATE.isPlaying = false;
}

// --- 3. MAIN GAME LOGIC (Phase 1 & Phase 2 ONLY) ---
async function startMainTask(mode) {
    // 1. Setup Phase State
    if (mode === 'task1') {
        STATE.phase = 1;
        STATE.configId = STATE.assignment.task1;
        document.getElementById('taskTag').innerText = "Phase 1";
        document.getElementById('taskTag').style.backgroundColor = "#2563eb"; 
    } else {
        STATE.phase = 2;
        STATE.configId = STATE.assignment.task2;
        document.getElementById('taskTag').innerText = "Phase 2";
        document.getElementById('taskTag').style.backgroundColor = "#16a34a"; 
    }

    // 2. Setup UI
    showPage('page-task');
    document.getElementById('roundLabel').innerText = `Round ${STATE.round} / 5`;
    document.getElementById('round-end-panel').classList.add('hidden');
    
    // 3. Reset Server
    STATE.isPlaying = true;
    STATE.gameOver = false;
    currentRoundSteps = [];

    const data = await api('/reset', { config_id: STATE.configId });
    drawGame(data.state, 'gameCanvas');
    document.getElementById('stepsLeft').innerText = data.steps_left || "45";
}

// --- 4. KEYBOARD LISTENER ---
document.addEventListener('keydown', async (e) => {
    // Filter Keys
    if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].indexOf(e.key) === -1) return;
    
    // Filter State
    if(!STATE.isPlaying || STATE.gameOver) return;
    
    // CRITICAL: Ignore Practice Phase (Let practice.js handle it)
    if(STATE.phase === 0) return; 

    e.preventDefault();

    // Send Key (Phase 1 or 2)
    const data = await api('/key_event', { key: e.key, config_id: STATE.configId });
    
    // Update Game
    drawGame(data.state, 'gameCanvas');
    document.getElementById('stepsLeft').innerText = data.steps_left;
    
    currentRoundSteps.push({ key:e.key, reward:data.cumulative_reward, steps:data.steps_left });

    // Check End of Round
    if(data.steps_left <= 0) {
        STATE.isPlaying = false;
        STATE.gameOver = true;
        document.getElementById('round-end-panel').classList.remove('hidden');
    }
});

// --- 5. NAVIGATION & BUTTONS ---

// Intro
const inputID = document.getElementById('prolificId');
const inputAge = document.getElementById('age');
const inputGender = document.getElementById('gender');
const btnConsent = document.getElementById('to-consent');

function validate() {
    if(!inputID || !inputAge || !inputGender) return;
    btnConsent.disabled = !(inputID.value.trim().length > 0 && parseInt(inputAge.value) >= 18 && inputGender.value !== "");
}
if(inputID) {
    [inputID, inputAge, inputGender].forEach(el => el.addEventListener('input', validate));
    inputGender.addEventListener('change', validate);
    btnConsent.onclick = () => {
        LOGS.prolificId = inputID.value.trim();
        assignConditions();
        showPage('page-consent');
    };
}
const check = document.getElementById('consentCheck');
const btnInst = document.getElementById('to-instruction');
if(check) {
    check.addEventListener('change', () => btnInst.disabled = !check.checked);
    btnInst.onclick = () => showPage('page-instruction-1');
}

// Navigation
document.getElementById('to-instruction-2').onclick = () => showPage('page-instruction-2a');

// Quiz 2a
const btnNext2a = document.getElementById('btn-next-2a');
const q1Radios = document.getElementsByName('q1');
const q1Error = document.getElementById('q1-error');
if(btnNext2a && q1Radios.length > 0) {
    btnNext2a.disabled = true;
    q1Radios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            const isCorrect = (e.target.value === 'correct');
            btnNext2a.disabled = !isCorrect;
            if(q1Error) {
                if(isCorrect) q1Error.classList.add('hidden');
                else q1Error.classList.remove('hidden');
            }
        });
    });
    btnNext2a.onclick = () => showPage('page-instruction-2b');
}

// Quiz 2b
const btnNext2b = document.getElementById('btn-next-2b');
const q2a = document.getElementsByName('q2a');
const q2b = document.getElementsByName('q2b');
function val2b(){ 
    let a=false, b=false;
    q2a.forEach(r=>{if(r.checked&&r.value==='correct')a=true});
    q2b.forEach(r=>{if(r.checked&&r.value==='correct')b=true});
    if(btnNext2b) btnNext2b.disabled = !(a&&b);
}
[...q2a,...q2b].forEach(r=>r.addEventListener('change',val2b));
if(btnNext2b) btnNext2b.onclick = () => showPage('page-instruction-2c');

// Quiz 2c -> Start Phase 1
const btnStart1 = document.getElementById('start-task-1');
const q3a = document.getElementsByName('q3a');
const q3b = document.getElementsByName('q3b');
const q3Error = document.getElementById('q3-error');

function val2c(){
    let a=false, b=false;
    q3a.forEach(r=>{if(r.checked&&r.value==='correct')a=true});
    q3b.forEach(r=>{if(r.checked&&r.value==='correct')b=true});
    
    // Validate Button State
    if(btnStart1) btnStart1.disabled = !(a&&b);
    
    // Show/Hide Error Text
    if(q3Error) {
       if(a&&b) q3Error.classList.add('hidden');
       else if(document.querySelector('input[name="q3a"]:checked') || document.querySelector('input[name="q3b"]:checked')) 
           q3Error.classList.remove('hidden');
    }
}
// Attach listeners safely
[...q3a,...q3b].forEach(r=>r.addEventListener('change',val2c));

// 1. Start Phase 1 (Go to Placeholder)
if(btnStart1) {
    btnStart1.onclick = () => { 
        showPage('page-phase-1');
    };
}

// 2. Finish Phase 1 -> Intermission
const btnFinishPhase1 = document.getElementById('btnFinishPhase1');
if(btnFinishPhase1) {
    btnFinishPhase1.onclick = () => {
        showPage('page-intermission');
    };
}

// 3. Start Phase 2 -> Go to Placeholder
const btnStartPhase2 = document.getElementById('btnStartPhase2');
if(btnStartPhase2) {
    btnStartPhase2.onclick = () => {
        showPage('page-phase-2');
    };
}

// 4. Finish Phase 2 -> Questionnaire
const btnFinishPhase2 = document.getElementById('btnFinishPhase2');
if(btnFinishPhase2) {
    btnFinishPhase2.onclick = () => {
        showPage('page-qs');
    };
}

// ==========================================================
// === QUESTIONNAIRE & INIT ===
// ==========================================================

document.getElementById('qsNext').onclick = () => {
    const s = document.getElementById('q_strategy').value;
    const a = document.querySelector('input[name="q_adapt"]:checked');
    if(!s || !a) return alert("Please answer all questions.");
    
    document.getElementById('completionCode').innerText = "MVP-SUCCESS";
    showPage('page-end');
};

preloadImages(() => {
    console.log("Images Loaded");
    showPage('page-intro');
});