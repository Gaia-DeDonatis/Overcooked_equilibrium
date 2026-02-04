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

// --- 2. NAVIGATION & SETUP ---
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
                   'page-pretask-1','page-pretask-2','page-task','page-qs','page-end'];
    pages.forEach(id => {
        const el = document.getElementById(id);
        if(el) el.classList.add('hidden');
    });
    const target = document.getElementById(pageId);
    if(target) target.classList.remove('hidden');
    window.scrollTo(0,0);
    STATE.isPlaying = false;
}

// --- 3. MAIN TASK LOGIC (Phase 1 & 2) ---
async function startMainTask(mode) {
    STATE.phase = (mode === 'task1') ? 1 : 2;
    STATE.configId = (mode === 'task1') ? STATE.assignment.task1 : STATE.assignment.task2;
    STATE.isPlaying = true;
    STATE.gameOver = false;
    currentRoundSteps = [];

    const data = await api('/reset', { config_id: STATE.configId });
    drawGame(data.state, 'gameCanvas');
    document.getElementById('stepsLeft').innerText = data.steps_left || "—";
}

async function startPreview(mode) {
    const cfg = (mode === 'preview1') ? STATE.assignment.task1 : STATE.assignment.task2;
    const canvas = (mode === 'preview1') ? 'gameCanvas_pretask1' : 'gameCanvas_pretask2';
    const data = await api('/reset', { config_id: cfg });
    drawGame(data.state, canvas);
}

// Main Task Input Handler
document.addEventListener('keydown', async (e) => {
    if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].indexOf(e.key) === -1) return;
    if(!STATE.isPlaying || STATE.gameOver) return;
    if(STATE.phase === 0) return; // IGNORE if in Practice Phase

    e.preventDefault();
    const data = await api('/key_event', { key: e.key, config_id: STATE.configId });
    
    drawGame(data.state, 'gameCanvas');
    document.getElementById('stepsLeft').innerText = data.steps_left;
    
    currentRoundSteps.push({ key:e.key, reward:data.cumulative_reward, steps:data.steps_left });

    if(data.steps_left <= 0) {
        STATE.isPlaying = false;
        STATE.gameOver = true;
        document.getElementById('round-end-panel').classList.remove('hidden');
    }
});

// --- 4. BUTTON LISTENERS ---

// Intro Validation
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

// Consent
const check = document.getElementById('consentCheck');
const btnInst = document.getElementById('to-instruction');
if(check) {
    check.addEventListener('change', () => btnInst.disabled = !check.checked);
    btnInst.onclick = () => showPage('page-instruction-1');
}

// Navigation (Standard)
document.getElementById('to-instruction-2').onclick = () => showPage('page-instruction-2a');

// PAGE 2a
const btnNext2a = document.getElementById('btn-next-2a');
const goalRadios = document.getElementsByName('goalQuiz'); // Matches HTML name

if(btnNext2a && goalRadios.length > 0) {
    // 1. Force Disable Initially (Just in case HTML didn't)
    btnNext2a.disabled = true;

    // 2. Listen for Selection
    goalRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            const isCorrect = (e.target.value === 'correct');
            
            // Unlock if correct
            btnNext2a.disabled = !isCorrect;

            if(!isCorrect) {
                alert("Incorrect. To get a high score, you should coordinate to minimize steps.");
                e.target.checked = false; // Reset the wrong choice
            }
        });
    });

    // 3. Navigation
    btnNext2a.onclick = () => showPage('page-instruction-2b');
}
// === FIX ENDS HERE ===

document.getElementById('btn-next-2b').onclick = () => showPage('page-instruction-2c');


// Phase 1
document.getElementById('start-task-1').onclick = () => { showPage('page-pretask-1'); startPreview('preview1'); };
document.getElementById('btnEnterTask1').onclick = () => { STATE.round=1; showPage('page-task'); startMainTask('task1'); };

// Phase 2
document.getElementById('btnEnterTask2').onclick = () => { STATE.round=1; showPage('page-task'); startMainTask('task2'); };

// Next Round
document.getElementById('btnNext').onclick = () => {
    let r = document.querySelector('input[name="personaFidelity"]:checked');
    if(!r) return alert("Please rate the AI.");
    
    LOGS.rounds.push({ phase: STATE.phase, round: STATE.round, config: STATE.configId, steps: currentRoundSteps, rating: r.value });
    
    if(STATE.round < CONFIG.TOTAL_ROUNDS) {
        STATE.round++;
        document.getElementById('roundLabel').innerText = `Round ${STATE.round} / ${CONFIG.TOTAL_ROUNDS}`;
        document.getElementById('round-end-panel').classList.add('hidden');
        document.querySelectorAll('input[name="personaFidelity"]').forEach(el => el.checked=false);
        startMainTask(STATE.phase === 1 ? 'task1' : 'task2');
    } else {
        if(STATE.phase === 1) { showPage('page-pretask-2'); startPreview('preview2'); }
        else { showPage('page-qs'); }
    }
};

// Submit
document.getElementById('qsNext').onclick = () => {
    const s = document.getElementById('q_strategy').value;
    const a = document.querySelector('input[name="q_adapt"]:checked');
    if(!s || !a) return alert("Answer all questions.");
    LOGS.questionnaire = { strategy: s, adapt: a.value };
    api('/submit_log', { log: LOGS }).then(res => {
        document.getElementById('completionCode').innerText = res.completion_code;
        showPage('page-end');
    });
};

// Init
preloadImages(() => {
    console.log("Images Loaded");
    showPage('page-intro');
});