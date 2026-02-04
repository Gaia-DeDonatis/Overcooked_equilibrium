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
const q1Radios = document.getElementsByName('q1'); // Matches your new HTML
const q1Error = document.getElementById('q1-error'); // Your error message box

if(btnNext2a && q1Radios.length > 0) {
    // 1. Force Disable Initially
    btnNext2a.disabled = true;

    // 2. Listen for Selection
    q1Radios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            const isCorrect = (e.target.value === 'correct');
            
            if (isCorrect) {
                // CORRECT: Enable button, hide error
                btnNext2a.disabled = false;
                if(q1Error) q1Error.classList.add('hidden');
            } else {
                // WRONG: Disable button, show error
                btnNext2a.disabled = true;
                if(q1Error) q1Error.classList.remove('hidden');
                
                // Optional: Uncheck the wrong answer after a split second
                setTimeout(() => { e.target.checked = false; }, 500);
            }
        });
    });

    // 3. Navigation to Page 2b
    btnNext2a.onclick = () => showPage('page-instruction-2b');
}

// PAGE 2b
const btnNext2b = document.getElementById('btn-next-2b');
const q2aRadios = document.getElementsByName('q2a');
const q2bRadios = document.getElementsByName('q2b');

function validateQuiz2b() {
    // Check if Q2a correct
    let q2aCorrect = false;
    q2aRadios.forEach(r => { if(r.checked && r.value === 'correct') q2aCorrect = true; });

    // Check if Q2b correct
    let q2bCorrect = false;
    q2bRadios.forEach(r => { if(r.checked && r.value === 'correct') q2bCorrect = true; });

    // Enable only if BOTH are correct
    if(btnNext2b) btnNext2b.disabled = !(q2aCorrect && q2bCorrect);
}

if(btnNext2b) {
    // Attach listeners
    [...q2aRadios, ...q2bRadios].forEach(r => r.addEventListener('change', validateQuiz2b));
    
    // Navigation
    btnNext2b.onclick = () => showPage('page-instruction-2c');
}


// PAGE 2c
const btnStartTask1 = document.getElementById('start-task-1');
const q3aRadios = document.getElementsByName('q3a');
const q3bRadios = document.getElementsByName('q3b');
const q3Error = document.getElementById('q3-error');

function validateQuiz2c() {
    // Check if Q3a correct
    let q3aCorrect = false;
    q3aRadios.forEach(r => { if(r.checked && r.value === 'correct') q3aCorrect = true; });

    // Check if Q3b correct
    let q3bCorrect = false;
    q3bRadios.forEach(r => { if(r.checked && r.value === 'correct') q3bCorrect = true; });

    const allCorrect = q3aCorrect && q3bCorrect;

    if(btnStartTask1) btnStartTask1.disabled = !allCorrect;
    
    // Show/Hide Error Message
    if(q3Error) {
        if(allCorrect) q3Error.classList.add('hidden');
        else if (document.querySelector('input[name="q3a"]:checked') || document.querySelector('input[name="q3b"]:checked')) {
            // Show error if they started answering but got it wrong/incomplete
            q3Error.classList.remove('hidden');
        }
    }
}

if(btnStartTask1) {
    // Attach listeners
    [...q3aRadios, ...q3bRadios].forEach(r => r.addEventListener('change', validateQuiz2c));

    // Navigation (Start Phase 1)
    btnStartTask1.onclick = () => {
        showPage('page-pretask-1');
        startPreview('preview1');
    };
}


// Phase 1
document.getElementById('start-task-1').onclick = () => { showPage('page-pretask-1'); startPreview('preview1'); };
document.getElementById('btnEnterTask1').onclick = () => { STATE.round=1; showPage('page-task'); startMainTask('task1'); };

// Phase 2
document.getElementById('btnEnterTask2').onclick = () => { STATE.round=1; showPage('page-task'); startMainTask('task2'); };

// Next Round
// PAGE 2c & START PHASE 1
if(btnStartTask1) {
    // Attach listeners
    [...q3aRadios, ...q3bRadios].forEach(r => r.addEventListener('change', validateQuiz2c));

    // NAVIGATION: Go straight to Game (Skip Pre-task)
    btnStartTask1.onclick = () => {
        STATE.round = 1;
        startMainTask('task1'); 
    };
}

// INTERMISSION: START PHASE 2
const btnStartPhase2 = document.getElementById('btnStartPhase2');
if(btnStartPhase2) {
    btnStartPhase2.onclick = () => {
        STATE.round = 1; // Reset round count for Phase 2
        startMainTask('task2');
    };
}

// NEXT ROUND BUTTON
document.getElementById('btnNext').onclick = () => {
    // 1. Save Data (Rating removed as requested)
    LOGS.rounds.push({ 
        phase: STATE.phase, 
        round: STATE.round, 
        config: STATE.configId, 
        steps: currentRoundSteps 
    });

    // 2. Logic
    if(STATE.round < 5) {
        // --- NEXT ROUND ---
        STATE.round++;
        document.getElementById('roundLabel').innerText = `Round ${STATE.round} / 5`;
        document.getElementById('round-end-panel').classList.add('hidden');
        
        // Restart same phase
        startMainTask(STATE.phase === 1 ? 'task1' : 'task2');
    } else {
        // --- PHASE FINISHED ---
        if(STATE.phase === 1) {
            // End of Phase 1 -> Show Intermission
            document.getElementById('round-end-panel').classList.add('hidden');
            showPage('page-intermission');
        } else {
            // End of Phase 2 -> Show Questionnaire
            showPage('page-qs');
        }
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