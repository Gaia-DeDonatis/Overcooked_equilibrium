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
    const pages = [
        'page-intro', 'page-consent', 'page-instruction-1',
        'page-instruction-2a','page-instruction-2b','page-instruction-2c',
        'page-phase-1', 'page-intermission', 'page-phase-2', 
        'page-qs', 'page-end'
    ];
    
    pages.forEach(id => {
        const el = document.getElementById(id);
        if(el) el.classList.add('hidden');
    });

    const target = document.getElementById(pageId);
    if(target) target.classList.remove('hidden');
    window.scrollTo(0,0);

    // CHANGE THIS: Only set isPlaying to false if we aren't going to a game page
    const gamePages = ['page-phase-1', 'page-phase-2', 'page-task', 'page-instruction-1'];
    if (!gamePages.includes(pageId)) {
        STATE.isPlaying = false;
    }
}

// --- 3. KEYBOARD LISTENER (COMBINED) ---
document.addEventListener('keydown', async (e) => {
    // Standard Filters
    if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].indexOf(e.key) === -1) return;
    if(!STATE.isPlaying || STATE.gameOver) return;
    
    e.preventDefault();

    // A. PRACTICE LOGIC
    if(STATE.phase === 0) {
        const data = await api('/key_event', { key: e.key, config_id: 'layout_practice' });
        drawGame(data.state, 'gameCanvas_practice');
        
        const prevScore = STATE.practiceScore;
        STATE.practiceScore = data.cumulative_reward || 0;

        if(STATE.practiceScore >= 200 && STATE.practiceScore > prevScore) {
            STATE.isPlaying = false;
            STATE.gameOver = true;
            document.getElementById('practiceHint').innerText = "Great job! Click 'Next' to continue.";
            document.getElementById('to-instruction-2').disabled = false;
            alert("Practice Complete! You delivered the salad.");
        }
    } 
    // B. MAIN TASK LOGIC (Phase 1 or 2)
    else {
    const data = await api('/key_event', { key: e.key, config_id: STATE.configId });
    drawGame(data.state, 'gameCanvas');
    document.getElementById('stepsLeft').innerText = data.steps_left;
    
    // Make sure this variable name matches your config.js
    currentRoundSteps.push({ key:e.key, reward:data.cumulative_reward, steps:data.steps_left });

    if(data.steps_left <= 0) {
        STATE.isPlaying = false;
        STATE.gameOver = true;
        document.getElementById('round-end-panel').classList.remove('hidden');
    }
    }
});

// --- 4. NAVIGATION & BUTTONS ---

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

const check = document.getElementById('consentCheck');
const btnInst = document.getElementById('to-instruction');
if(check) {
    check.addEventListener('change', () => btnInst.disabled = !check.checked);
    btnInst.onclick = () => showPage('page-instruction-1');
}

// --- 4. NAVIGATION & BUTTONS ---

// 1. Validation for Quiz 2a
const btnNext2a = document.getElementById('btn-next-2a');
const q1Radios = document.getElementsByName('q1');
if(btnNext2a) {
    q1Radios.forEach(r => r.addEventListener('change', (e) => {
        btnNext2a.disabled = (e.target.value !== 'correct');
    }));
}

// 2. Validation for Quiz 2b
const btnNext2b = document.getElementById('btn-next-2b');
const q2Radios = [...document.getElementsByName('q2a'), ...document.getElementsByName('q2b')];
function val2b() {
    const a = document.querySelector('input[name="q2a"]:checked')?.value === 'correct';
    const b = document.querySelector('input[name="q2b"]:checked')?.value === 'correct';
    btnNext2b.disabled = !(a && b);
}
q2Radios.forEach(r => r.addEventListener('change', val2b));

// 3. Validation for Quiz 2c
const btnStart1 = document.getElementById('start-task-1');
const q3Radios = [...document.getElementsByName('q3a'), ...document.getElementsByName('q3b')];
function val2c() {
    const a = document.querySelector('input[name="q3a"]:checked')?.value === 'correct';
    const b = document.querySelector('input[name="q3b"]:checked')?.value === 'correct';
    btnStart1.disabled = !(a && b);
}
q3Radios.forEach(r => r.addEventListener('change', val2c));

// Flow Buttons
document.getElementById('to-instruction-2').onclick = () => showPage('page-instruction-2a');
document.getElementById('btn-next-2a').onclick = () => showPage('page-instruction-2b');
document.getElementById('btn-next-2b').onclick = () => showPage('page-instruction-2c');

document.getElementById('start-task-1').onclick = () => showPage('page-phase-1');
document.getElementById('btnFinishPhase1').onclick = () => showPage('page-intermission');
document.getElementById('btnStartPhase2').onclick = () => showPage('page-phase-2');
document.getElementById('btnFinishPhase2').onclick = () => showPage('page-qs');

// --- 5. FINAL SUBMISSION ---
document.getElementById('qsNext').onclick = () => {
    const strategy = document.getElementById('q_strategy').value;
    const adaptation = document.getElementById('q_adaptation').value;
    const workflow = document.getElementById('q_workflow').value;
    const trust = document.getElementById('q_trust').value;
    const experience = document.querySelector('input[name="q_experience"]:checked');

    if(strategy.length < 5 || adaptation.length < 5 || workflow.length < 5 || trust.length < 5 || !experience) {
        return alert("Please provide an answer for all questions.");
    }

    LOGS.questionnaire = {
        strategy_text: strategy, adaptation_text: adaptation,
        workflow_text: workflow, trust_reliance: trust,
        played_overcooked_before: experience.value
    };

    showPage('page-end');
    document.getElementById('completionCodeWrap').classList.remove('hidden');
    document.getElementById('completionCode').innerText = "C15-協同-2026";
    document.getElementById('completionHint').classList.add('hidden');
    document.getElementById('btnFinish').classList.add('hidden');
};

window.onload = () => {
    preloadImages(() => {
        console.log("Images loaded and game is ready.");
    });
};