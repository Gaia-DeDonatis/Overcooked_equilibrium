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


// --- 2. CONDITION ASSIGNMENT ---
function assignConditions() {
    const pid = STATE.prolificId || "test";
    let h = 0;
    for(let i=0; i<pid.length; i++) h = (h*31 + pid.charCodeAt(i)) >>> 0;
    
    // Randomly assign layout and models
    const layouts = ["cramped", "circuit", "asymmetric", "ring", "forced"];
    const models = ["model1", "model2", "model3", "model4"];
    
    STATE.assignment.layout = layouts[h % layouts.length];
    
    // Assign Models (Phase 1 vs Phase 2)
    STATE.assignment.phase1Model = models[h % models.length];
    STATE.assignment.phase2Model = models[(h + 1) % models.length];
    
    console.log("Assigned:", STATE.assignment);
}

// --- 3. PAGE NAVIGATION ---
function showPage(pageId) {
    const pages = [
        'page-intro', 'page-consent', 'page-instruction-1',
        'page-instruction-2a','page-instruction-2b','page-instruction-2c',
        'page-phase-1', 'page-phase-1-qs', 'page-intermission', 
        'page-phase-2', 'page-phase-2-qs', 'page-end'
    ];
    
    pages.forEach(id => {
        const el = document.getElementById(id);
        if(el) el.classList.add('hidden');
    });

    const target = document.getElementById(pageId);
    if(target) target.classList.remove('hidden');
    window.scrollTo(0,0);

    // Only set isPlaying for game pages
    const gamePages = ['page-phase-1', 'page-phase-2', 'page-instruction-1'];
    if (!gamePages.includes(pageId)) {
        STATE.isPlaying = false;
    }
}

// --- 4. GAME INITIALIZATION (TIME-BASED) ---

let gameTimer = null;
let timeLeft = 0;

// A. START PHASE (Setup Model & Layout)
async function startPhase(phaseNum) {

    if (!STATE.assignment || !STATE.assignment.layout) {
            console.warn("Conditions missing. Auto-assigning defaults.");
            // Make sure these match your Backend keys exactly!
            STATE.assignment = {
                layout: "layout1", 
                phase1Model: "model1", 
                phase2Model: "model2"
            };
        }

    STATE.phase = phaseNum;
    STATE.round = 1;
    STATE.gameOver = false;
    
    // Determine which model to use
    let modelId = (phaseNum === 1) ? STATE.assignment.phase1Model : STATE.assignment.phase2Model;
    STATE.configId = `${STATE.assignment.layout}_${modelId}`;

    if (phaseNum === 1) {
        showPage('page-phase-1');
    } else if (phaseNum === 2) {
        showPage('page-phase-2');
    }
    
    console.log(`Starting Phase ${phaseNum} with ${STATE.configId}`);
    await startRound();
}

// B. START ROUND
async function startRound() {
    STATE.isPlaying = false;
    STATE.gameOver = false;
    
    if (gameTimer) clearInterval(gameTimer);

    DataManager.startNewRound(STATE.phase, STATE.configId);
    
    try {
        const data = await api('/reset', { config_id: STATE.configId });
        
        if (data.state) {
            STATE.isPlaying = true;
            
            const currentCanvasId = (STATE.phase === 2) ? 'gameCanvas_2' : 'gameCanvas';
            drawGame(data.state, currentCanvasId); 
            
            startTimer(CONFIG.ROUND_DURATION_SEC);
            updateGameUI();
        }
    } catch (err) {
        console.error("Round Start Error:", err);
        alert("Failed to start round. Please refresh.");
    }
}

// C. TIMER LOGIC
function startTimer(duration) {
    timeLeft = duration;
    updateTimerDisplay();

    gameTimer = setInterval(() => {
        timeLeft--;
        updateTimerDisplay();

        if (timeLeft <= 0) {
            clearInterval(gameTimer);
            finishTimeBasedRound();
        }
    }, 1000);
}

function updateTimerDisplay() {
    let timerId = (STATE.phase === 2) ? 'stepsLeft_2' : 'stepsLeft';
    
    const el = document.getElementById(timerId); 
    
    if (el) {
        const m = Math.floor(timeLeft / 60);
        const s = timeLeft % 60;
        el.innerText = `${m}:${s < 10 ? '0' : ''}${s}`;
        
        el.style.color = timeLeft <= 10 ? '#dc2626' : '#2563eb';
    } else {
        console.warn(`Timer element '${timerId}' not found!`);
    }
}

// D. UI UPDATER
function updateGameUI() {
    let suffix = (STATE.phase === 2) ? '_2' : '';
    
    const phaseEl = document.getElementById(`currentPhase${suffix}`);
    const roundEl = document.getElementById(`currentRound${suffix}`);
    
    if(phaseEl) phaseEl.innerText = STATE.phase;
    if(roundEl) roundEl.innerText = `${STATE.round} / ${CONFIG.ROUNDS_PER_PHASE}`;
}

// E. END ROUND
async function finishTimeBasedRound() {
    if (STATE.gameOver) return;
    
    STATE.isPlaying = false;
    STATE.gameOver = true;
    console.log("TIME IS UP!");

    DataManager.endRound();
    STATE.totalRounds++; 
    
    // 2. Final Score
    let scoreId = (STATE.phase === 2) ? 'currentScore_2' : 'currentScore';
    const scoreEl = document.getElementById(scoreId);
    const finalScore = scoreEl ? parseInt(scoreEl.innerText) : 0;
    
    // FIX: Determine Overlay IDs dynamically
    let suffix = (STATE.phase === 2) ? '_2' : '';
    const overlayId = `round-overlay${suffix}`;
    const titleId   = `overlay-title${suffix}`;
    const subId     = `overlay-subtitle${suffix}`;

    if (STATE.round < CONFIG.ROUNDS_PER_PHASE) {
        // --- CASE A: NEXT ROUND ---
        const overlay = document.getElementById(overlayId);
        const title   = document.getElementById(titleId);
        const sub     = document.getElementById(subId);
        
        if(overlay) {
            if(title) {
                title.innerText = `ROUND ${STATE.round} COMPLETE`;
                title.style.color = "#16a34a";
            }
            if(sub) sub.innerText = `Score: ${finalScore} | Next round in 3...`;
            
            overlay.classList.remove('hidden');
            overlay.style.opacity = '0';
            setTimeout(() => overlay.style.opacity = '1', 50); 
        }

        let countdown = 3;
        const interval = setInterval(() => {
            countdown--;
            if(sub) sub.innerText = `Score: ${finalScore} | Next round in ${countdown}...`;
        }, 1000);

        setTimeout(() => {
            clearInterval(interval);
            STATE.round++;
            
            if(title) {
                title.innerText = `ROUND ${STATE.round}`;
                title.style.color = "#2563eb"; 
            }
            if(sub) sub.innerText = "GO!";
            
            startRound().then(() => {
                setTimeout(() => {
                    if(overlay) overlay.style.opacity = '0';
                    setTimeout(() => {
                        if(overlay) overlay.classList.add('hidden');
                    }, 500); 
                }, 500); 
            });
            
        }, 3000);
        
    } else {
        // --- CASE B: PHASE COMPLETE ---
        console.log(`Phase ${STATE.phase} Complete!`);
        document.getElementById(overlayId)?.classList.add('hidden');

        // FIX: Ensure Summary handles the current phase
        renderPhaseSummary(); 

        if(STATE.phase === 1) {
            showPage('page-phase-1-qs');
        } else {
            // End of Experiment
            showPage('page-phase-2-qs'); 
        }
    }
}

// F. SUMMARY TABLE
function renderPhaseSummary() {
    let tableId = (STATE.phase === 2) ? 'summary-table-body-2' : 'summary-table-body';
    const tbody = document.getElementById(tableId);
    if(!tbody) return;

    tbody.innerHTML = ''; 

    const rounds = DataManager.LOGS.rounds.filter(r => r.phase === STATE.phase);

    rounds.forEach((round, index) => {
        const realDishes = round.dishesServed || 0;
        
        const displayRoundNum = index + 1;

        const row = `
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 12px; font-weight: bold; color: #4b5563;">
                    ${displayRoundNum}
                </td>
                <td style="padding: 12px; font-weight: bold; color: #16a34a;">
                    ${round.finalScore}
                </td>
                <td style="padding: 12px;">
                    ${realDishes}
                </td>
                <td style="padding: 12px; color: #2563eb;">
                    ${round.humanSteps || 0}
                </td>
            </tr>
        `;
        tbody.innerHTML += row;
    });
}

// --- 5. KEYBOARD LISTENER ---
document.addEventListener('keydown', async (e) => {
    if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].indexOf(e.key) === -1) return;
    if(!STATE.isPlaying || STATE.gameOver) return;
    
    e.preventDefault();

    // A. PRACTICE LOGIC
    if(STATE.phase === 0) {
        const data = await api('/key_event', { key: e.key, config_id: 'layout_practice' });
        drawGame(data.state, 'gameCanvas_practice');
        
        const prevScore = STATE.practiceScore;
        STATE.practiceScore = data.cumulative_reward || 0;

        if(STATE.practiceScore >= CONFIG.PRACTICE_SCORE && STATE.practiceScore > prevScore) {
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
        
        const currentCanvasId = (STATE.phase === 2) ? 'gameCanvas_2' : 'gameCanvas';
        drawGame(data.state, currentCanvasId); 
        
        // Update UI
        let scoreId = (STATE.phase === 2) ? 'currentScore_2' : 'currentScore';
        const scoreEl = document.getElementById(scoreId);
        if(scoreEl) scoreEl.innerText = Math.floor(data.cumulative_reward || 0);
        
        DataManager.logStep(data, e.key);
    }
});

// --- 6. INTRO PAGE VALIDATION ---
const inputID = document.getElementById('prolificId');
const inputAge = document.getElementById('age');
const inputGender = document.getElementById('gender');
const btnConsent = document.getElementById('to-consent');

function validateIntro() {
    if(!inputID || !inputAge || !inputGender) return;
    btnConsent.disabled = !(
        inputID.value.trim().length > 0 && 
        parseInt(inputAge.value) >= 18 && 
        inputGender.value !== ""
    );
}

if(inputID) {
    [inputID, inputAge, inputGender].forEach(el => el.addEventListener('input', validateIntro));
    inputGender.addEventListener('change', validateIntro);
    
    btnConsent.onclick = () => {
        STATE.prolificId = inputID.value.trim();
        const age = parseInt(inputAge.value);
        const gender = inputGender.value;
        
        assignConditions(); 

        DataManager.initUser(STATE.prolificId, age, gender, STATE.assignment);

        showPage('page-consent');
    };
}

// --- 7. CONSENT PAGE ---
const consentCheck = document.getElementById('consentCheck');
const btnInstruction = document.getElementById('to-instruction');

if(consentCheck && btnInstruction) {
    consentCheck.addEventListener('change', () => {
        btnInstruction.disabled = !consentCheck.checked;
    });

    btnInstruction.onclick = () => {
        showPage('page-instruction-1');
        // Start practice round when page loads
        setTimeout(() => {
            if(typeof startPracticeRound === 'function') {
                startPracticeRound();
            }
        }, 100);
    };
}

const btnToInst2 = document.getElementById('to-instruction-2');
if(btnToInst2) {
    btnToInst2.onclick = () => {
        // Go to the Quiz/Instruction 2a
        showPage('page-instruction-2a');
    };
}

// --- 8. SILENT QUIZ & EXCLUSION LOGIC ---

let QUIZ_ERRORS = 0;

function calculatePageErrors(questionNames) {
    let pageErrors = 0;
    
    questionNames.forEach(name => {
        const selected = document.querySelector(`input[name="${name}"]:checked`);
        
        if (!selected || selected.value !== 'correct') {
            pageErrors++;
            console.log(`Mistake on question: ${name}`);
        }
    });
    
    return pageErrors;
}

// --- BUTTON LISTENERS ---

// 1. Page 2a (Goal & Attention Check) -> Move to 2b
const btnNext2a = document.getElementById('btn-next-2a');
if (btnNext2a) {
    btnNext2a.onclick = () => {
        const errors = calculatePageErrors(['q1', 'q1_att']);
        QUIZ_ERRORS += errors;
        
        console.log(`Page 2a Errors: ${errors} | Current Total: ${QUIZ_ERRORS}`);
        //showPage('page-instruction-2b');
        showPage('page-instruction-2b');
    };
}

// 2. Page 2b (Structure & Observation) -> Move to 2c
const btnNext2b = document.getElementById('btn-next-2b');
if (btnNext2b) {
    btnNext2b.onclick = () => {
        const errors = calculatePageErrors(['q2a', 'q2b']);
        QUIZ_ERRORS += errors;
        
        console.log(`Page 2b Errors: ${errors} | Current Total: ${QUIZ_ERRORS}`);
        showPage('page-instruction-2c');
    };
}

// 3. Page 2c (Strategy) -> START GAME (Final Filter)
const btnStartTask = document.getElementById('start-task-1');
if (btnStartTask) {
    btnStartTask.onclick = () => {
        const errors = calculatePageErrors(['q3a', 'q3b']);
        QUIZ_ERRORS += errors;

        console.log(`Final Check. Total Cumulative Errors: ${QUIZ_ERRORS}`);

        if (QUIZ_ERRORS > 2) {
            alert("Qualification Failed.\n\nYou answered too many comprehension questions incorrectly.");
            location.reload(); 
        } else {
            console.log("Quiz Passed. Starting Phase 1...");
            
            // 1. CRITICAL: Ensure Assignment Exists
            if (!STATE.assignment || !STATE.assignment.layout) {
                assignConditions();
            }

            // 2. Start Phase 1 (This handles Config ID and Showing Page)
            startPhase(1); 
        }
    };
}

// --- 9. PHASE TRANSITIONS ---

// A. Phase 1 Mid-Survey -> Start Phase 2
document.getElementById('btn-submit-mid-survey')?.addEventListener('click', () => {
    // 1. Validation
    const qLoad = document.querySelector('input[name="mid_load"]:checked');
    const qCollab = document.querySelector('input[name="mid_collab"]:checked');

    if (!qLoad || !qCollab) {
        alert("Please answer the required questions (Load & Cooperation).");
        return;
    }

    // 2. Save Data to dataManager
    const answers = {
        cognitiveLoad: parseInt(qLoad.value),
        collaboration: parseInt(qCollab.value),
        strategy: document.querySelector('input[name="mid_strategy"]:checked')?.value,
        predictability: document.querySelector('input[name="mid_predict"]:checked')?.value
    };
    
    DataManager.saveQuestionnaire(1, answers);

    // 3. Start Phase 2
    startPhase(2);
});

// B. Submit Data
document.getElementById('btn-submit-final-survey')?.addEventListener('click', async () => {
    // 1. Validation
    const qLoad = document.querySelector('input[name="post_load"]:checked');
    const qCollab = document.querySelector('input[name="post_collab"]:checked');

    if (!qLoad || !qCollab) {
        alert("Please answer the required questions.");
        return;
    }

    // 2. Save Data
    const answers = {
        cognitiveLoad: parseInt(qLoad.value),
        collaboration: parseInt(qCollab.value),
        strategy: document.querySelector('input[name="post_strategy"]:checked')?.value,
        predictability: document.querySelector('input[name="post_predict"]:checked')?.value
    };

    DataManager.saveQuestionnaire(2, answers);
    DataManager.saveFinalFeedback(document.getElementById('final-feedback')?.value || "");

    // 3. Submit to Backend
    await submitData();
});

// --- 10. FINAL SUBMISSION ---
async function submitData() {
    try {
        const response = await DataManager.submitToServer();
        
        if(response.success) {
            showPage('page-end');
            document.getElementById('completionCodeWrap')?.classList.remove('hidden');
            document.getElementById('completionCode').innerText = response.completion_code;
        } else {
            alert("Submission failed. Please contact the researcher.");
        }
    } catch(err) {
        console.error("Submission error:", err);
        alert("Network error during submission. Please try again.");
    }
}

// --- 11. INITIALIZE ---
window.onload = () => {
    preloadImages(() => {
        console.log("Images loaded and game is ready.");
    });
};