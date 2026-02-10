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
    const pid = LOGS.prolificId || "test";
    let h = 0;
    for(let i=0; i<pid.length; i++) h = (h*31 + pid.charCodeAt(i)) >>> 0;
    
    // Randomly assign layout and models
    const layouts = ["layout1", "layout2", "layout3", "layout4"];
    const models = ["model1", "model2", "model3", "model4"];
    
    STATE.assignment.layout = layouts[h % 4];
    STATE.assignment.phase1Model = models[h % 4];
    STATE.assignment.phase2Model = models[(h + 1) % 4]; // Different model for phase 2
    
    LOGS.assignment = STATE.assignment;
    
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

let gameTimer = null; // Stores the interval ID
let timeLeft = 0;     // Stores seconds remaining

// A. START PHASE (Setup Model & Layout)
async function startPhase(phaseNum) {
    STATE.phase = phaseNum;
    STATE.round = 1;
    STATE.gameOver = false;
    
    // Determine which model to use
    let modelId = phaseNum === 1 ? STATE.assignment.phase1Model : STATE.assignment.phase2Model;
    STATE.configId = `${STATE.assignment.layout}_${modelId}`;
    
    console.log(`Starting Phase ${phaseNum} with ${STATE.configId}`);
    await startRound();
}

// B. START ROUND (Reset Backend & Start Timer)
async function startRound() {
    STATE.isPlaying = false;
    STATE.gameOver = false;
    
    // 1. Clear any old timer
    if (gameTimer) clearInterval(gameTimer);

    // 2. Initialize Round Data
    currentRoundData = {
        roundNumber: STATE.totalRounds + 1,
        phase: STATE.phase,
        configId: STATE.configId,
        steps: [],
        finalScore: 0,
        humanSteps: 0,
        startTime: Date.now()
    };
    
    try {
        // 3. Reset Environment
        const data = await api('/reset', { config_id: STATE.configId });
        
        if (data.state) {
            STATE.isPlaying = true;
            drawGame(data.state, 'gameCanvas');
            
            // 4. Start the 45-Second Timer
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
    updateTimerDisplay(); // Update immediately

    gameTimer = setInterval(() => {
        timeLeft--;
        updateTimerDisplay();

        if (timeLeft <= 0) {
            // TIME IS UP!
            clearInterval(gameTimer);
            finishTimeBasedRound();
        }
    }, 1000);
}

function updateTimerDisplay() {
    const el = document.getElementById('stepsLeft'); // Keeping ID 'stepsLeft' to avoid HTML changes
    if (el) {
        const m = Math.floor(timeLeft / 60);
        const s = timeLeft % 60;
        el.innerText = `${m}:${s < 10 ? '0' : ''}${s}`;
        
        // Red warning color at 10 seconds
        el.style.color = timeLeft <= 10 ? '#dc2626' : '#2563eb';
    }
}

// D. UI UPDATER
function updateGameUI() {
    const phaseEl = document.getElementById('currentPhase');
    const roundEl = document.getElementById('currentRound');
    
    if(phaseEl) phaseEl.innerText = STATE.phase;
    if(roundEl) roundEl.innerText = `${STATE.round} / ${CONFIG.ROUNDS_PER_PHASE}`;
}

// E. END ROUND (Triggered by Timer)
async function finishTimeBasedRound() {
    if (STATE.gameOver) return;
    
    STATE.isPlaying = false;
    STATE.gameOver = true;
    console.log("TIME IS UP!");

    // 1. Finalize Data
    const scoreEl = document.getElementById('currentScore');
    const finalScore = scoreEl ? parseInt(scoreEl.innerText) : 0;
    
    currentRoundData.finalScore = finalScore;
    currentRoundData.endTime = Date.now();
    currentRoundData.duration = CONFIG.ROUND_DURATION_SEC;
    
    // 2. Save Log
    LOGS.rounds.push({...currentRoundData});
    STATE.totalRounds++; 
    
    // Rounds repetition
    if (STATE.round < CONFIG.ROUNDS_PER_PHASE) {
        
        // animation overlay
        const overlay = document.getElementById('round-overlay');
        const title = document.getElementById('overlay-title');
        const sub = document.getElementById('overlay-subtitle');
        
        if(overlay) {
            // Update Text for "Round Complete"
            if(title) {
                title.innerText = `ROUND ${STATE.round} COMPLETE`;
                title.style.color = "#16a34a"; // Green
            }
            if(sub) sub.innerText = `Score: ${finalScore} | Next round in 3...`;
            
            // Fade In
            overlay.classList.remove('hidden');
            overlay.style.opacity = '0';
            setTimeout(() => overlay.style.opacity = '1', 50); 
        }

        // countdown
        let countdown = 3;
        const interval = setInterval(() => {
            countdown--;
            if(sub) sub.innerText = `Score: ${finalScore} | Next round in ${countdown}...`;
        }, 1000);

        // AUTO-START NEXT ROUND
        setTimeout(() => {
            clearInterval(interval);
            
            // Increment Round
            STATE.round++;
            
            // Update Overlay for "Start"
            if(title) {
                title.innerText = `ROUND ${STATE.round}`;
                title.style.color = "#2563eb"; // Blue
            }
            if(sub) sub.innerText = "GO!";
            
            // Start the actual game logic
            startRound().then(() => {
                // Fade Out Overlay when game is ready
                setTimeout(() => {
                    if(overlay) overlay.style.opacity = '0';
                    setTimeout(() => {
                        if(overlay) overlay.classList.add('hidden');
                    }, 500); 
                }, 500); 
            });
            
        }, 3000); // 3 seconds total pause
        
    } else {
        // CASE B: PHASE COMPLETE
        console.log(`Phase ${STATE.phase} Complete!`);
        
        // Hide overlay if visible
        document.getElementById('round-overlay')?.classList.add('hidden');

        if(STATE.phase === 1) {
            // 1. SUMMARY TABLE
            if(typeof renderPhaseSummary === 'function') {
                renderPhaseSummary();
            }
            // 2. QUESTIONNAIRE PAGE
            showPage('page-mid-questionnaire');
        } else {
            // End of Experiment
            finishExperiment(); 
        }
    }
}


// F. SUMMARY TABLE
function renderPhaseSummary() {
    const tbody = document.getElementById('summary-table-body');
    if(!tbody) return;

    tbody.innerHTML = ''; 

    const phase1Rounds = LOGS.rounds.filter(r => r.phase === 1);

    phase1Rounds.forEach(round => {
        const realDishes = round.dishesServed || 0;

        const row = `
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 12px; font-weight: bold; color: #4b5563;">
                    ${round.phaseRound || round.roundNumber}
                </td>
                <td style="padding: 12px; font-weight: bold; color: #16a34a;">
                    ${round.finalScore}
                </td>
                <td style="padding: 12px;">
                    ${realDishes}
                </td>
                <td style="padding: 12px; color: #2563eb;">
                    ${round.humanSteps}
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
        drawGame(data.state, 'gameCanvas');
        
        // Update UI
        document.getElementById('currentScore').innerText = Math.floor(data.cumulative_reward || 0);
        
        // Log the step
        currentRoundData.steps.push({ 
            key: e.key, 
            reward: data.cumulative_reward,
            timestamp: Date.now()
        });
        
        // Count human steps (not "Stay")
        if(e.key !== 'Stay') {
            currentRoundData.humanSteps++;
        }
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
        LOGS.prolificId = inputID.value.trim();
        LOGS.age = parseInt(inputAge.value);
        LOGS.gender = inputGender.value;
        assignConditions();
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
        
        // Logic: 
        // 1. If nothing selected -> Error
        // 2. If selected value is NOT 'correct' -> Error
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

        // --- EXCLUSION LOGIC ---
        if (QUIZ_ERRORS > 2) {
            
            // OPTION A: Show an alert and reload (Simple)
            alert("Qualification Failed.\n\nYou answered too many comprehension questions incorrectly.\nTo ensure high-quality data, you cannot proceed with this experiment.");
            location.reload(); 
            
            // OPTION B: Send them to the specific 'page-disqualified' div if you have one
            // showPage('page-disqualified');

        } else {
            // PASS: Start the Experiment
            console.log("Quiz Passed. Starting Phase 1...");
            
            // 1. Show Game Canvas
            showPage('page-phase-1');

            // 2. Initialize Game State
            STATE.phase = 1; 
            STATE.round = 1;
            STATE.totalRounds = 0;
            
            // 3. Assign Experimental Conditions (Layout/Model)
            assignConditions();
            console.log("Conditions Assigned:", STATE.assignment);

            // 4. Trigger Backend to Load Round 1
            startRound(); 
        }
    };
}

// --- 9. PHASE TRANSITIONS ---

// After Phase 1 Questionnaire
document.getElementById('btn-phase1-qs-submit')?.addEventListener('click', () => {
    // Collect Phase 1 questionnaire data
    LOGS.phase1Questionnaire = {
        cognitiveLoad: parseInt(document.querySelector('input[name="p1_cognitive"]:checked')?.value || 0),
        collaboration: parseInt(document.querySelector('input[name="p1_collab"]:checked')?.value || 0)
    };
    
    if(LOGS.phase1Questionnaire.cognitiveLoad === 0 || LOGS.phase1Questionnaire.collaboration === 0) {
        alert("Please answer both questions.");
        return;
    }
    
    showPage('page-intermission');
});

// Start Phase 2
document.getElementById('btnStartPhase2')?.addEventListener('click', () => {
    showPage('page-phase-2');
    startPhase(2);
});

// After Phase 2 Questionnaire
document.getElementById('btn-phase2-qs-submit')?.addEventListener('click', () => {
    // Collect Phase 2 questionnaire data
    LOGS.phase2Questionnaire = {
        cognitiveLoad: parseInt(document.querySelector('input[name="p2_cognitive"]:checked')?.value || 0),
        collaboration: parseInt(document.querySelector('input[name="p2_collab"]:checked')?.value || 0)
    };
    
    const feedback = document.getElementById('final-feedback')?.value.trim() || "";
    LOGS.finalFeedback = feedback;
    
    if(LOGS.phase2Questionnaire.cognitiveLoad === 0 || LOGS.phase2Questionnaire.collaboration === 0) {
        alert("Please answer both rating questions.");
        return;
    }
    
    // Submit all data to backend
    submitData();
});

// --- 10. FINAL SUBMISSION ---
async function submitData() {
    try {
        const response = await api('/submit_log', { log: LOGS });
        
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