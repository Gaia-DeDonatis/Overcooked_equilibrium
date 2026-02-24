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
    
    // Randomly assign layout (kept for logging / future; backend currently uses a fixed map)
    const layouts = ["cramped", "circuit", "asymmetric", "ring", "forced"];
    STATE.assignment.layout = layouts[h % layouts.length];
    
    console.log("Assigned:", STATE.assignment);
}

// --- 3. PAGE NAVIGATION ---
function showPage(pageId) {
    const pages = [
        'page-intro', 'page-consent', 'page-instruction-1',
        'page-instruction-2a','page-instruction-2b','page-instruction-2c',
        'page-phase-1', 'page-episode-break',
        'page-end'
    ];
    
    pages.forEach(id => {
        const el = document.getElementById(id);
        if(el) el.classList.add('hidden');
    });

    const target = document.getElementById(pageId);
    if(target) target.classList.remove('hidden');
    window.scrollTo(0,0);

    // Only set isPlaying for game pages
    const gamePages = ['page-phase-1', 'page-instruction-1'];
    if (!gamePages.includes(pageId)) {
        STATE.isPlaying = false;
    }
}

// --- 4. GAME INITIALIZATION (TIME-BASED) ---

function getEpisodePhase(episodeIndex) {
  if (episodeIndex <= CONFIG.EPISODES_SEED) return 'seed';
  if (episodeIndex <= CONFIG.EPISODES_SEED + CONFIG.EPISODES_BO) return 'bo';
  return 'stress';
}

let gameTimer = null;
let timeLeft = 0;
let aiTickTimer = null;
let aiTickInFlight = false;
const AI_TICK_MS = 250;

let bufferedHumanKey = 'Stay';
let roundStartPerfMs = 0;
let lastHumanPressMs = null;  

function stopAiTick() {
  if (aiTickTimer) {
    clearInterval(aiTickTimer);
    aiTickTimer = null;
  }
  aiTickInFlight = false;
}

async function doOneTick() {
  if (!STATE.isPlaying || STATE.gameOver) return;
  if (aiTickInFlight) return; // prevent request pile-up
  aiTickInFlight = true;

  try {
    const keyToSend = bufferedHumanKey || 'Stay';
    bufferedHumanKey = 'Stay';

    const data = await api('/key_event', { key: keyToSend, config_id: STATE.configId });

    drawGame(data.state, 'gameCanvas');

    const appliedMs = performance.now() - roundStartPerfMs;
    DataManager.logStep(data, keyToSend, { appliedMs, humanPressMs: lastHumanPressMs });
    lastHumanPressMs = null;

    const roundObj = DataManager.getCurrentRound();
    if (roundObj && roundObj.summary) {
      // Backend dishes_served is per-round; store it so we can sum across rounds in the episode.
      if (data.dishes_served != null) roundObj.summary.dishes_served = data.dishes_served;
    }

    STATE.lastScore = data.cumulative_reward ?? STATE.lastScore ?? 0;

    const roundNow = DataManager.getCurrentRound();

    const dishesNow = roundNow?.summary?.dishes_served ?? 0;
    const stepsNow  = roundNow?.summary?.human_steps ?? 0;

    const dishesEl = document.getElementById('dishesServed');
    if (dishesEl) dishesEl.innerText = String(dishesNow);

    const stepsEl = document.getElementById('humanSteps');
    if (stepsEl) stepsEl.innerText = String(stepsNow);
} catch (err) {
    console.error("Tick error:", err);
  } finally {
    aiTickInFlight = false;
  }
}



function startAiTick() {
  stopAiTick();
  aiTickTimer = setInterval(doOneTick, AI_TICK_MS);
}




// A. START EPISODE (3 rounds each)
async function startEpisode(episodeIndex) {
  if (!STATE.assignment || !STATE.assignment.layout) {
    assignConditions();
  }

  STATE.phase = 1; // main task
  STATE.episodeIndex = episodeIndex;
  STATE.roundInEpisode = 1;
  STATE.episodePhase = getEpisodePhase(episodeIndex);
  STATE.gameOver = false;
  STATE.configId = `${STATE.assignment.layout}_experiment`;

  showPage('page-phase-1');
  updateGameUI();
  await startRound({ newEpisode: true });
}

// B. START ROUND
async function startRound({ newEpisode = false } = {}) {
  STATE.isPlaying = false;
  STATE.gameOver = false;

  if (gameTimer) clearInterval(gameTimer);

  try {
    const data = await api('/reset', {
      config_id: STATE.configId,
      episode_index: STATE.episodeIndex,
      round_in_episode: STATE.roundInEpisode,
      episode_phase: STATE.episodePhase,
      new_episode: !!newEpisode
    });
    bufferedHumanKey = 'Stay';
    aiTickInFlight = false;

    if (DataManager.LOGS.meta.tick_ms == null) {
      DataManager.LOGS.meta.tick_ms = AI_TICK_MS;
    }

    // Keep a first-seen policy id for convenience
    if (DataManager.LOGS.meta.policy_id_phase1 == null && data.model_id != null) {
      DataManager.LOGS.meta.policy_id_phase1 = data.model_id;
    }

    // START NEW ROUND
    DataManager.startNewRound(STATE.phase, STATE.configId, {
      mapTopology: `${data.map_type}_${data.grid_dim}`,
      policyId: data.model_id,
      chosenCkpt: data.chosen_ckpt
      ,
      episode_index: STATE.episodeIndex,
      round_in_episode: STATE.roundInEpisode,
      episode_phase: STATE.episodePhase
    });

    if (data.state) {
      STATE.isPlaying = true;
    
      roundStartPerfMs = performance.now();
      lastHumanPressMs = null;

      drawGame(data.state, 'gameCanvas');
      startAiTick();
      startTimer(CONFIG.ROUND_DURATION_SEC);
      updateGameUI();

      const roundNow = DataManager.getCurrentRound();

      const dishesEl = document.getElementById('dishesServed');
      if (dishesEl) dishesEl.innerText = String(roundNow?.summary?.dishes_served ?? 0);

      const stepsEl = document.getElementById('humanSteps');
      if (stepsEl) stepsEl.innerText = String(roundNow?.summary?.human_steps ?? 0);
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
    const el = document.getElementById('timeRemaining');
    
    if (el) {
        const m = Math.floor(timeLeft / 60);
        const s = timeLeft % 60;
        el.innerText = `${m}:${s < 10 ? '0' : ''}${s}`;
        
        el.style.color = timeLeft <= 10 ? '#dc2626' : '#2563eb';
    } else {
        console.warn("Timer element 'timeRemaining' not found!");
    }
}

// D. UI UPDATER
function updateGameUI() {
    const epEl = document.getElementById('currentEpisode');
    if (epEl) epEl.innerText = `${STATE.episodeIndex} / ${CONFIG.TOTAL_EPISODES}`;
}

// E. END ROUND
async function finishTimeBasedRound() {
    if (STATE.gameOver) return;
    
    STATE.isPlaying = false;
    STATE.gameOver = true;
    console.log("TIME IS UP!");

    stopAiTick();
    bufferedHumanKey = 'Stay';


    DataManager.endRound();
    
    // 2. Final Score
    const r = DataManager.getCurrentRound();
    const finalScore = Math.floor(r?.summary?.final_score ?? 0);
    
    const overlay = document.getElementById('round-overlay');
    const title   = document.getElementById('overlay-title');
    const sub     = document.getElementById('overlay-subtitle');

    if (STATE.roundInEpisode < CONFIG.ROUNDS_PER_EPISODE) {
        // --- CASE A: NEXT ROUND (within the same episode) ---
        
        if (overlay) {
          if (title) {
            title.innerText = `ROUND ${STATE.roundInEpisode} COMPLETE`;
            title.style.color = "#16a34a";
          }
          if (sub) sub.innerText = `Score: ${finalScore} | Next round in 3...`;

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
            STATE.roundInEpisode++;

            if (title) {
              title.innerText = `ROUND ${STATE.roundInEpisode}`;
              title.style.color = "#2563eb";
            }
            if (sub) sub.innerText = "GO!";

            startRound({ newEpisode: false }).then(() => {
              setTimeout(() => {
                if (overlay) overlay.style.opacity = '0';
                setTimeout(() => {
                  if (overlay) overlay.classList.add('hidden');
                }, 500);
              }, 500);
            });
            
        }, 3000);
        
    } else {
        // --- CASE B: EPISODE COMPLETE ---
        console.log(`Episode ${STATE.episodeIndex} Complete!`);

        if (overlay) {
          if (title) {
            title.innerText = `EPISODE ${STATE.episodeIndex} COMPLETE`;
            title.style.color = "#16a34a";
          }
          if (sub) sub.innerText = "Short break...";
          overlay.classList.remove('hidden');
          overlay.style.opacity = '1';
        }

        // Move to the break page immediately (no phase summary, no big questionnaire)
        setTimeout(() => {
          if (overlay) overlay.classList.add('hidden');
          showEpisodeBreak();
        }, 300);
    }
}


// --- EPISODE BREAK (fixed 30s, auto-advance) ---
let breakTimer = null;
let breakTimeLeft = 0;
let breakFinishedFired = false;

function updateBreakCountdown() {
  const el = document.getElementById('breakCountdown');
  if (!el) return;
  const m = Math.floor(breakTimeLeft / 60);
  const s = breakTimeLeft % 60;
  el.innerText = `${m}:${s < 10 ? '0' : ''}${s}`;
}

async function finishEpisodeBreak() {
  if (breakFinishedFired) return;
  breakFinishedFired = true;

  const qEffort = document.querySelector('input[name="ep_effort"]:checked');
  const qCoord = document.querySelector('input[name="ep_coord"]:checked');

  DataManager.saveEpisodeSurvey(
    STATE.episodeIndex,
    STATE.episodePhase,
    {
      mental_effort: qEffort ? parseInt(qEffort.value) : null,
      coordination_quality: qCoord ? parseInt(qCoord.value) : null
    }
  );

  if (breakTimer) {
    clearInterval(breakTimer);
    breakTimer = null;
  }

  if (STATE.episodeIndex < CONFIG.TOTAL_EPISODES) {
    await startEpisode(STATE.episodeIndex + 1);
  } else {
    await submitData();
  }
}

function showEpisodeBreak() {
  STATE.isPlaying = false;
  STATE.gameOver = true;

  showPage('page-episode-break');

  // Reset form
  document.querySelectorAll('input[name="ep_effort"], input[name="ep_coord"]').forEach(el => { el.checked = false; });

  const epLabel = document.getElementById('breakEpisodeLabel');
  if (epLabel) {
    const next = (STATE.episodeIndex < CONFIG.TOTAL_EPISODES) ? (STATE.episodeIndex + 1) : null;
    epLabel.innerText = next
      ? `Episode ${STATE.episodeIndex} complete. Next: Episode ${next} will start automatically.`
      : `Episode ${STATE.episodeIndex} complete. The study will finish automatically.`;
  }

  // Start fixed 30s timer (always full duration, auto-advance)
  breakFinishedFired = false;
  if (breakTimer) clearInterval(breakTimer);

  breakTimeLeft = CONFIG.EPISODE_BREAK_SEC;
  updateBreakCountdown();

  breakTimer = setInterval(() => {
    breakTimeLeft -= 1;
    if (breakTimeLeft <= 0) {
      breakTimeLeft = 0;
      updateBreakCountdown();
      clearInterval(breakTimer);
      breakTimer = null;
      finishEpisodeBreak().catch(console.error);
      return;
    }
    updateBreakCountdown();
  }, 1000);
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
            //alert("Practice Complete! You delivered the salad.");
        }
    } 
    // B. MAIN TASK LOGIC (Phase 1 or 2)
    else {
        bufferedHumanKey = e.key;
        lastHumanPressMs = performance.now() - roundStartPerfMs;
        doOneTick();
    }
});

// --- 6. INTRO PAGE VALIDATION ---
const inputID = document.getElementById('prolificId');
const inputAge = document.getElementById('age');
const inputGender = document.getElementById('gender');
const inputExp = document.getElementById('experience');
const btnConsent = document.getElementById('to-consent');

function validateIntro() {
    if(!inputID || !inputAge || !inputGender || !inputExp) return;
    btnConsent.disabled = !(
        inputID.value.trim().length > 0 && 
        parseInt(inputAge.value) >= 18 && 
        inputGender.value !== "" &&
        inputExp.value !== ""
    );
}

if(inputID) {
    [inputID, inputAge, inputGender, inputExp].forEach(el => el.addEventListener('input', validateIntro));
    inputGender.addEventListener('change', validateIntro);
    
    btnConsent.onclick = () => {
        STATE.prolificId = inputID.value.trim();
        const age = parseInt(inputAge.value);
        const gender = inputGender.value;
        const experience = inputExp.value;
        
        assignConditions(); 

        DataManager.initUser(STATE.prolificId, age, gender, STATE.assignment, {experience: experience});

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

// 3. Page 2c -> SUBMIT QUIZ & FILTER
const btnSubmitQuiz = document.getElementById('btn-submit-quiz');
if (btnSubmitQuiz) {
    btnSubmitQuiz.onclick = () => {
        const errors = calculatePageErrors(['q3a', 'q3b']);
        QUIZ_ERRORS += errors;

        console.log(`Final Check. Total Cumulative Errors: ${QUIZ_ERRORS}`);

        if (QUIZ_ERRORS > 2) {
            // Disqualify
            STATE.isPlaying = false;
            STATE.gameOver = true;
            stopAiTick();
            if (gameTimer) clearInterval(gameTimer);
            alert("Qualification Failed.\n\nYou answered too many comprehension questions incorrectly.");
            window.location.href = "https://app.prolific.com/submissions/complete?cc=CGDMBD6O";
            return; 
        } else {
            // Pass! Reveal the "All Set" box and Start button
            console.log("Quiz Passed. Revealing start button.");
            document.getElementById('submit-quiz-container').classList.add('hidden');
            document.getElementById('all-set-container').classList.remove('hidden');
        }
    };
}

// 4. START GAME
const btnStartTask = document.getElementById('start-task-1');
if (btnStartTask) {
    btnStartTask.onclick = () => {
        console.log("Starting Episode 1...");
        startEpisode(1);
    };
}

// --- 10. FINAL SUBMISSION ---
async function submitData() {
    try {
        const response = await DataManager.submitToServer();
        
        if(response.success) {
            showPage('page-end');
           
            const code = response.completion_code || "CK4KW637";
            const prolificUrl = `https://app.prolific.com/submissions/complete?cc=${encodeURIComponent(code)}`;

            // Show the completion code
            document.getElementById('completionCodeWrap')?.classList.remove('hidden');

            const codeEl = document.getElementById('completionCode');
            if (codeEl) codeEl.innerText = code;

            // "copy/paste" fallback text element
            const codeInlineEl = document.getElementById('completionCodeInline');
            if (codeInlineEl) codeInlineEl.innerText = code;

            // "Return to Prolific" link
            const linkEl = document.getElementById('prolificReturnLink');
            if (linkEl) linkEl.href = prolificUrl;
            
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