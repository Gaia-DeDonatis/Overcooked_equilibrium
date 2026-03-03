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
        'page-game', 'page-episode-break',
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
    const gamePages = ['page-game', 'page-instruction-1'];
    if (!gamePages.includes(pageId)) {
        STATE.isPlaying = false;
    }
}

// --- 4. GAME INITIALIZATION (TIME-BASED) ---
function getEpisodePhase(episodeIndex) {
  if (episodeIndex <= CONFIG.EPISODES_SEED) return 'seed';
  if (episodeIndex <= CONFIG.PHASE_BO_END) return 'bo';
  if (episodeIndex <= CONFIG.PHASE_STRESS_END) return 'stress';
  return 'replay_optimal';
}

function getExperimentPhase(episodeIndex) {
  // 1 = seed, 2 = BO, 3 = stress (kNN), 4 = replay optimal policy (new strategy)
  if (episodeIndex <= CONFIG.EPISODES_SEED) return 1;
  if (episodeIndex <= CONFIG.PHASE_BO_END) return 2;
  if (episodeIndex <= CONFIG.PHASE_STRESS_END) return 3;
  return 4;
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
  STATE.experimentPhase = getExperimentPhase(episodeIndex);
  STATE.gameOver = false;
  STATE.configId = 'experiment';

  showPage('page-game');
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
      prolificId: STATE.prolificId,
      n_init: CONFIG.EPISODES_SEED,
      n_bo:   CONFIG.EPISODES_BO,
      n_knn: CONFIG.EPISODES_STRESS,
      solo_episode: (typeof isSoloEpisode === 'function') ? isSoloEpisode(STATE.episodeIndex) : false,
      new_episode: !!newEpisode
    });
    bufferedHumanKey = 'Stay';
    aiTickInFlight = false;

    if (data.map_type) {
      STATE.assignment.layout = data.map_type;
      //STATE.configId = data.map_type + '_experiment';
    }

    if (DataManager.LOGS.meta.tick_ms == null) {
      DataManager.LOGS.meta.tick_ms = AI_TICK_MS;
    }

    // START NEW ROUND
    const solo = isSoloEpisode(STATE.episodeIndex);
    const policyIdForLog = solo ? "no_ai" : data.model_id;

    DataManager.startNewRound(STATE.phase, STATE.configId, {
      mapTopology: `${data.map_type}_${data.grid_dim}`,
      policyId: policyIdForLog,
      chosenCkpt: data.chosen_ckpt,
      episode_index: STATE.episodeIndex,
      round_in_episode: STATE.roundInEpisode,
      episode_phase: STATE.episodePhase,
      experiment_phase: STATE.experimentPhase,
      stressPolicyDistance: data.stress_policy_distance ?? null,
      optimalPolicyId: data.optimal_policy_id ?? null
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

    // Phase 4 banner (replay optimal AI policy, ask participant to change strategy)
    const phase4Banner = document.getElementById('phase4Prompt');
    const phase4Text   = document.getElementById('phase4PromptText');
    const subtitleEl   = document.getElementById('mainTaskSubtitle');

    const isPhase4 = (STATE.episodePhase === 'replay_optimal');
    if (phase4Banner) {
      if (isPhase4) phase4Banner.classList.remove('hidden');
      else phase4Banner.classList.add('hidden');
    }
    if (phase4Text) {
      phase4Text.innerText = "Could you find a new way to work with the AI? Try changing your strategy in this episode.";
    }
    if (subtitleEl) {
      subtitleEl.innerText = isPhase4
        ? "Final episode: you will play again with the AI teammate from earlier — but please try a different collaboration strategy."
        : "Play with the AI teammate and try to deliver as many dishes as possible.";
    }
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

    STATE.episodeScore += finalScore;
    
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

        const soloNow = (typeof isSoloEpisode === 'function') ? isSoloEpisode(STATE.episodeIndex) : false;
        const shouldTell = (!soloNow) && (STATE.episodePhase !== 'replay_optimal');
        if (shouldTell) {
          await api('/tell', {
            prolificId: STATE.prolificId,
            score: finalScore
          });
        }
    if (overlay) {
          if (title) {
            title.innerText = `EPISODE ${STATE.episodeIndex} COMPLETE`;
            title.style.color = "#16a34a";
          }
          if (sub) sub.innerText = "Short break...";
          overlay.classList.remove('hidden');
          overlay.style.opacity = '1';
        }

        // Move to the break page
        setTimeout(() => {
          if (overlay) overlay.classList.add('hidden');
          showEpisodeBreak();
        }, 300);
    }
}


// --- EPISODE BREAK (fixed timer + TLX sliders) ---
let breakTimer = null;
let breakTimeLeft = 0;
let breakFinishedFired = false;
let breakTimerFinished = false;

function updateBreakCountdown() {
  const el = document.getElementById('breakCountdown');
  if (!el) return;
  const m = Math.floor(breakTimeLeft / 60);
  const s = breakTimeLeft % 60;
  el.innerText = `${m}:${s < 10 ? '0' : ''}${s}`;
}

// --- TLX slider helpers ---
function _isTouched(el){ return !!el && (el.getAttribute('data-touched') === 'true'); }

function _setRangePct(el){
  if(!el) return;
  const min = parseInt(el.min || '1', 10);
  const max = parseInt(el.max || '20', 10);
  const v = parseInt(el.value || String(Math.round((min + max) / 2)), 10);
  const pct = ((v - min) * 100) / (max - min);
  el.style.setProperty('--pct', `${pct}%`);
}

function _setUntouched(el, chip, valueSpan){
  if(el){
    el.value = el.value || 10;
    el.setAttribute('data-touched', 'false');
    el.classList.add('untouched');
    _setRangePct(el);
  }
  if(chip) chip.classList.add('untouched');
  if(valueSpan) valueSpan.innerText = '—';
}

function _markTouched(el, chip, valueSpan){
  if(el){
    el.setAttribute('data-touched', 'true');
    el.classList.remove('untouched');
    _setRangePct(el);
  }
  if(chip) chip.classList.remove('untouched');
  if(valueSpan && el) valueSpan.innerText = el.value;
}

function updateBreakContinueState() {
  const btn = document.getElementById('breakContinueBtn');
  const hint = document.getElementById('breakContinueHint');

  // Prefer TLX sliders if present; otherwise fall back to radio buttons.
  const md = document.getElementById('ep_mental_demand');
  const pf = document.getElementById('ep_performance');

  let answered = false;
  if (md && pf) {
    answered = _isTouched(md) && _isTouched(pf);
  } else {
    const qEffort = document.querySelector('input[name="ep_effort"]:checked');
    const qCoord  = document.querySelector('input[name="ep_coord"]:checked');
    answered = !!qEffort && !!qCoord;
  }

  const canContinue = breakTimerFinished && answered;

  if (btn) btn.disabled = !canContinue;

  if (hint) {
    if (!breakTimerFinished) {
      hint.innerText = "You can continue when the timer reaches 0.";
    } else if (!answered) {
      hint.innerText = "Please rate both questions to continue.";
    } else {
      hint.innerText = "Break finished — click Continue when ready.";
    }
  }
}

async function finishEpisodeBreak() {
  if (breakFinishedFired) return;
  breakFinishedFired = true;

  // Prefer TLX sliders if present; otherwise fall back to radio buttons.
  const md = document.getElementById('ep_mental_demand');
  const pf = document.getElementById('ep_performance');

  let mental_demand = null;
  let performance = null;

  if (md && pf) {
    mental_demand = _isTouched(md) ? parseInt(md.value) : null;
    performance   = _isTouched(pf) ? parseInt(pf.value) : null;

    if (mental_demand == null || performance == null) {
      alert("Please answer both questions before continuing.");
      breakFinishedFired = false;
      return;
    }
  } else {
    const qEffort = document.querySelector('input[name="ep_effort"]:checked');
    const qCoord  = document.querySelector('input[name="ep_coord"]:checked');
    mental_demand = qEffort ? parseInt(qEffort.value) : null;
    performance   = qCoord ? parseInt(qCoord.value) : null;
  }

  DataManager.saveEpisodeSurvey(
    STATE.episodeIndex,
    STATE.episodePhase,
    { mental_demand, performance }
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

  // --- Banner (make solo episodes obvious) ---
  const banner = document.getElementById('aiDayOffBanner');
  const bannerText = document.getElementById('aiDayOffBannerText');

  const next = (STATE.episodeIndex < CONFIG.TOTAL_EPISODES) ? (STATE.episodeIndex + 1) : null;
  const nextPhase = next ? getEpisodePhase(next) : null;

  const soloNext = next && (typeof isSoloEpisode === 'function') ? isSoloEpisode(next) : false;
  const soloNow  = (typeof isSoloEpisode === 'function') ? isSoloEpisode(STATE.episodeIndex) : false;

  if (banner) {
    if (nextPhase === 'replay_optimal') {
      banner.classList.remove('hidden');
      if (bannerText) bannerText.innerText = `Next episode (final): you will play again with the AI teammate from the end of the BO phase. Please try a NEW way to work with the AI.`;
    } else if (soloNext) {
      banner.classList.remove('hidden');
      if (bannerText) bannerText.innerText = `Next episode: your AI teammate has to charge its battery — you will play on your own.`;
    } else if (soloNow) {
      banner.classList.remove('hidden');
      if (bannerText) bannerText.innerText = `Next episode: your AI teammate is back.`;
    } else {
      banner.classList.add('hidden');
      if (bannerText) bannerText.innerText = '';
    }
  }

  // --- Update the short label text ---
  const epLabel = document.getElementById('breakEpisodeLabel');
  if (epLabel) {
    if (!next) {
      epLabel.innerText = `Episode ${STATE.episodeIndex} complete. You can finish when the countdown reaches 0.`;
    } else if (nextPhase === 'replay_optimal') {
      epLabel.innerText = `Episode ${STATE.episodeIndex} complete. Next: Final episode (try a new way to work with the AI).`;
    } else if (soloNext) {
      epLabel.innerText = `Episode ${STATE.episodeIndex} complete. Next: Episode ${next}.`;
    } else if (soloNow) {
      epLabel.innerText = `Episode ${STATE.episodeIndex} complete (solo). Next: Episode ${next}.`;
    } else {
      epLabel.innerText = `Episode ${STATE.episodeIndex} complete. Next: Episode ${next}.`;
    }
  }

  // --- Switch question wording for solo episodes ---
  const qMental = document.getElementById('tlx_q_mental');
  const qPerf   = document.getElementById('tlx_q_perf');
  if (qMental) {
    qMental.innerText = soloNow
      ? "How mentally demanding was it to play on your own?"
      : "How mentally demanding was it to play with the AI teammate?";
  }
  if (qPerf) {
    qPerf.innerText = soloNow
      ? "How successful were you in accomplishing the goal (serve as many dishes as possible while minimizing your own effort/steps)?"
      : "How successful were you and your teammate in accomplishing the goal (serve as many dishes as possible while minimizing your own effort/steps)?";
  }

  // --- Reset TLX sliders if present ---
  const md = document.getElementById('ep_mental_demand');
  const pf = document.getElementById('ep_performance');
  const mdV = document.getElementById('ep_mental_demand_value');
  const pfV = document.getElementById('ep_performance_value');
  const mdChip = document.getElementById('ep_mental_demand_chip');
  const pfChip = document.getElementById('ep_performance_chip');

  if (md && pf) {
    if (md) md.value = 10;
    if (pf) pf.value = 10;

    _setUntouched(md, mdChip, mdV);
    _setUntouched(pf, pfChip, pfV);

    const onMd = () => { _markTouched(md, mdChip, mdV); updateBreakContinueState(); };
    const onPf = () => { _markTouched(pf, pfChip, pfV); updateBreakContinueState(); };

    md.oninput = onMd;
    pf.oninput = onPf;
    md.onchange = onMd;
    pf.onchange = onPf;
  } else {
    // If using radio buttons, reset them
    document.querySelectorAll('input[name="ep_effort"], input[name="ep_coord"]').forEach(el => { el.checked = false; });
    document.querySelectorAll('input[name="ep_effort"], input[name="ep_coord"]').forEach(el => {
      el.onchange = () => updateBreakContinueState();
    });
  }

  const btn = document.getElementById('breakContinueBtn');
  if (btn) btn.onclick = () => finishEpisodeBreak().catch(console.error);

  // gating: timer must finish + both answers must be provided
  breakTimerFinished = false;
  updateBreakContinueState();

  // Start fixed break timer
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

      breakTimerFinished = true;
      updateBreakContinueState();
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
    // B. MAIN TASK LOGIC (Phase 1 and 2)
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

        DataManager.initUser(STATE.prolificId, age, gender, STATE.assignment, {
            experience: experience,
        });

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

// 1. Page 2a -> Move to 2b
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

// 2. Page 2b -> Move to 2c
const btnNext2b = document.getElementById('btn-next-2b');
if (btnNext2b) {
    btnNext2b.onclick = () => {
        const errors = calculatePageErrors(['q2a', 'q2b']);
        QUIZ_ERRORS += errors;
        
        console.log(`Page 2b Errors: ${errors} | Current Total: ${QUIZ_ERRORS}`);
        showPage('page-instruction-2c');
    };
}

// 3. Page 2c -> SUBMIT QUIZ
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