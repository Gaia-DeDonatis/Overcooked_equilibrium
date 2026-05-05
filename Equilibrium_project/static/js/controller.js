// static/js/controller.js


// --- 1. API HELPER ---
async function api(endpoint, data = {}) {
    if (!STATE.sessionId) {
        const storedSid = localStorage.getItem('session_id');
        if (storedSid) {
            STATE.sessionId = storedSid;
        }
    }

    if (!STATE.sessionId) {
      const res = await fetch(`${SERVER_URL}/new_session`, { method: 'POST' });
      const d = await res.json();
      STATE.sessionId = d.session_id;
      try {
          localStorage.setItem('session_id', d.session_id);
      } catch (err) {
          console.warn("Could not persist session_id:", err);
      }
    }

    const res = await fetch(`${SERVER_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...data, session_id: STATE.sessionId })
    });
    return await res.json();
}


// --- 2. CONDITION ASSIGNMENT ---
function assignConditions() {
    const pid = (STATE.prolificId || "test").trim();

    const availableMaps = (
        Array.isArray(CONFIG.EXPERIMENT_MAPS) && CONFIG.EXPERIMENT_MAPS.length
    ) ? CONFIG.EXPERIMENT_MAPS : ["circle", "counter", "thinpath"];

    // Keep assignment stable for the same participant
    const storageKey = `assigned_experiment_map_${pid}`;
    
    let savedMap = null;
    try {
        savedMap = localStorage.getItem(storageKey);
    } catch (err) {
        console.warn("Could not read assigned map from localStorage:", err);
    }

    if (savedMap && availableMaps.includes(savedMap)) {
        STATE.assignment.layout = savedMap;
        console.log("Loaded assigned map:", STATE.assignment.layout);
        return;
    }

    // Deterministic pseudo-random assignment from participant ID
    let h = 0;
    for (let i = 0; i < pid.length; i++) {
        h = (h * 31 + pid.charCodeAt(i)) >>> 0;
    }

    const assignedMap = availableMaps[h % availableMaps.length];
    STATE.assignment.layout = assignedMap;
    
    try {
        localStorage.setItem(storageKey, assignedMap);
    } catch (err) {
        console.warn("Could not persist assigned map:", err);
    }

    console.log("Assigned experiment map:", STATE.assignment.layout);
}

function getSelectionMode() {
    return 'bo';
}

STATE.assignment.condition = getSelectionMode();

// --- 3. PAGE NAVIGATION ---
function showPage(pageId) {
    const pages = [
        'page-intro', 'page-consent', 'page-instruction-1',
        'page-instruction-2a','page-instruction-2b','page-instruction-2c',
        'page-game', 'page-episode-break', 'page-submitting',
        'page-end', 'page-quiz-fail'
    ];
    
    pages.forEach(id => {
        const el = document.getElementById(id);
        if(el) el.classList.add('hidden');
    });

    const target = document.getElementById(pageId);
    if(target) target.classList.remove('hidden');
    window.scrollTo(0,0);

    if (pageId === 'page-game' || pageId === 'page-instruction-1') {
      focusGameSurface();
    }

    // Only set isPlaying for game pages
    const gamePages = ['page-game', 'page-instruction-1'];
    if (!gamePages.includes(pageId)) {
        STATE.isPlaying = false;
    }
}

function setupSuccessCompletionUI() {
  const wrapEl = document.getElementById('completionCodeWrap');
  const codeEl = document.getElementById('completionCode');
  const codeInlineEl = document.getElementById('completionCodeInline');
  const linkEl = document.getElementById('prolificReturnLink');

  if (wrapEl) wrapEl.classList.remove('hidden');
  if (codeEl) codeEl.innerText = PROLIFIC_SUCCESS_CODE;
  if (codeInlineEl) codeInlineEl.innerText = PROLIFIC_SUCCESS_CODE;
  if (linkEl) linkEl.href = PROLIFIC_SUCCESS_URL;
}

function setupFailCompletionUI() {
  const codeEl = document.getElementById('failCompletionCode');
  const codeInlineEl = document.getElementById('failCompletionCodeInline');
  const linkEl = document.getElementById('failProlificReturnLink');

  if (codeEl) codeEl.innerText = PROLIFIC_FAIL_CODE;
  if (codeInlineEl) codeInlineEl.innerText = PROLIFIC_FAIL_CODE;
  if (linkEl) linkEl.href = PROLIFIC_FAIL_URL;
}

function focusGameSurface() {
  const canvasId = (STATE.phase === 0) ? 'gameCanvas_practice' : 'gameCanvas';

  setTimeout(() => {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    canvas.setAttribute('tabindex', '0');
    try { window.focus(); } catch (err) {}
    try { canvas.focus({ preventScroll: true }); } catch (err) { canvas.focus(); }
  }, 0);
}

// --- 4. GAME INITIALIZATION (TIME-BASED) ---
function getEpisodePhase(episodeIndex) {
  if (episodeIndex <= CONFIG.EPISODES_SEED) return 'seed';

  // BO window: includes 5 BO AI acquisition episodes + 2 solo/no-AI episodes
  if (episodeIndex <= CONFIG.PHASE_BO_END) {
    if (isSoloEpisode(episodeIndex)) return 'solo';
    return 'bo';
  }

  // Separate replay-best episode
  if (episodeIndex <= CONFIG.PHASE_BO_REPLAY_END) return 'bo_replay_best';

  if (episodeIndex <= CONFIG.PHASE_STRESS_END) return 'stress';

  return 'replay_optimal';
}

function getExperimentPhase(episodeIndex) {
  // 1 = seed, 2 = BO (including replay-best), 3 = stress, 4 = final replay
  if (episodeIndex <= CONFIG.EPISODES_SEED) return 1;
  if (episodeIndex <= CONFIG.PHASE_BO_REPLAY_END) return 2;
  if (episodeIndex <= CONFIG.PHASE_STRESS_END) return 3;
  return 4;
}

let gameTimer = null;
let timeLeft = 0;
let aiTickTimer = null;
let aiTickInFlight = false;
const AI_TICK_MS = 150;

let autosaveTimer = null;
const AUTOSAVE_MS = 10000;

let bufferedHumanKey = 'Stay';

const PROLIFIC_SUCCESS_CODE = 'CK4KW637';
const PROLIFIC_SUCCESS_URL = 'https://app.prolific.com/submissions/complete?cc=CK4KW637';

const PROLIFIC_FAIL_CODE = 'CGDMBD6O';
const PROLIFIC_FAIL_URL = 'https://app.prolific.com/submissions/complete?cc=CGDMBD6O';


function stopAiTick() {
  if (aiTickTimer) {
    clearInterval(aiTickTimer);
    aiTickTimer = null;
  }
  aiTickInFlight = false;
}

function stopAutosave() {
  if (autosaveTimer) {
    clearInterval(autosaveTimer);
    autosaveTimer = null;
  }
}

function startAutosave() {
  stopAutosave();
  autosaveTimer = setInterval(() => {
    if (!STATE.isPlaying || STATE.gameOver) return;
    DataManager.persistLocalBackup('interval_autosave');
  }, AUTOSAVE_MS);
}

async function doOneTick() {
  if (!STATE.isPlaying || STATE.gameOver) return;
  if (aiTickInFlight) return;
  aiTickInFlight = true;

  const t0 = performance.now();

  try {
    const keyToSend = bufferedHumanKey || 'Stay';
    bufferedHumanKey = 'Stay';

    const tApi0 = performance.now();
    const data = await api('/key_event', {
      key: keyToSend,
      config_id: STATE.configId,
      map_type: STATE.assignment.layout
    });
    const tApi1 = performance.now();

    const tDraw0 = performance.now();
    drawGame(data.state, 'gameCanvas');
    const tDraw1 = performance.now();

    const tLog0 = performance.now();
    DataManager.logStep(data, keyToSend);
    const tLog1 = performance.now();

    console.log(
      `[FRONTEND_TICK] total=${(performance.now()-t0).toFixed(1)}ms ` +
      `api=${(tApi1-tApi0).toFixed(1)}ms ` +
      `draw=${(tDraw1-tDraw0).toFixed(1)}ms ` +
      `log=${(tLog1-tLog0).toFixed(1)}ms`
    );

    const roundObj = DataManager.getCurrentRound();
    if (roundObj && roundObj.summary) {
      if (data.dishes_served != null) roundObj.summary.dishes_served = data.dishes_served;
    }

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

function syncRoundHudFromLogs() {
  const roundNow = DataManager.getCurrentRound();

  const dishesNow = roundNow?.summary?.dishes_served ?? 0;
  const stepsNow  = roundNow?.summary?.human_steps ?? 0;

  const dishesEl = document.getElementById('dishesServed');
  if (dishesEl) dishesEl.innerText = String(dishesNow);

  const stepsEl = document.getElementById('humanSteps');
  if (stepsEl) stepsEl.innerText = String(stepsNow);
}

function applyResumeMeta(resume) {
  if (!resume) return;

  STATE.sessionId = resume.sessionId || localStorage.getItem('session_id') || STATE.sessionId || null;
  
  if (STATE.sessionId) {
    try {
        localStorage.setItem('session_id', STATE.sessionId);
    } catch (err) {
        console.warn("Could not persist restored session_id:", err);
    }
  }

  if (resume.phase != null) STATE.phase = resume.phase;
  if (resume.configId != null) STATE.configId = resume.configId;
  if (resume.episodeIndex != null) STATE.episodeIndex = resume.episodeIndex;
  if (resume.roundInEpisode != null) STATE.roundInEpisode = resume.roundInEpisode;
  if (resume.episodePhase != null) STATE.episodePhase = resume.episodePhase;
  if (resume.experimentPhase != null) STATE.experimentPhase = resume.experimentPhase;

  if (resume.layout != null) STATE.assignment.layout = resume.layout;
  if (resume.condition != null) STATE.assignment.condition = resume.condition;
}

async function resumeCurrentRoundFromServer(backup) {
  const resume = backup?.resume_meta || backup?.log?.meta?.resume_meta;
  if (!resume || resume.episodeIndex == null) return false;

  applyResumeMeta(resume);

  if (backup?.log) {
    DataManager.restoreLogs(backup.log);
  }

  STATE.gameOver = false;
  STATE.isPlaying = false;
  bufferedHumanKey = 'Stay';
  aiTickInFlight = false;

  showPage('page-game');
  updateGameUI();

  const data = await api('/get_state', {});
  if (!data?.success || !data?.state) {
    throw new Error(data?.error || 'Could not restore live round state');
  }

  STATE.phase = 1;
  STATE.configId = 'experiment';
  STATE.gameOver = false;
  STATE.isPlaying = true;

  drawGame(data.state, 'gameCanvas');
  focusGameSurface();
  startAiTick();
  startAutosave();

  const restoredTime =
    Number.isFinite(resume.timeLeft) && resume.timeLeft > 0
      ? resume.timeLeft
      : CONFIG.ROUND_DURATION_SEC;

  startTimer(restoredTime);
  updateGameUI();
  updateSkipPolicyUI();
  syncRoundHudFromLogs();

  return true;
}

async function restartCurrentEpisodeFromBackup(backup) {
  const resume = backup?.resume_meta || backup?.log?.meta?.resume_meta;
  if (!resume || resume.episodeIndex == null) return false;

  if (backup?.log) {
    DataManager.restoreLogs(backup.log);

    // Drop the interrupted episode so it can restart cleanly.
    DataManager.LOGS.rounds = DataManager.LOGS.rounds.filter(
      r => r.episode_index !== resume.episodeIndex
    );
    DataManager.LOGS.episodes = DataManager.LOGS.episodes.filter(
      ep => ep.episode_index !== resume.episodeIndex
    );
  }

  STATE.phase = 1;
  STATE.episodeIndex = resume.episodeIndex;
  STATE.roundInEpisode = 1;
  STATE.skipEpisodeRequested = false;
  STATE.episodePhase = getEpisodePhase(STATE.episodeIndex);
  STATE.experimentPhase = getExperimentPhase(STATE.episodeIndex);
  STATE.gameOver = false;
  STATE.configId = 'experiment';

  if (resume.layout != null) STATE.assignment.layout = resume.layout;
  if (resume.condition != null) STATE.assignment.condition = resume.condition;

  showPage('page-game');
  updateGameUI();
  await startRound({ newEpisode: true });
  return true;
}

function hasResumableBackup(backup) {
  const resume = backup?.resume_meta || backup?.log?.meta?.resume_meta;
  const hasRounds = Array.isArray(backup?.log?.rounds) && backup.log.rounds.length > 0;

  return Boolean(
    resume &&
    resume.canResume === true &&
    resume.phase === 1 &&
    resume.episodeIndex != null &&
    resume.roundInEpisode != null &&
    resume.isPlaying === true &&
    resume.gameOver !== true &&
    hasRounds
  );
}

async function tryResumeInterruptedSession() {
  const backup = DataManager.readLocalBackup(STATE.prolificId);
  if (!hasResumableBackup(backup)) return false;

  const resume = backup?.resume_meta || backup?.log?.meta?.resume_meta;

  const ok = window.confirm(
    `We found interrupted progress for Episode ${resume.episodeIndex}, ` +
    `Round ${resume.roundInEpisode ?? 1}. Press OK to resume. ` +
    `If the old live round is no longer available, the current episode will restart from the beginning.`
  );
  if (!ok) return false;

  try {
    return await resumeCurrentRoundFromServer(backup);
  } catch (err) {
    console.warn("Live round resume failed, restarting current episode:", err);
    return await restartCurrentEpisodeFromBackup(backup);
  }
}

function getLastSavedProlificId() {
  try {
    return localStorage.getItem('last_prolific_id') || null;
  } catch (err) {
    return null;
  }
}

async function tryAutoResumeOnLoad() {
  const savedPid = getLastSavedProlificId();
  if (!savedPid) return false;

  STATE.prolificId = savedPid;

  const resumed = await tryResumeInterruptedSession();
  return resumed;
}

// A. START EPISODE (3 rounds each)
async function startEpisode(episodeIndex) {
  if (!STATE.assignment || !STATE.assignment.layout) {
    assignConditions();
  }

  STATE.phase = 1; // main task
  STATE.episodeIndex = episodeIndex;
  STATE.roundInEpisode = 1;
  STATE.skipEpisodeRequested = false;
  
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
      map_type: STATE.assignment.layout,
      selection_mode: getSelectionMode(),
      episode_index: STATE.episodeIndex,
      round_in_episode: STATE.roundInEpisode,
      episode_phase: STATE.episodePhase,
      prolificId: STATE.prolificId,
      n_init: CONFIG.EPISODES_SEED,
      n_bo: CONFIG.EPISODES_BO_ACQUISITION,
      n_knn: CONFIG.EPISODES_STRESS,
      solo_episode: (typeof isSoloEpisode === 'function') ? isSoloEpisode(STATE.episodeIndex) : false,
      new_episode: !!newEpisode
    });
    bufferedHumanKey = 'Stay';
    aiTickInFlight = false;

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
      DataManager.setRoundInitialState(data.state);
        STATE.phase = 1;
        STATE.configId = 'experiment';
        STATE.gameOver = false;
        STATE.isPlaying = true;

      DataManager.persistLocalBackup('round_started');

      drawGame(data.state, 'gameCanvas');
      focusGameSurface();
      startAiTick();
      startAutosave();
      startTimer(CONFIG.ROUND_DURATION_SEC);
      updateGameUI();
      updateSkipPolicyUI();

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

// skip episode that creates problems
function updateSkipPolicyUI() {
  const wrap = document.getElementById('skipPolicyWrap');
  const btn = document.getElementById('skipPolicyBtn');
  const confirmBox = document.getElementById('skipConfirmBox');
  if (!wrap || !btn) return;

  const canShow =
    STATE.phase === 1 &&
    STATE.isPlaying &&
    !STATE.gameOver &&
    STATE.roundInEpisode >= 2;

  wrap.classList.toggle('hidden', !canShow);
  btn.disabled = !canShow;

  if (!canShow && confirmBox) {
    confirmBox.classList.add('hidden');
  }
}

async function actuallySkipCurrentPolicyEpisode() {
  if (!STATE.isPlaying || STATE.gameOver) return;

  STATE.skipEpisodeRequested = true;
  STATE.isPlaying = false;
  STATE.gameOver = true;

  if (gameTimer) {
    clearInterval(gameTimer);
    gameTimer = null;
  }

  stopAiTick();
  stopAutosave();
  bufferedHumanKey = 'Stay';

  DataManager.endRound();
  await DataManager.saveProgressToServer('episode_skipped');

  const soloNow = (typeof isSoloEpisode === 'function') ? isSoloEpisode(STATE.episodeIndex) : false;
  const replayPhaseNow = ['bo_replay_best', 'replay_optimal'].includes(STATE.episodePhase);
  const shouldTell = !soloNow && !replayPhaseNow && getSelectionMode() === 'bo';

  if (shouldTell) {
    await api('/tell', {
      prolificId: STATE.prolificId,
      map_type: STATE.assignment.layout,
      selection_mode: getSelectionMode()
    });
  }

  showEpisodeBreak();
}

function setupSkipPolicyUI() {
  const skipBtn = document.getElementById('skipPolicyBtn');
  const confirmBox = document.getElementById('skipConfirmBox');
  const yesBtn = document.getElementById('skipConfirmYes');
  const noBtn = document.getElementById('skipConfirmNo');

  if (!skipBtn || !confirmBox || !yesBtn || !noBtn) return;

  skipBtn.onclick = () => {
    confirmBox.classList.remove('hidden');
  };

  noBtn.onclick = () => {
    confirmBox.classList.add('hidden');
  };

  yesBtn.onclick = () => {
    confirmBox.classList.add('hidden');

    actuallySkipCurrentPolicyEpisode().catch(err => {
      console.error("Skip policy error:", err);
      alert("Could not skip this episode. Please try again.");
    });
  };
}

// E. END ROUND
async function finishTimeBasedRound() {
    if (STATE.gameOver) return;
    
    STATE.isPlaying = false;
    STATE.gameOver = true;
    console.log("TIME IS UP!");

    stopAiTick();
    stopAutosave();
    bufferedHumanKey = 'Stay';

    DataManager.endRound();
    DataManager.saveProgressToServer('round_complete').catch(console.warn);

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
          if (sub) sub.innerText = `Next round in 3...`;

          overlay.classList.remove('hidden');
          overlay.style.opacity = '0';
          setTimeout(() => overlay.style.opacity = '1', 50);
        }

        let countdown = 3;
        const interval = setInterval(() => {
            countdown--;
            if(sub) sub.innerText = `Next round in ${countdown}...`;
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
        const replayPhaseNow = ['bo_replay_best', 'replay_optimal'].includes(STATE.episodePhase);
        const shouldTell = !soloNow && !replayPhaseNow && getSelectionMode() === 'bo';
                
        if (shouldTell) {
          await api('/tell', {
            prolificId: STATE.prolificId,
            map_type: STATE.assignment.layout,
            selection_mode: getSelectionMode()
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
  await DataManager.saveProgressToServer('episode_feedback');

  if (breakTimer) {
    clearInterval(breakTimer);
    breakTimer = null;
  }

  if (STATE.episodeIndex < CONFIG.TOTAL_EPISODES) {
    await startEpisode(STATE.episodeIndex + 1);
  } else {
    const btn = document.getElementById('breakContinueBtn');
    if (btn) {
      btn.disabled = true;
      btn.innerText = 'Saving...';
    }

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
    if (bannerText) bannerText.innerText = `Next episode (final): you will play one last episode with your AI teammate but this time try a NEW way to work with it.`;
  } else if (soloNext) {
    banner.classList.remove('hidden');
    if (bannerText) bannerText.innerText = `Next episode: your AI teammate is on a break — you will play on your own.`;
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
window.addEventListener('keydown', async (e) => {
    if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].indexOf(e.key) === -1) return;
    if(!STATE.isPlaying || STATE.gameOver) return;
    
    e.preventDefault();

    // A. PRACTICE LOGIC
    const practiceVisible = !document.getElementById('page-instruction-1')?.classList.contains('hidden');
    const gameVisible = !document.getElementById('page-game')?.classList.contains('hidden');

    // A. PRACTICE LOGIC
    if (practiceVisible) {
        const data = await practiceApi('/key_event', { key: e.key, config_id: 'layout_practice' });
        drawGame(data.state, 'gameCanvas_practice');
        
        const prevDishes = STATE.practiceDishes || 0;
        const dishesNow = (typeof data.dishes_served === 'number') ? data.dishes_served : 0;
        STATE.practiceDishes = dishesNow;

        if (dishesNow >= 1 && dishesNow > prevDishes) {
            STATE.isPlaying = false;
            STATE.gameOver = true;
            clearPracticeSession();
            document.getElementById('practiceHint').innerText = "Great job! Click 'Next' to continue.";
            document.getElementById('to-instruction-2').disabled = false;
        }
    }
    // B. MAIN TASK LOGIC
    else if (gameVisible) {
        bufferedHumanKey = e.key;
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
        STATE.assignment.condition = getSelectionMode();

        DataManager.initUser(STATE.prolificId, age, gender, STATE.assignment, {
            experience: experience,
        });

        showPage('page-consent');
    };
}

// --- 7. CONSENT PAGE ---
const consentCheck = document.getElementById('consentCheck');
const btnInstruction = document.getElementById('to-instruction');

if (consentCheck && btnInstruction) {
    consentCheck.addEventListener('change', () => {
        btnInstruction.disabled = !consentCheck.checked;
    });

    btnInstruction.onclick = async () => {
        if (!consentCheck.checked) return;

        DataManager.setConsent(true);

        const backup = DataManager.readLocalBackup(STATE.prolificId);

        // If interrupted main-task progress exists, try to resume that first.
        if (hasResumableBackup(backup)) {
            try {
                const resumed = await tryResumeInterruptedSession();
                if (resumed) return;
            } catch (err) {
                console.warn("Resume before practice failed:", err);
            }
        }

        showPage('page-instruction-1');

        setTimeout(() => {
            if (typeof startPracticeRound === 'function') {
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

function areAllQuestionsAnswered(questionNames) {
    return questionNames.every(name => {
        return document.querySelector(`input[name="${name}"]:checked`);
    });
}

function setupQuestionGate(questionNames, buttonId) {
    const button = document.getElementById(buttonId);
    if (!button) return;

    const updateButtonState = () => {
        const ready = areAllQuestionsAnswered(questionNames);
        button.disabled = !ready;
        button.style.opacity = ready ? '1' : '0.6';
        button.style.cursor = ready ? 'pointer' : 'not-allowed';
    };

    questionNames.forEach(name => {
        document.querySelectorAll(`input[name="${name}"]`).forEach(input => {
            input.addEventListener('change', updateButtonState);
        });
    });

    updateButtonState();
}

// Enable Next/Submit only when all answers on that page are selected
setupQuestionGate(['q1', 'q1b', 'q1c'], 'btn-next-2a');
setupQuestionGate(['q2a', 'q2b', 'q2c'], 'btn-next-2b');
setupQuestionGate(['q3a', 'q3c'], 'btn-submit-quiz');

// --- BUTTON LISTENERS ---

// 1. Page 2a -> Move to 2b
const btnNext2a = document.getElementById('btn-next-2a');
if (btnNext2a) {
    btnNext2a.onclick = () => {
        if (!areAllQuestionsAnswered(['q1', 'q1b', 'q1c'])) return;

        const errors = calculatePageErrors(['q1', 'q1b', 'q1c']);
        QUIZ_ERRORS += errors;

        console.log(`Page 2a Errors: ${errors} | Current Total: ${QUIZ_ERRORS}`);
        showPage('page-instruction-2b');
    };
}

// 2. Page 2b -> Move to 2c
const btnNext2b = document.getElementById('btn-next-2b');
if (btnNext2b) {
    btnNext2b.onclick = () => {
        if (!areAllQuestionsAnswered(['q2a', 'q2b', 'q2c'])) return;

        const errors = calculatePageErrors(['q2a', 'q2b', 'q2c']);
        QUIZ_ERRORS += errors;

        console.log(`Page 2b Errors: ${errors} | Current Total: ${QUIZ_ERRORS}`);
        showPage('page-instruction-2c');
    };
}

// 3. Page 2c -> start game
const btnSubmitQuiz = document.getElementById('btn-submit-quiz');
if (btnSubmitQuiz) {
    btnSubmitQuiz.onclick = () => {
        if (!areAllQuestionsAnswered(['q3a', 'q3c'])) return;

        const errors = calculatePageErrors(['q3a', 'q3c']);
        QUIZ_ERRORS += errors;

        console.log(`Final Check. Total Cumulative Errors: ${QUIZ_ERRORS}`);

        if (QUIZ_ERRORS > 2) {
          STATE.isPlaying = false;
          STATE.gameOver = true;
          stopAiTick();
          stopAutosave();
          if (gameTimer) clearInterval(gameTimer);

          showPage('page-quiz-fail');
          setupFailCompletionUI();
          return;
        } else {
            console.log("Quiz Passed. Revealing start button.");
            document.getElementById('submit-quiz-container').classList.add('hidden');
            document.getElementById('all-set-container').classList.remove('hidden');
        }
    };
}

// 4. START GAME
const btnStartTask = document.getElementById('start-task-1');
if (btnStartTask) {
  btnStartTask.onclick = async () => {
    try {
      const resumed = await tryResumeInterruptedSession();
      if (resumed) return;

      console.log("Starting Episode 1...");
      await startEpisode(1);
    } catch (err) {
      console.error("Start/resume error:", err);
      alert("Could not restore the interrupted session. Please contact the researcher.");
    }
  };
}

// --- 10. FINAL SUBMISSION ---
async function submitData() {
    showPage('page-submitting');

    // Let the browser paint the loading page before the network work starts
    await new Promise(resolve => setTimeout(resolve, 50));

    try {
        const response = await DataManager.submitToServer();

        if (response.success) {
            showPage('page-end');
            setupSuccessCompletionUI();
        } else {
            alert("Submission failed. Please contact the researcher.");
            showPage('page-episode-break');
        }
    } catch (err) {
        console.error("Submission error:", err);
        alert("Network error during submission. Please contact the researcher.");
        showPage('page-episode-break');
    }
}

// --- 11. INITIALIZE ---
window.onload = async () => {
    preloadImages(() => {
        console.log("Images loaded and game is ready.");
    });

    setupSkipPolicyUI();

    try {
        const resumed = await tryAutoResumeOnLoad();
        if (resumed) {
            console.log("Interrupted session resumed.");
        }
    } catch (err) {
        console.warn("Auto-resume on load failed:", err);
    }
};

window.addEventListener('resize', () => {
  const pageGameVisible = !document.getElementById('page-game')?.classList.contains('hidden');
  if (pageGameVisible && STATE?.isPlaying) {
    api('/get_state', {}).then(data => {
      if (data?.state) drawGame(data.state, 'gameCanvas');
    }).catch(() => {});
  }

  const practiceCanvas = document.getElementById('gameCanvas_practice');
  const practiceVisible = practiceCanvas && practiceCanvas.offsetParent !== null;
  if (practiceVisible) {
    practiceApi('/get_state', {}).then(data => {
      if (data?.state) drawGame(data.state, 'gameCanvas_practice');
    }).catch(() => {});
  }
});

window.addEventListener('pagehide', () => {
  DataManager.sendBeaconProgress('pagehide');
});

window.addEventListener('beforeunload', () => {
  DataManager.sendBeaconProgress('beforeunload');
});

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    DataManager.sendBeaconProgress('visibility_hidden');
  }
});