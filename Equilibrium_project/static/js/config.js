// static/js/config.js

// const SERVER_URL = 'http://localhost:5000';
const SERVER_URL = window.location.origin;

// Experiment Configuration
const CONFIG = {
  PRACTICE_SCORE: 200,
  ROUND_DURATION_SEC: 45, // put 45

    // Episode structure
  ROUNDS_PER_EPISODE: 3, // 3
  EPISODES_SEED: 5,      // put 5  # LOGIC BREAKS IF THIS IS <2
  EPISODES_BO: 5,        // put 5 BO + 2 without AI
  EPISODES_BO_SOLO: 2, //2
  BO_SOLO_POSITIONS: [3,5],
  EPISODES_BO_REPLAY_BEST: 1,
  EPISODES_STRESS: 3,    // put 3 kNN
  EPISODES_REPLAY_OPTIMAL: 1,
  EPISODE_BREAK_SEC: 10   // 10
};

CONFIG.EXPERIMENT_MAPS = ["circle", "counter", "thinpath"];
CONFIG.SELECTION_MODE = "bo";

// Total BO length shown to participants (AI BO episodes + solo "day off" episodes)
CONFIG.EPISODES_BO_TOTAL = CONFIG.EPISODES_BO + CONFIG.EPISODES_BO_SOLO;

// Phase boundaries used by controller.js
CONFIG.PHASE_BO_END = CONFIG.EPISODES_SEED + CONFIG.EPISODES_BO_TOTAL;
CONFIG.PHASE_BO_REPLAY_END = CONFIG.PHASE_BO_END + CONFIG.EPISODES_BO_REPLAY_BEST;
CONFIG.PHASE_STRESS_END = CONFIG.PHASE_BO_REPLAY_END + CONFIG.EPISODES_STRESS;

// Total episodes shown to participants
CONFIG.PHASE_REPLAY_END = CONFIG.PHASE_STRESS_END + CONFIG.EPISODES_REPLAY_OPTIMAL;
CONFIG.TOTAL_EPISODES = CONFIG.PHASE_REPLAY_END;

// High-level phase counts
CONFIG.PHASE_1_EPISODES = CONFIG.EPISODES_SEED + CONFIG.EPISODES_BO_TOTAL + CONFIG.EPISODES_BO_REPLAY_BEST;
CONFIG.PHASE_2_EPISODES = CONFIG.EPISODES_STRESS;
CONFIG.PHASE_3_EPISODES = CONFIG.EPISODES_REPLAY_OPTIMAL;

// --- Solo episode scheduling helpers ---

function _normalizeSoloPositions() {
  const total = CONFIG.EPISODES_BO_TOTAL;
  let pos = Array.isArray(CONFIG.BO_SOLO_POSITIONS) ? CONFIG.BO_SOLO_POSITIONS.slice() : [];

  // If the list is missing/wrong length, default to first N positions.
  if (pos.length !== CONFIG.EPISODES_BO_SOLO) {
    pos = Array.from({ length: CONFIG.EPISODES_BO_SOLO }, (_, i) => i + 1);
  }

  // Keep only valid unique ints within [1, BO_TOTAL]
  pos = [...new Set(pos.map(x => parseInt(x, 10)).filter(x => Number.isFinite(x) && x >= 1 && x <= total))].sort((a,b) => a-b);

  // If trimming caused fewer positions than needed, fill from the start.
  while (pos.length < CONFIG.EPISODES_BO_SOLO) {
    const candidate = pos.length ? pos[pos.length - 1] + 1 : 1;
    if (candidate <= total) pos.push(candidate);
    else break;
  }

  CONFIG.BO_SOLO_POSITIONS = pos;
}

_normalizeSoloPositions();

function isSoloEpisode(episodeIndex) {
  const boStart = CONFIG.EPISODES_SEED + 1;
  const boEnd = boStart + CONFIG.EPISODES_BO_TOTAL - 1;
  if (episodeIndex < boStart || episodeIndex > boEnd) return false;

  const boPos = episodeIndex - boStart + 1;
  return CONFIG.BO_SOLO_POSITIONS.includes(boPos);
}

function countSoloBeforeEpisode(episodeIndex) {
  const boStart = CONFIG.EPISODES_SEED + 1;
  const boEnd = boStart + CONFIG.EPISODES_BO_TOTAL - 1;

  if (episodeIndex < boStart) return 0;

  if (episodeIndex > boEnd) return CONFIG.BO_SOLO_POSITIONS.length;

  const boPos = episodeIndex - boStart + 1;
  return CONFIG.BO_SOLO_POSITIONS.filter(p => p < boPos).length;
}

// Global State
const STATE = {
  sessionId: null,
  prolificId: null,

  // 0 = practice, 1 = main task
  phase: 0,

  // Main experiment loop
  episodeIndex: 1,
  roundInEpisode: 1,
  episodePhase: 'seed',
  episodeScore: 0,

  // Gameplay flags
  isPlaying: false,
  gameOver: false,

  // Practice
  practiceScore: 0,

  // Backend config id (backend currently ignores for main task)
  configId: null,

  //episode skiper
  skipEpisodeRequested: false,

  // Condition assignment (BO only)
  assignment: {
    layout: null,
    condition: CONFIG.SELECTION_MODE
  }
};

const ROBOT_SKINS = [
  'agent-robot.png',
  'agent-robot2.png',
  'agent-robot3.png',
  'agent-robot4.png',
  'agent-robot5.png',
  'agent-robot6.png',
  'agent-robot7.png',
  'agent-robot8.png',
  'agent-robot9.png',
  'agent-robot10.png',
  'agent-robot11.png',
  'agent-robot12.png',
  'agent-robot13.png',
  'agent-robot14.png'
];
