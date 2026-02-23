// static/js/config.js

const SERVER_URL = 'http://localhost:5000';

// Experiment Configuration
const CONFIG = {
  PRACTICE_SCORE: 200,
  ROUND_DURATION_SEC: 5,

  // Episode structure
  ROUNDS_PER_EPISODE: 3,
  EPISODES_SEED: 1, //put 3
  EPISODES_BO: 1,   //put 5
  EPISODES_STRESS: 1, //put 3

  // Between-episode break
  EPISODE_BREAK_SEC: 15
};

CONFIG.TOTAL_EPISODES = CONFIG.EPISODES_SEED + CONFIG.EPISODES_BO + CONFIG.EPISODES_STRESS;

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

  // Gameplay flags
  isPlaying: false,
  gameOver: false,

  // Practice
  practiceScore: 0,

  // Backend config id (backend currently ignores for main task)
  configId: null,

  // Condition assignment (kept for logging / future)
  assignment: {
    layout: null
  }
};