// static/js/config.js

const SERVER_URL = 'http://localhost:5000';

// Configuration
const CONFIG = {
    PRACTICE_SCORE: 200,
    ROUNDS_PER_PHASE: 7,
    ROUND_DURATION_SEC: 45
};

// Global State (Shared across all files)
const STATE = {
    sessionId: null,
    phase: 0,               // 0: Practice, 1: Phase 1, 2: Phase 2
    round: 1,               // Current round within phase (1-7)
    totalRounds: 0,         // Total rounds completed (0-14)
    isPlaying: false,
    gameOver: false,
    practiceScore: 0,
    configId: null,         // Current config sent to server (e.g., "layout1_model1")
    assignment: {
        layout: null,       // Assigned layout (layout1-4)
        phase1Model: null,  // Phase 1 model (model1-4)
        phase2Model: null   // Phase 2 model (model1-4)
    }
};

// Logs to send to server at the end
const LOGS = {
    prolificId: "",
    age: null,
    gender: "",
    assignment: {},
    rounds: [],             // Array of round data
    phase1Questionnaire: {},
    phase2Questionnaire: {},
    finalFeedback: ""
};

// Temporary buffer for current round
let currentRoundData = {
    roundNumber: 0,
    phase: 0,
    steps: [],
    finalScore: 0,
    humanSteps: 0
};