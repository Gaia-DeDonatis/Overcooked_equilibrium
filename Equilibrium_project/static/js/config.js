// static/js/config.js

const SERVER_URL = 'http://localhost:5000';

// Configuration
const CONFIG = {
    PRACTICE_SCORE: 200, // Points needed to pass practice
    TOTAL_ROUNDS: 5     // Rounds per phase
};

// Global State (Shared across all files)
const STATE = {
    sessionId: null,
    phase: 0,        // 0: Practice, 1: Phase 1, 2: Phase 2
    round: 1,
    isPlaying: false,
    gameOver: false,
    practiceScore: 0,
    configId: null,  // Current layout ID sent to server
    assignment: {}   // Stores which model the user gets
};

// Logs to send to server
const LOGS = {
    prolificId: "",
    rounds: [],
    questionnaire: {}
};
let currentRoundSteps = []; // Buffer for current round steps