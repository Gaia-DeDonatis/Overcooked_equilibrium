// static/js/config.js

const SERVER_URL = 'http://localhost:5000';

// Configuration
const CONFIG = {
    PRACTICE_SCORE: 200,
    ROUNDS_PER_PHASE: 1, // to change with 10
    ROUND_DURATION_SEC: 45
};

// Global State
const STATE = {
    sessionId: null,
    prolificId: null,
    phase: 0,
    round: 1,
    totalRounds: 0,
    isPlaying: false,
    gameOver: false,
    practiceScore: 0,
    configId: null,
    assignment: {
        layout: null,       
        phase1Model: null,
        phase2Model: null
    }
};

/*
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
};*/