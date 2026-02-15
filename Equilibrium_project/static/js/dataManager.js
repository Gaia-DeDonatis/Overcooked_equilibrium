const DataManager = {
    // 1. Data Structure
    LOGS: {
        prolificId: "unknown",
        startTime: null,
        rounds: [],
        questionnaires: {
            phase1: {},
            phase2: {}
        },
        finalFeedback: ""
    },

    // 2. Initialization
    initUser(prolificId, age, gender, assignedCondition) {
        this.LOGS.prolificId = prolificId;
        this.LOGS.startTime = new Date().toISOString();
        this.LOGS.metadata = {
            age: age,
            gender: gender,
            condition: assignedCondition
        };
        console.log("DataManager Initialized for:", prolificId);
    },

    // 3. Round Management
    startNewRound(phase, configId) {
        const newRound = {
            phase: phase,
            configId: configId,
            startTime: Date.now(),
            steps: [],
            finalScore: 0,
            dishesServed: 0,
            humanSteps: 0
        };

        this.LOGS.rounds.push(newRound);
    },

    // 4. Logging Actions
    logStep(serverData, humanKey) {
        const currentRound = this.LOGS.rounds[this.LOGS.rounds.length - 1];
        if (!currentRound) return;

        // A. Calculate the Score Jump 
        const previousScore = currentRound.finalScore || 0;
        const currentScore = serverData.cumulative_reward;
        const delta = currentScore - previousScore;

        // B. Detect a Delivery
        if (delta > 100) {
            currentRound.dishesServed = (currentRound.dishesServed || 0) + 1;
        }

        // C. Update the current score for the next frame's comparison
        currentRound.finalScore = currentScore;

        // D. Log the step
        currentRound.steps.push({
            step: serverData.state.cur_step,
            timestamp: Date.now(),
            humanAction: humanKey,
            reward: currentScore,
            agents: serverData.state.agents || [] 
        });

        if (humanKey !== 'Stay') {
            currentRound.humanSteps++;
        }
    },

    // 5. End Round
    endRound() {
        const currentRound = this.LOGS.rounds[this.LOGS.rounds.length - 1];
        if (currentRound) {
            currentRound.endTime = Date.now();
            currentRound.duration = (currentRound.endTime - currentRound.startTime) / 1000;
        }
    },

    // 6. Questionnaires
    saveQuestionnaire(phase, data) {
        if (phase === 1) this.LOGS.questionnaires.phase1 = data;
        if (phase === 2) this.LOGS.questionnaires.phase2 = data;
    },

    saveFinalFeedback(text) {
        this.LOGS.finalFeedback = text;
    },

    // 7. Submission
    async submitToServer() {
        console.log("Submitting Data...", this.LOGS);
        // to match the @app.route('/submit_log') in backend_new.py
        return await fetch(`${SERVER_URL}/submit_log`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ log: this.LOGS }) 
        }).then(res => res.json());
    }
};