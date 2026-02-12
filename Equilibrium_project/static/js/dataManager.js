// dataManager.js
//this file is still not connected, it's just a draft of the data we need to collect from the partecipants

const DataManager = {
    // 1. Data Structure
    LOGS: {
        metadata: {
            participantId: null,
            condition: null, // "Control" or "Experiment"
            age: null,
            gender: null,
            //sessionStart: null
        },
        preGameQuiz: { attempts: 0, passed: false },
        phase1: { rounds: [], questionnaire: {} },
        phase2: { rounds: [], questionnaire: {} },
        finalFeedback: ""
    },


    // 2. Metadata & Quiz Handlers
    initUser(prolificId, age, gender, assignedGroup) {
        this.LOGS.metadata = {
            participantId: prolificId,
            condition: assignedGroup, // Received from the backend
            age: age,
            gender: gender,
            //sessionStart: new Date().toISOString()
        };
        
        STATE.condition = assignedGroup; 
    },

    saveQuiz(attempts, passed) {
        this.LOGS.preGameQuiz = { 
            attempts, 
            passed, 
            //timestamp: new Date().toISOString() 
        };
    },

    // 3. Round Management (Level 1)
    startNewRound(phaseNumber, mapTopology, policyId) {
        const phaseKey = `phase${phaseNumber}`;
        const newRound = {
            // STATIC DATA (Does not change during the round)
            roundMetadata: {
                roundNumber: this.LOGS[phaseKey].rounds.length + 1,
                map: mapTopology,
                policy_id: policyId,
                startTime: Date.now()
            },
            // SUMMARY METRICS (Updated live, but represents the whole round)
            metrics: { 
                totalScore: 0, 
                humanSubtasks: 0, 
                aiSubtasks: 0 
            },
            // THE ACTION LOG (Level 2 - where the stream goes)
            stateUpdateLog: [] 
        };
        this.LOGS[phaseKey].rounds.push(newRound);
    },

    // 4. TELEMETRY: State-Update Logger (Level 3)
    logAction(serverData, humanKey, phaseNumber) {
        const phaseKey = `phase${phaseNumber}`;
        const rounds = this.LOGS[phaseKey].rounds;
        const currentRound = rounds[rounds.length - 1];
        
        if (!currentRound) return;

        const entry = {
            timestamp: Date.now(),
            // POSITIONS & ACTIONS
            agents: {
                human: {
                    pos: serverData.human_pos, 
                    action: humanKey
                },
                ai: {
                    pos: serverData.ai_pos,    
                    action: serverData.ai_action
                }
            },
            // OBJECT STATES (The "Kitchen State")
            objects: serverData.objects ? serverData.objects.map(obj => ({
                type: obj.name,                  
                pos: [obj.x, obj.y],
                state: obj.status // e.g., "fresh", "chopped", "plated"
            })) : []
        };
        
        currentRound.stateUpdateLog.push(entry);

        // Update the Level 1 score automatically
        currentRound.metrics.totalScore = serverData.cumulative_reward || 0;
    },

    // 5. Questionnaire & Final Submission
    saveQuestionnaire(phaseNumber, answers) {
        this.LOGS[`phase${phaseNumber}`].questionnaire = answers;
    },

    async submitToServer() {
        console.log("Submitting final log...", this.LOGS);
        return await fetch('/submit_log', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(this.LOGS)
        }).then(res => res.json());
    },

    saveFinalFeedback(text) {
        this.LOGS.finalFeedback = text;
    }
};