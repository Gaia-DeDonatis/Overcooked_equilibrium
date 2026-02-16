const DataManager = {
  // 1) Top-level log structure
  LOGS: {
    meta: {
      prolificId: "unknown",
      age: null,
      gender: null,
      condition: null,          //control or experimental
      mapTopology: null,
      tick_ms: null, 
      seed: 42,                 //if fixed; otherwise set from backend
      policy_id_phase1: null,
      policy_id_phase2: null,
      startTimeISO: null
    },

    // phase1/phase2 questionnaires + feedback
    questionnaires: { phase1: {}, phase2: {} },
    finalFeedback: "",

    // rounds are ordered in time
    rounds: []
  },

  //  2) Initialization
  initUser(prolificId, age, gender, assignedCondition, extraMeta = {}) {
    this.LOGS.meta.prolificId = prolificId;
    this.LOGS.meta.age = age;
    this.LOGS.meta.gender = gender;
    this.LOGS.meta.condition = assignedCondition;
    this.LOGS.meta.startTimeISO = new Date().toISOString();

    // experiment-wide meta
    if (extraMeta.mapTopology != null) this.LOGS.meta.mapTopology = extraMeta.mapTopology;
    if (extraMeta.tick_ms != null) this.LOGS.meta.tick_ms = extraMeta.tick_ms;
    if (extraMeta.seed != null) this.LOGS.meta.seed = extraMeta.seed;

    // phase-level policy info
    if (extraMeta.policy_id_phase1 != null) this.LOGS.meta.policy_id_phase1 = extraMeta.policy_id_phase1;
    if (extraMeta.policy_id_phase2 != null) this.LOGS.meta.policy_id_phase2 = extraMeta.policy_id_phase2;

    console.log("DataManager Initialized for:", prolificId);
  },

  // 3) Round management
  startNewRound(phase, configId, extraMeta = {}) {
    const newRound = {
      phase,
      configId,
      round_index: extraMeta.round_index ?? null,

      mapTopology: extraMeta.mapTopology ?? null,
      policyId: extraMeta.policyId ?? null,
      chosenCkpt: extraMeta.chosenCkpt ?? null,

      startTimeISO: new Date().toISOString(),

      tape: [],

      // Sparse CRC/events
      events: [],

      // Summary fields (easy to analyze quickly)
      summary: {
        final_score: 0,
        dishes_served: 0,
        human_steps: 0
      },

      //for score delta detection
      _prevScore: 0
    };

    this.LOGS.rounds.push(newRound);

    if (this.LOGS.meta.tick_ms == null && extraMeta.tick_ms != null) this.LOGS.meta.tick_ms = extraMeta.tick_ms;
    if (this.LOGS.meta.mapTopology == null && extraMeta.mapTopology != null) this.LOGS.meta.mapTopology = extraMeta.mapTopology;
  },

  endRound(extra = {}) {
    const r = this.getCurrentRound();
    if (!r) return;
    r.endTimeISO = new Date().toISOString();

    // allow controller to pass dishes_served
    if (extra.final_score != null) r.summary.final_score = extra.final_score;
    if (extra.dishes_served != null) r.summary.dishes_served = extra.dishes_served;
    if (extra.human_steps != null) r.summary.human_steps = extra.human_steps;

    // cleanup internal
    delete r._prevScore;
  },

  getCurrentRound() {
    if (!this.LOGS.rounds.length) return null;
    return this.LOGS.rounds[this.LOGS.rounds.length - 1];
  },

  // 4) Logging
  /**
   * Log ONE applied environment step (one tick).
   * @param {object} serverData - response from backend /key_event
   * @param {string} humanKey   - the human key applied this step (Arrow... or Stay)
   * @param {number} timeLeftSec - seconds remaining in the round (countdown). Optional.
   */
  logStep(serverData, humanKey, timing = {}) {
    const r = this.getCurrentRound();
    if (!r || !serverData) return;

    const state = serverData.state || {};
    const agents = state.agents || [];

    // agent indexing: [0]=AI, [1]=Human (as in backend)
    const ai = agents[0] || {};
    const human = agents[1] || {};

    // step index
    const t = (typeof state.cur_step === "number") ? state.cur_step : null;

    // reward tracking
    const currentScore = (typeof serverData.cumulative_reward === "number") ? serverData.cumulative_reward : r.summary.final_score;
    const delta = currentScore - (r._prevScore || 0);
    r._prevScore = currentScore;
    r.summary.final_score = currentScore;

    // dish detection
    if (delta > 100) r.summary.dishes_served += 1;

    // human step count (non-Stay)
    if (humanKey !== "Stay") r.summary.human_steps += 1;

    // AI actions
    const aiLow = serverData.robot_last_action?.low_level_action ?? null;

    // ---- REPLAY TAPE ENTRY ----
    const applied_ms = (typeof timing?.appliedMs === "number") ? timing.appliedMs : null;
    const human_press_ms = (typeof timing?.humanPressMs === "number") ? timing.humanPressMs : null;

    r.tape.push({
    t,

    applied_ms,
    human_press_ms,

    human_action: humanKey,
    ai_low: aiLow,

    human_pos: (human.x != null && human.y != null) ? [human.x, human.y] : null,
    ai_pos: (ai.x != null && ai.y != null) ? [ai.x, ai.y] : null,

    human_holding: human.holding ?? null,
    ai_holding: ai.holding ?? null,

    score: currentScore,
    delta_score: delta
    });


    // ---- SPARSE EVENTS (CRC) ----
    // If backend returns serverData.events = [{actor:"human|ai", event:"pick_plate"}...]
    if (Array.isArray(serverData.events) && serverData.events.length > 0) {
      for (const ev of serverData.events) {
        r.events.push({
          t,
          applied_ms,
          actor: ev.actor ?? null,
          event: ev.event ?? ev
        });
      }
    }
  },

  // 5) Questionnaires & feedback
  saveQuestionnaire(phase, data) {
    if (phase === 1) this.LOGS.questionnaires.phase1 = data;
    if (phase === 2) this.LOGS.questionnaires.phase2 = data;
  },

  saveFinalFeedback(text) {
    this.LOGS.finalFeedback = text;
  },

  // 6) Submission
  async submitToServer() {
    console.log("Submitting Data...", this.LOGS);
    return await fetch(`${SERVER_URL}/submit_log`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ log: this.LOGS })
    }).then(res => res.json());
  }
};
