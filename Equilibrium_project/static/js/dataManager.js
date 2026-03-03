// static/js/dataManager.js

const DataManager = {
  LOGS: {
    prolificId: 'unknown',

    // participant metadata
    meta: {
      prolificId: 'unknown',
      age: null,
      gender: null,
      experience: null,

      assignment: {
        condition: null, // "BO" or "STATIC" (or any label you use)
        map: null        // one of your map labels (e.g. "ring")
      },

      // run-level timing
      startTimeISO: null,

      // experiment parameters (filled when known)
      tick_ms: null,
      round_duration_sec: null,
      rounds_per_episode: null,

      // if you use a global seed (optional)
      seed: null,

      // optional: store a snapshot of the client config for reproducibility
      client_config_snapshot: null,

      // optional: browser info for debugging
      user_agent: (typeof navigator !== 'undefined') ? navigator.userAgent : null
    },

    // episodes (non-redundant: no nested round objects, only references)
    episodes: [],

    // canonical list of all rounds (the heavy data)
    rounds: []
  },

  // ----------------------
  // 1) Init / user metadata
  // ----------------------
  initUser(prolificId, age, gender, assigned = {}, extraMeta = {}) {
    const pid = prolificId || 'unknown';
    this.LOGS.prolificId = pid;
    this.LOGS.meta.prolificId = pid;
    this.LOGS.meta.age = (age != null) ? age : null;
    this.LOGS.meta.gender = (gender != null) ? gender : null;
    this.LOGS.meta.experience = extraMeta.experience ?? null;
    this.LOGS.meta.startTimeISO = new Date().toISOString();

    if (typeof assigned === 'string') {
      this.LOGS.meta.assignment.condition = assigned;
    } else if (assigned && typeof assigned === 'object') {
      const map = assigned.map ?? null;
      const condition = assigned.condition ?? null;
      if (map != null) this.LOGS.meta.assignment.map = map;
      if (condition != null) this.LOGS.meta.assignment.condition = condition;
    }

    if (extraMeta.tick_ms != null) this.LOGS.meta.tick_ms = extraMeta.tick_ms;
    if (extraMeta.round_duration_sec != null) this.LOGS.meta.round_duration_sec = extraMeta.round_duration_sec;
    if (extraMeta.rounds_per_episode != null) this.LOGS.meta.rounds_per_episode = extraMeta.rounds_per_episode;
    if (extraMeta.seed != null) this.LOGS.meta.seed = extraMeta.seed;

    // Optional: store client config snapshot once (pass CONFIG from controller if you want)
    if (extraMeta.client_config_snapshot != null && this.LOGS.meta.client_config_snapshot == null) {
      this.LOGS.meta.client_config_snapshot = extraMeta.client_config_snapshot;
    }
  },

  // ----------------------
  // 2) Episode helpers
  // ----------------------
  _getEpisode(episode_index) {
    return this.LOGS.episodes.find(e => e.episode_index === episode_index) || null;
  },

  _ensureEpisode(episode_index, episode_phase) {
    if (episode_index == null) return null;

    let ep = this._getEpisode(episode_index);
    if (ep) {
      if (episode_phase != null) ep.episode_phase = episode_phase;
      return ep;
    }

    ep = {
      episode_index,
      episode_phase: episode_phase ?? null, // e.g. "seed" | "bo" | "stress" | "replay_optimal"
      experiment_phase: null,               // set on first round of episode
      policy_id: null,                      // primary policy_id for the episode (if applicable)
      optimal_policy_id: null,              // for stress: reference optimal policy

      startTimeISO: new Date().toISOString(),

      // per-episode feedback (keep both naming conventions to avoid losing older fields)
      feedback: {
        scale: "tlx_20",
        mental_demand: null,        // legacy
        performance: null,          // legacy
        mental_effort: null,        // current controller fields
        coordination_quality: null, // current controller fields
        submittedAtISO: null
      },

      // non-redundant references to rounds in LOGS.rounds
      round_index_globals: [],

      // backward-compat: keep an empty array so older code that expects ep.rounds won't crash
      rounds: []
    };

    this.LOGS.episodes.push(ep);
    return ep;
  },

  // convenience (used by UI to show episode totals)
  getEpisodeTotals(episode_index) {
    const totals = { dishes_served: 0, human_steps: 0, final_score_sum: 0 };
    for (const r of this.LOGS.rounds) {
      if (r.episode_index !== episode_index) continue;
      totals.dishes_served += (r.summary?.dishes_served ?? 0);
      totals.human_steps += (r.summary?.human_steps ?? 0);
      totals.final_score_sum += (r.summary?.final_score ?? 0);
    }
    return totals;
  },

  // ----------------------
  // 3) Round lifecycle
  // ----------------------
  startNewRound(phase, configId, extraMeta = {}) {
    const episode_index = extraMeta.episode_index ?? null;
    const episode_phase = extraMeta.episode_phase ?? null;
    const experiment_phase = extraMeta.experiment_phase ?? null;
    const round_in_episode = extraMeta.round_in_episode ?? null;

    const ep = this._ensureEpisode(episode_index, episode_phase);

    if (ep && extraMeta.policyId != null) ep.policy_id = extraMeta.policyId;
    if (ep && experiment_phase != null) ep.experiment_phase = experiment_phase;

    // For stress episodes, record optimal reference policy (if provided)
    if (episode_phase === 'stress') {
      if (extraMeta.optimalPolicyId != null && ep) ep.optimal_policy_id = extraMeta.optimalPolicyId ?? ep.optimal_policy_id ?? null;
    }

    // store experiment parameters when first known
    if (this.LOGS.meta.tick_ms == null && extraMeta.tick_ms != null) this.LOGS.meta.tick_ms = extraMeta.tick_ms;
    if (this.LOGS.meta.round_duration_sec == null && extraMeta.round_duration_sec != null) this.LOGS.meta.round_duration_sec = extraMeta.round_duration_sec;
    if (this.LOGS.meta.rounds_per_episode == null && extraMeta.rounds_per_episode != null) this.LOGS.meta.rounds_per_episode = extraMeta.rounds_per_episode;
    if (this.LOGS.meta.seed == null && extraMeta.seed != null) this.LOGS.meta.seed = extraMeta.seed;

    const mapLabel = extraMeta.mapTopology ?? this.LOGS.meta.assignment.map ?? null;
    if (mapLabel != null && (phase == null || Number(phase) !== 0)) {
      this.LOGS.meta.assignment.map = mapLabel;
    }

    const round_index_global = this.LOGS.rounds.length + 1;

    const round = {
      round_index_global,
      phase: phase ?? null,
      configId: configId ?? null,

      // nesting keys
      episode_index,
      episode_phase,
      experiment_phase,
      round_in_episode,

      // experiment variables (policy / layout identifiers)
      policy_id: extraMeta.policyId ?? null,

      // store checkpoint/path identifiers to enable exact replay
      chosen_ckpt: extraMeta.chosenCkpt ?? null,
      chosen_policy_dir: extraMeta.chosenPolicyDir ?? null,
      layout_id: extraMeta.layoutId ?? null,
      backend_config_id: extraMeta.backendConfigId ?? null,

      // map label shown by frontend (e.g., "ring_5,5")
      map: mapLabel,

      // distance from optimal policy (stress)
      stress_policy_distance: extraMeta.stressPolicyDistance ?? null,
      optimal_policy_id: extraMeta.optimalPolicyId ?? null,

      // static map snapshot (saved once per round; avoids redundancy per tick)
      static_map: null, // will be filled on first logStep unless you set it explicitly
      xlen: null,
      ylen: null,

      startTimeISO: new Date().toISOString(),
      endTimeISO: null,

      // quick summary (PER ROUND)
      summary: {
        final_score: 0,
        dishes_served: 0,
        human_steps: 0
      },

      // FULL-FIDELITY PER-TICK LOG:
      // Each tick stores (a) actions, (b) rewards, (c) dynamic world snapshot.
      // This enables precise playback and post-hoc analysis.
      tick_log: [],

      // sparse events from backend (optional but useful)
      events: [],

      // wall-clock timestamp at round start (ms) — internal-only; removed on submission
      _roundStartWallMs: Date.now()
    };

    this.LOGS.rounds.push(round);

    // non-redundant: store only reference in episode
    if (ep) {
      ep.round_index_globals.push(round_index_global);
    }
  },

  getCurrentRound() {
    return this.LOGS.rounds.length ? this.LOGS.rounds[this.LOGS.rounds.length - 1] : null;
  },

  // Optional: if controller wants to store reset state before the first step
  setRoundInitialState(state) {
    const r = this.getCurrentRound();
    if (!r || !state) return;

    if (r.static_map == null && state.map != null) r.static_map = state.map;
    if (r.xlen == null && state.xlen != null) r.xlen = state.xlen;
    if (r.ylen == null && state.ylen != null) r.ylen = state.ylen;

    // also store a "tick 0" snapshot so you have the exact pre-action state
    // (if you call this right after /reset).
    r.tick_log.push({
      t: (typeof state.cur_step === 'number') ? state.cur_step : 0,
      wall_ms: 0,
      kind: "reset_state",
      state: this._compactState(state)
    });
  },

  endRound(extra = {}) {
    const r = this.getCurrentRound();
    if (!r) return;
    r.endTimeISO = new Date().toISOString();

    if (extra.final_score != null) r.summary.final_score = extra.final_score;
    if (extra.dishes_served != null) r.summary.dishes_served = extra.dishes_served;
    if (extra.human_steps != null) r.summary.human_steps = extra.human_steps;
  },

  // ----------------------
  // 4) Per-tick logging
  // ----------------------
  _normalizeHumanAction(key) {
    if (key === 'ArrowUp') return 'UP';
    if (key === 'ArrowDown') return 'DOWN';
    if (key === 'ArrowLeft') return 'LEFT';
    if (key === 'ArrowRight') return 'RIGHT';
    if (key === 'Stay') return 'STAY';
    return key ?? null;
  },

  _normalizeAiAction(a) {
    const map = { 0: 'RIGHT', 1: 'DOWN', 2: 'LEFT', 3: 'UP', 4: 'STAY' };
    return (a != null) ? (map[a] ?? String(a)) : null;
  },

  _packHolding(agent) {
    if (!agent) return null;
    const h = agent.holding;
    if (h == null) return null;
    if (typeof h === 'string') {
      return { item: h, containing: agent.holding_containing ?? null };
    }
    return h;
  },

  _compactState(state) {
    if (!state) return null;
    // Keep only dynamic parts each tick to avoid redundancy
    return {
      cur_step: state.cur_step ?? null,
      agents: Array.isArray(state.agents) ? state.agents : [],
      items: Array.isArray(state.items) ? state.items : []
    };
  },

  logStep(serverData, humanKey, timing = null) {
    const r = this.getCurrentRound();
    if (!r || !serverData) return;

    const state = serverData.state || {};
    const agents = Array.isArray(state.agents) ? state.agents : [];

    // Backend indexing: [0]=AI, [1]=Human
    const ai = agents[0] || {};
    const human = agents[1] || {};

    // step index
    const t = (typeof state.cur_step === 'number') ? state.cur_step : r.tick_log.length;

    // elapsed time since round start (ms)
    const wall_ms = (r._roundStartWallMs != null) ? (Date.now() - r._roundStartWallMs) : null;

    // Save static map once per round (avoid repeating per tick)
    if (r.static_map == null && state.map != null) r.static_map = state.map;
    if (r.xlen == null && state.xlen != null) r.xlen = state.xlen;
    if (r.ylen == null && state.ylen != null) r.ylen = state.ylen;

    // Update per-round summary with backend counters
    if (typeof serverData.cumulative_reward === 'number') r.summary.final_score = serverData.cumulative_reward;
    if (typeof serverData.dishes_served === 'number') r.summary.dishes_served = serverData.dishes_served;
    if (humanKey !== 'Stay') r.summary.human_steps += 1;

    // capture robot action info (full object, not only low-level)
    const aiLast = serverData.robot_last_action || null;
    const aiLow = (aiLast && aiLast.low_level_action != null) ? aiLast.low_level_action : null;
    const aiMacro = (aiLast && aiLast.ai_macro_action != null) ? aiLast.ai_macro_action : null;

    const tick = {
      t,
      wall_ms,

      // actions (store both raw and normalized)
      human: {
        raw_key: humanKey ?? null,
        action: this._normalizeHumanAction(humanKey),
        pos: (human.x != null && human.y != null) ? [human.x, human.y] : null,
        holding: this._packHolding(human)
      },
      ai: {
        // keep raw macro/low for exact reconstruction
        ai_macro_action: (aiMacro != null) ? Number(aiMacro) : null,
        low_level_action: (aiLow != null) ? Number(aiLow) : null,
        action: this._normalizeAiAction(aiLow),
        pos: (ai.x != null && ai.y != null) ? [ai.x, ai.y] : null,
        holding: this._packHolding(ai),
        robot_last_action: aiLast
      },

      // rewards / counters per tick
      reward: {
        raw_reward: (typeof serverData.raw_reward === 'number') ? serverData.raw_reward : null,
        adjusted_reward: (typeof serverData.adjusted_reward === 'number') ? serverData.adjusted_reward : null,
        cumulative_reward: (typeof serverData.cumulative_reward === 'number') ? serverData.cumulative_reward : null,
        dishes_served: (typeof serverData.dishes_served === 'number') ? serverData.dishes_served : null,
        steps_left: (typeof serverData.steps_left === 'number') ? serverData.steps_left : null
      },

      // dynamic world snapshot (no map)
      state: this._compactState(state)
    };

    // Optional client timing info (if passed)
    if (timing && typeof timing === 'object') {
      tick.timing = timing;
    }

    r.tick_log.push(tick);

    // Sparse backend events (optional but useful)
    if (Array.isArray(serverData.events) && serverData.events.length > 0) {
      for (const ev of serverData.events) {
        r.events.push({
          t,
          wall_ms,
          actor: ev.actor ?? null,
          event: ev.event ?? null,
          payload: ev.payload ?? null
        });
      }
    }
  },

  // ----------------------
  // 5) Episode feedback
  // ----------------------
  saveEpisodeSurvey(episode_index, episode_phase, answers) {
    const ep = this._ensureEpisode(episode_index, episode_phase);
    if (!ep) return;

    // Store with current field names
    if (answers?.mental_effort != null) ep.feedback.mental_effort = answers.mental_effort;
    if (answers?.coordination_quality != null) ep.feedback.coordination_quality = answers.coordination_quality;

    // Also mirror to legacy names (so you never lose data if some scripts expect those)
    if (answers?.mental_effort != null && ep.feedback.mental_demand == null) ep.feedback.mental_demand = answers.mental_effort;
    if (answers?.coordination_quality != null && ep.feedback.performance == null) ep.feedback.performance = answers.coordination_quality;

    ep.feedback.submittedAtISO = new Date().toISOString();
  },

  // backwards-compat (in case older pages still call these)
  saveQuestionnaire(_phase, _answers) {},
  saveFinalFeedback(_text) {},

  // ----------------------
  // 6) Submission
  // ----------------------
  async submitToServer() {
    // deep-clone logs and strip internal-only fields before sending
    const payload = JSON.parse(JSON.stringify(this.LOGS));

    // Remove internal-only timing field
    for (const r of payload.rounds) {
      delete r._roundStartWallMs;
    }

    // Ensure episodes do not contain duplicated rounds
    for (const ep of payload.episodes) {
      // keep backward-compat empty array
      if (Array.isArray(ep.rounds)) ep.rounds = [];
    }

    const res = await fetch(`${SERVER_URL}/submit_log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ log: payload })
    });

    const _res = await fetch(`${SERVER_URL}/close_optimizer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prolificId: STATE.prolificId })
    });

    return await res.json();
  }
};