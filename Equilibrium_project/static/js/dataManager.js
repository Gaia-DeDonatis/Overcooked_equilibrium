// static/js/dataManager.js

const DataManager = {
  OPTS: {
    LOG_STATE_EACH_TICK: true,
    LOG_COUNTERS_EACH_TICK: true,
    LOG_WALL_MS: true
  },

  LOGS: {
    prolificId: 'unknown',
    meta: {
      prolificId: 'unknown',
      age: null,
      gender: null,
      experience: null,
      assignment: { condition: null, map: null },
      startTimeISO: null,
      tick_ms: null,
      round_duration_sec: null,
      rounds_per_episode: null,
      seed: null,
      client_config_snapshot: null,
      user_agent: (typeof navigator !== 'undefined') ? navigator.userAgent : null
    },
    episodes: [],
    rounds: []
  },

  // ----------------------
  // 1) Init / metadata
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
      if (assigned.map != null) this.LOGS.meta.assignment.map = assigned.map;
      if (assigned.condition != null) this.LOGS.meta.assignment.condition = assigned.condition;
    }

    if (extraMeta.tick_ms != null) this.LOGS.meta.tick_ms = extraMeta.tick_ms;
    if (extraMeta.round_duration_sec != null) this.LOGS.meta.round_duration_sec = extraMeta.round_duration_sec;
    if (extraMeta.rounds_per_episode != null) this.LOGS.meta.rounds_per_episode = extraMeta.rounds_per_episode;
    if (extraMeta.seed != null) this.LOGS.meta.seed = extraMeta.seed;

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
      episode_phase: episode_phase ?? null,
      experiment_phase: null,
      policy_id: null,
      optimal_policy_id: null,
      startTimeISO: new Date().toISOString(),
      feedback: {
        scale: "tlx_20",
        mental_effort: null,
        coordination_quality: null,
        submittedAtISO: null
      },
      round_index_globals: [],
      rounds: []
    };

    this.LOGS.episodes.push(ep);
    return ep;
  },

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
    if (ep && experiment_phase != null) ep.experiment_phase = experiment_phase;
    if (ep && extraMeta.policyId != null) ep.policy_id = extraMeta.policyId;
    if (episode_phase === 'stress' && ep && extraMeta.optimalPolicyId != null) ep.optimal_policy_id = extraMeta.optimalPolicyId;

    if (this.LOGS.meta.tick_ms == null && extraMeta.tick_ms != null) this.LOGS.meta.tick_ms = extraMeta.tick_ms;
    if (this.LOGS.meta.round_duration_sec == null && extraMeta.round_duration_sec != null) this.LOGS.meta.round_duration_sec = extraMeta.round_duration_sec;
    if (this.LOGS.meta.rounds_per_episode == null && extraMeta.rounds_per_episode != null) this.LOGS.meta.rounds_per_episode = extraMeta.rounds_per_episode;
    if (this.LOGS.meta.seed == null && extraMeta.seed != null) this.LOGS.meta.seed = extraMeta.seed;

    const mapLabel = extraMeta.mapTopology ?? this.LOGS.meta.assignment.map ?? null;
    if (mapLabel != null) this.LOGS.meta.assignment.map = mapLabel;

    const round_index_global = this.LOGS.rounds.length + 1;

    const round = {
      round_index_global,
      phase: phase ?? null,
      configId: configId ?? null,

      episode_index,
      episode_phase,
      experiment_phase,
      round_in_episode,

      policy_id: extraMeta.policyId ?? null,

      // mappa
      static_map: null,
      xlen: null,
      ylen: null,
      map: mapLabel,

      summary: {
        final_score: 0,
        dishes_served: 0,
        human_steps: 0
      },

      // ACTION LOGs
      action_log: {
        human: [], // {t, key, action, pos, holding, wall_ms?, timing?}
        ai: []     // {t, low, macro, arrow, pos, holding, wall_ms?}
      },

      state_log: [],    // {t, state:{agents,items}, wall_ms?} + reset_state
      counter_log: [],  // {t, score, dishes_served, wall_ms?}

      _roundStartWallMs: Date.now()
    };

    this.LOGS.rounds.push(round);
    if (ep) ep.round_index_globals.push(round_index_global);
  },

  getCurrentRound() {
    return this.LOGS.rounds.length ? this.LOGS.rounds[this.LOGS.rounds.length - 1] : null;
  },

  // inital state
  setRoundInitialState(state) {
    const r = this.getCurrentRound();
    if (!r || !state) return;

    if (r.static_map == null && state.map != null) r.static_map = state.map;
    if (r.xlen == null && state.xlen != null) r.xlen = state.xlen;
    if (r.ylen == null && state.ylen != null) r.ylen = state.ylen;

    if (this.OPTS.LOG_STATE_EACH_TICK) {
      r.state_log.push({
        t: (typeof state.cur_step === 'number') ? state.cur_step : 0,
        wall_ms: 0,
        kind: "reset_state",
        state: this._compactState(state)
      });
    }
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

  _aiArrowFromLowLevel(a) {
    const map = { 0: 'RIGHT', 1: 'DOWN', 2: 'LEFT', 3: 'UP', 4: 'STAY' };
    return (a != null) ? (map[a] ?? String(a)) : null;
  },

  _packHolding(agent) {
    if (!agent) return null;
    const h = agent.holding;
    if (h == null) return null;
    if (typeof h === 'string') return { item: h, containing: agent.holding_containing ?? null };
    return h;
  },

  _compactState(state) {
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

    // agents[0]=AI, agents[1]=Human
    const ai = agents[0] || {};
    const human = agents[1] || {};

    const t = (typeof state.cur_step === 'number') ? state.cur_step : r.action_log.human.length;
    const wall_ms = (this.OPTS.LOG_WALL_MS && r._roundStartWallMs != null) ? (Date.now() - r._roundStartWallMs) : null;

    // static map
    if (r.static_map == null && state.map != null) r.static_map = state.map;
    if (r.xlen == null && state.xlen != null) r.xlen = state.xlen;
    if (r.ylen == null && state.ylen != null) r.ylen = state.ylen;

    // summary
    if (typeof serverData.cumulative_reward === 'number') r.summary.final_score = serverData.cumulative_reward;
    if (typeof serverData.dishes_served === 'number') r.summary.dishes_served = serverData.dishes_served;
    if (humanKey !== 'Stay') r.summary.human_steps += 1;

    // AI
    const aiLast = serverData.robot_last_action || null;
    const aiLow = (aiLast && aiLast.low_level_action != null) ? Number(aiLast.low_level_action) : null;
    const aiMacro = (aiLast && aiLast.ai_macro_action != null) ? Number(aiLast.ai_macro_action) : null;

    // 1) human
    r.action_log.human.push({
      t,
      ...(this.OPTS.LOG_WALL_MS ? { wall_ms } : {}),
      key: humanKey ?? null,
      action: this._normalizeHumanAction(humanKey),
      pos: (human.x != null && human.y != null) ? [human.x, human.y] : null,
      holding: this._packHolding(human),
      ...(timing && typeof timing === 'object' ? { timing } : {})
    });

    // 2) AI
    r.action_log.ai.push({
      t,
      ...(this.OPTS.LOG_WALL_MS ? { wall_ms } : {}),
      macro: aiMacro,
      low: aiLow,
      arrow: this._aiArrowFromLowLevel(aiLow),
      pos: (ai.x != null && ai.y != null) ? [ai.x, ai.y] : null,
      holding: this._packHolding(ai)
    });

    // 3) counters
    if (this.OPTS.LOG_COUNTERS_EACH_TICK) {
      r.counter_log.push({
        t,
        ...(this.OPTS.LOG_WALL_MS ? { wall_ms } : {}),
        score: (typeof serverData.cumulative_reward === 'number') ? serverData.cumulative_reward : null,
        dishes_served: (typeof serverData.dishes_served === 'number') ? serverData.dishes_served : null
      });
    }

    // 4) environment state (x replay)
    if (this.OPTS.LOG_STATE_EACH_TICK) {
      r.state_log.push({
        t,
        ...(this.OPTS.LOG_WALL_MS ? { wall_ms } : {}),
        state: this._compactState(state)
      });
    }
  },

  // ----------------------
  // 5) Episode feedback
  // ----------------------
  saveEpisodeSurvey(episode_index, episode_phase, answers) {
    const ep = this._ensureEpisode(episode_index, episode_phase);
    if (!ep) return;

    if (answers?.mental_effort != null) ep.feedback.mental_effort = answers.mental_effort;
    if (answers?.coordination_quality != null) ep.feedback.coordination_quality = answers.coordination_quality;
    ep.feedback.submittedAtISO = new Date().toISOString();
  },

  // ---------------
  // 6) Submission
  // ---------------
  async submitToServer() {
    const payload = JSON.parse(JSON.stringify(this.LOGS));

    for (const r of payload.rounds) delete r._roundStartWallMs;

    for (const ep of payload.episodes) {
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
