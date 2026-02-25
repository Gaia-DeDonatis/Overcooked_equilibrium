// static/js/dataManager.js

const DataManager = {
  LOGS: {
    prolificId: 'unknown',

    //participant metadata
    meta: {
      prolificId: 'unknown',
      age: null,
      gender: null,
      experience: null,

      assignment: {
        condition: null, // "BO" or "STATIC"
        map: null        // one of your 5 map labels (e.g. "asymmetric")
      },
      startTimeISO: null
    },

    // Episodes in chronological order
    episodes: [],

    // Rounds in chronological order (backend requires this)
    rounds: []
  },

  // ----------------------
  // 1) Participant/session
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
      episode_phase: episode_phase ?? null, // e.g. "seed" | "bo" | "stress"
      experiment_phase: null,               // 1 or 2 — set on first round of episode
      policy_id: null,
      optimal_policy_id: null,
      startTimeISO: new Date().toISOString(),

      // only 2 questions per episode
      feedback: {
        mental_effort: null,
        coordination_quality: null,
        submittedAtISO: null
      },

      // nested rounds
      rounds: []
    };

    this.LOGS.episodes.push(ep);
    return ep;
  },

  // convenience (used by your UI to show episode totals)
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

    // For stress episodes, record which policy was the optimal reference and how far the current (neighboring) policy is from it.
    if (episode_phase === 'stress') {
      if (extraMeta.optimalPolicyId != null && ep) ep.optimal_policy_id = extraMeta.optimalPolicyId ?? ep.optimal_policy_id ?? null;
    }

    if (this.LOGS.meta.tick_ms == null && extraMeta.tick_ms != null) {
      this.LOGS.meta.tick_ms = extraMeta.tick_ms;
    }

    if (this.LOGS.meta.round_duration_sec == null && extraMeta.round_duration_sec != null) {
      this.LOGS.meta.round_duration_sec = extraMeta.round_duration_sec;
    }
    if (this.LOGS.meta.rounds_per_episode == null && extraMeta.rounds_per_episode != null) {
      this.LOGS.meta.rounds_per_episode = extraMeta.rounds_per_episode;
    }
    if (this.LOGS.meta.seed == null && extraMeta.seed != null) {
      this.LOGS.meta.seed = extraMeta.seed;
    }

    const mapLabel = extraMeta.mapTopology ?? this.LOGS.meta.assignment.map ?? null;
    if (mapLabel != null && (phase == null || Number(phase) !== 0)) {
      this.LOGS.meta.assignment.map = mapLabel;
    }

    const round = {
      round_index_global: this.LOGS.rounds.length + 1,
      phase: phase ?? null,
      configId: configId ?? null,

      // nesting keys
      episode_index,
      episode_phase,
      experiment_phase,
      round_in_episode,

      // experiment variables
      policy_id: extraMeta.policyId ?? null,
      map: mapLabel,

      // distance from optimal policy (to set via backend)
      stress_policy_distance: extraMeta.stressPolicyDistance ?? null,

      startTimeISO: new Date().toISOString(),
      endTimeISO: null,

      // quick summary (PER ROUND)
      summary: {
        final_score: 0,
        dishes_served: 0,
        human_steps: 0
      },

      // ACTION LOGS (separate streams)
      action_log: {
        human: [],
        ai: []
      },

      // sparse events from backend (helps CRC/equilibrium)
      events: [],

    // wall-clock timestamp at round start (ms)
      _roundStartWallMs: Date.now()
    };

    this.LOGS.rounds.push(round);
    if (ep) ep.rounds.push(round);
  },

  getCurrentRound() {
    return this.LOGS.rounds.length ? this.LOGS.rounds[this.LOGS.rounds.length - 1] : null;
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

  logStep(serverData, humanKey, timing = null) {
    const r = this.getCurrentRound();
    if (!r || !serverData) return;

    const state = serverData.state || {};
    const agents = Array.isArray(state.agents) ? state.agents : [];

    // Backend indexing: [0]=AI, [1]=Human
    const ai = agents[0] || {};
    const human = agents[1] || {};

    const t = (typeof state.cur_step === 'number') ? state.cur_step : r.action_log.human.length;
    const wall_ms = (r._roundStartWallMs != null) ? (Date.now() - r._roundStartWallMs) : null;

    // Update per-round summary with backend counters
    if (typeof serverData.cumulative_reward === 'number') r.summary.final_score = serverData.cumulative_reward;
    if (typeof serverData.dishes_served === 'number') r.summary.dishes_served = serverData.dishes_served;
    if (humanKey !== 'Stay') r.summary.human_steps += 1;

    const aiLast = serverData.robot_last_action || {};
    const aiLow = (aiLast.low_level_action != null) ? aiLast.low_level_action : null;

     const humanHolding = this._packHolding(human);
    const aiHolding = this._packHolding(ai);

    const humanEntry = {
      t,
      wall_ms,
      action: this._normalizeHumanAction(humanKey),
      pos: (human.x != null && human.y != null) ? [human.x, human.y] : null,
    };
    if (humanHolding != null) humanEntry.holding = humanHolding;
    r.action_log.human.push(humanEntry);

    const aiEntry = {
      t,
      wall_ms,
      action: this._normalizeAiAction(aiLow),
      pos: (ai.x != null && ai.y != null) ? [ai.x, ai.y] : null,
    };
    if (aiHolding != null) aiEntry.holding = aiHolding;
    r.action_log.ai.push(aiEntry);

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
    ep.feedback = {
      mental_effort: (answers?.mental_effort != null) ? answers.mental_effort : null,
      coordination_quality: (answers?.coordination_quality != null) ? answers.coordination_quality : null,
      submittedAtISO: new Date().toISOString()
    };
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
    for (const r of payload.rounds) {
      delete r._roundStartWallMs;
    }
    // strip from nested episode.rounds
    for (const ep of payload.episodes) {
      for (const r of ep.rounds) {
        delete r._roundStartWallMs;
      }
    }

    const res = await fetch(`${SERVER_URL}/submit_log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ log: payload })
    });


     const _res = await fetch(`${SERVER_URL}/close_optimizer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prolificId: STATE.prolificId})
    });
    
    return await res.json();
  }
};