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
      consentGiven: false,
      startTimeISO: null,
      tick_ms: null,
    },
    episodes: [],
    rounds: []
  },

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
      if (assigned.layout != null) this.LOGS.meta.assignment.map = assigned.layout;
      if (assigned.condition != null) this.LOGS.meta.assignment.condition = assigned.condition;
    }

    if (extraMeta.tick_ms != null) this.LOGS.meta.tick_ms = extraMeta.tick_ms;
    if (extraMeta.round_duration_sec != null) this.LOGS.meta.round_duration_sec = extraMeta.round_duration_sec;
    if (extraMeta.rounds_per_episode != null) this.LOGS.meta.rounds_per_episode = extraMeta.rounds_per_episode;
    if (extraMeta.seed != null) this.LOGS.meta.seed = extraMeta.seed;

    if (extraMeta.client_config_snapshot != null && this.LOGS.meta.client_config_snapshot == null) {
      this.LOGS.meta.client_config_snapshot = extraMeta.client_config_snapshot;
    }
    try { localStorage.setItem('last_prolific_id', pid); } catch (err) {}
  },

  setConsent(consentGiven = true) {
    this.LOGS.meta.consentGiven = Boolean(consentGiven);
    this.LOGS.meta.consentTimeISO = consentGiven ? new Date().toISOString() : null;
  },

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
      /* optimal_policy_id: null,*/
      startTimeISO: new Date().toISOString(),
      feedback: {
        scale: "tlx_20",
        mental_demand: null,
        performance: null,
        submittedAtISO: null
      },
      round_index_globals: [],
      rounds: []
    };

    this.LOGS.episodes.push(ep);
    return ep;
  },

  getEpisodeTotals(episode_index) {
    const totals = {
      dishes_served: 0,
      human_steps: 0,
      ai_steps: 0,
      human_reward_score_sum: 0,
      ai_reward_score_sum: 0,
      team_reward_score_sum: 0
    };

    for (const r of this.LOGS.rounds) {
      if (r.episode_index !== episode_index) continue;
      totals.dishes_served += (r.summary?.dishes_served ?? 0);
      totals.human_steps += (r.summary?.human_steps ?? 0);
      totals.ai_steps += (r.summary?.ai_steps ?? 0);
      totals.human_reward_score_sum += (r.summary?.human_reward_score ?? 0);
      totals.ai_reward_score_sum += (r.summary?.ai_reward_score ?? 0);
      totals.team_reward_score_sum += (r.summary?.team_reward_score ?? 0);
    }

    return totals;
  },

  startNewRound(phase, configId, extraMeta = {}) {
    const episode_index = extraMeta.episode_index ?? null;
    const episode_phase = extraMeta.episode_phase ?? null;
    const experiment_phase = extraMeta.experiment_phase ?? null;
    const round_in_episode = extraMeta.round_in_episode ?? null;

    const ep = this._ensureEpisode(episode_index, episode_phase);
    if (ep && experiment_phase != null) ep.experiment_phase = experiment_phase;
    if (ep && extraMeta.policyId != null) ep.policy_id = extraMeta.policyId;
    /*if (['stress', 'bo_replay_best', 'replay_optimal'].includes(episode_phase) && ep && extraMeta.optimalPolicyId != null) {
      ep.optimal_policy_id = extraMeta.optimalPolicyId;
    }*/

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
      map: mapLabel,

      summary: {
        dishes_served: 0,
        human_steps: 0,
        ai_steps: 0,
        team_reward_score: 0,
        human_reward_score: 0,
        ai_reward_score: 0
      },

      action_log: {
        human: [],
        ai: []
      },

      state_log: [],
      counter_log: [],

      _roundStartWallMs: Date.now()
    };

    this.LOGS.rounds.push(round);
    if (ep) ep.round_index_globals.push(round_index_global);
  },

  getCurrentRound() {
    return this.LOGS.rounds.length ? this.LOGS.rounds[this.LOGS.rounds.length - 1] : null;
  },

  setRoundInitialState(state) {
    const r = this.getCurrentRound();
    if (!r || !state) return;

    if (r.static_map == null && state.map != null) r.static_map = state.map;
    if (r.xlen == null && state.xlen != null) r.xlen = state.xlen;
    if (r.ylen == null && state.ylen != null) r.ylen = state.ylen;

    const compact = this._compactState(state);
    if (r.initial_state == null) r.initial_state = compact;

    if (this.OPTS.LOG_STATE_EACH_TICK) {
      const alreadyLoggedReset = Array.isArray(r.state_log) && r.state_log.some(entry => entry && entry.kind === "reset_state");
      if (!alreadyLoggedReset) {
        r.state_log.push({
          t: (typeof state.cur_step === 'number') ? state.cur_step : 0,
          wall_ms: 0,
          kind: "reset_state",
          state: compact
        });
      }
    }
  },

  endRound(extra = {}) {
    const r = this.getCurrentRound();
    if (!r) return;
    r.endTimeISO = new Date().toISOString();

    if (extra.dishes_served != null) r.summary.dishes_served = extra.dishes_served;
    if (extra.human_steps != null) r.summary.human_steps = extra.human_steps;
    if (extra.ai_steps != null) r.summary.ai_steps = extra.ai_steps;
  },

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

  logStep(serverData, humanKey) {
    const r = this.getCurrentRound();
    if (!r || !serverData) return;

    const state = serverData.state || {};
    const agents = Array.isArray(state.agents) ? state.agents : [];

    const soloNow =
      (r?.policy_id === 'no_ai') ||
      (Array.isArray(agents) && agents.length === 1);

    const ai = soloNow ? {} : (agents[0] || {});
    const human = soloNow ? (agents[0] || {}) : (agents[1] || {});

    const t = (typeof state.cur_step === 'number') ? state.cur_step : r.action_log.human.length;
    const wall_ms = (this.OPTS.LOG_WALL_MS && r._roundStartWallMs != null)
      ? (Date.now() - r._roundStartWallMs)
      : null;

    if (r.static_map == null && state.map != null) r.static_map = state.map;
    if (r.xlen == null && state.xlen != null) r.xlen = state.xlen;
    if (r.ylen == null && state.ylen != null) r.ylen = state.ylen;

    if (typeof serverData.dishes_served === 'number') {
      r.summary.dishes_served = serverData.dishes_served;
    }

    const humanAction = this._normalizeHumanAction(humanKey);
    if (humanAction != null && humanAction !== 'STAY') {
      r.summary.human_steps += 1;
    }

    const teamAdj = (typeof serverData.team_reward_adjusted === 'number')
      ? serverData.team_reward_adjusted
      : ((typeof serverData.adjusted_reward === 'number') ? serverData.adjusted_reward : null);

    const huAdj = (typeof serverData.human_reward_adjusted === 'number')
      ? serverData.human_reward_adjusted
      : null;

    const aiAdj = (typeof serverData.ai_reward_adjusted === 'number')
      ? serverData.ai_reward_adjusted
      : null;

    if (typeof r.summary.team_reward_score !== 'number') r.summary.team_reward_score = 0;
    if (typeof r.summary.human_reward_score !== 'number') r.summary.human_reward_score = 0;
    if (typeof r.summary.ai_reward_score !== 'number') r.summary.ai_reward_score = 0;

    if (teamAdj != null && Number.isFinite(teamAdj)) r.summary.team_reward_score += teamAdj;
    if (huAdj != null && Number.isFinite(huAdj)) r.summary.human_reward_score += huAdj;
    if (aiAdj != null && Number.isFinite(aiAdj)) r.summary.ai_reward_score += aiAdj;

    const aiLast = serverData.robot_last_action || null;
    const aiLow = (aiLast && aiLast.low_level_action != null) ? Number(aiLast.low_level_action) : null;
    const aiMacro = (aiLast && aiLast.ai_macro_action != null) ? Number(aiLast.ai_macro_action) : null;
    const aiArrow = this._aiArrowFromLowLevel(aiLow);

    if (aiArrow != null && aiArrow !== 'STAY') {
      r.summary.ai_steps += 1;
    }

    r.action_log.human.push({
      t,
      ...(this.OPTS.LOG_WALL_MS ? { wall_ms } : {}),
      key: humanKey ?? null,
      action: this._normalizeHumanAction(humanKey),
      pos: (human.x != null && human.y != null) ? [human.x, human.y] : null,
      holding: this._packHolding(human)
    });

    r.action_log.ai.push({
      t,
      ...(this.OPTS.LOG_WALL_MS ? { wall_ms } : {}),
      macro: aiMacro,
      low: aiLow,
      arrow: aiArrow,
      pos: (ai.x != null && ai.y != null) ? [ai.x, ai.y] : null,
      holding: this._packHolding(ai)
    });

    if (this.OPTS.LOG_COUNTERS_EACH_TICK) {
      r.counter_log.push({
        t,
        ...(this.OPTS.LOG_WALL_MS ? { wall_ms } : {}),
        dishes_served: (typeof r.summary.dishes_served === 'number') ? r.summary.dishes_served : null,
        human_steps: (typeof r.summary.human_steps === 'number') ? r.summary.human_steps : null,
        ai_steps: (typeof r.summary.ai_steps === 'number') ? r.summary.ai_steps : null,
        team_reward_score: (typeof r.summary.team_reward_score === 'number') ? r.summary.team_reward_score : null,
        human_reward_score: (typeof r.summary.human_reward_score === 'number') ? r.summary.human_reward_score : null,
        ai_reward_score: (typeof r.summary.ai_reward_score === 'number') ? r.summary.ai_reward_score : null
      });
    }

    if (this.OPTS.LOG_STATE_EACH_TICK) {
      r.state_log.push({
        t,
        ...(this.OPTS.LOG_WALL_MS ? { wall_ms } : {}),
        state: this._compactState(state)
      });
    }
  },

  saveEpisodeSurvey(episode_index, episode_phase, answers) {
    const ep = this._ensureEpisode(episode_index, episode_phase);
    if (!ep) return;

    if (answers?.mental_demand != null) ep.feedback.mental_demand = answers.mental_demand;
    if (answers?.performance != null) ep.feedback.performance = answers.performance;
    ep.feedback.submittedAtISO = new Date().toISOString();
  },

  buildPayload() {
    const payload = JSON.parse(JSON.stringify(this.LOGS));

    for (const r of payload.rounds) delete r._roundStartWallMs;

    for (const ep of payload.episodes) {
      if (Array.isArray(ep.rounds)) ep.rounds = [];
    }

    return payload;
  },

  _hasActiveMainTaskRound() {
    return Boolean(
      STATE.phase === 1 &&
      STATE.isPlaying &&
      !STATE.gameOver &&
      STATE.episodeIndex != null &&
      STATE.roundInEpisode != null &&
      Array.isArray(this.LOGS.rounds) &&
      this.LOGS.rounds.length > 0
    );
  },

  _buildResumeMeta() {
    const canResume = this._hasActiveMainTaskRound();

    return {
      sessionId: STATE.sessionId ?? null,
      prolificId: STATE.prolificId ?? null,
      phase: STATE.phase ?? null,
      configId: STATE.configId ?? null,
      episodeIndex: STATE.episodeIndex ?? null,
      roundInEpisode: STATE.roundInEpisode ?? null,
      episodePhase: STATE.episodePhase ?? null,
      experimentPhase: STATE.experimentPhase ?? null,
      layout: STATE.assignment?.layout ?? null,
      condition: STATE.assignment?.condition ?? null,
      timeLeft: (typeof timeLeft === 'number') ? timeLeft : null,
      isPlaying: !!STATE.isPlaying,
      gameOver: !!STATE.gameOver,
      canResume
    };
  },

  readLocalBackup(prolificId = null) {
    try {
      const pid = prolificId || this.LOGS.prolificId || STATE.prolificId || 'unknown';
      const raw = localStorage.getItem(`overcooked_progress_${pid}`);
      return raw ? JSON.parse(raw) : null;
    } catch (err) {
      console.warn("Could not read local backup:", err);
      return null;
    }
  },

  restoreLogs(logPayload) {
    if (!logPayload || typeof logPayload !== 'object') return;
    this.LOGS = JSON.parse(JSON.stringify(logPayload));
  },

  persistLocalBackup(reason = 'autosave') {
    try {
      const pid = this.LOGS.prolificId || STATE.prolificId || 'unknown';
      const key = `overcooked_progress_${pid}`;
      const resumeMeta = this._buildResumeMeta();

      try { localStorage.setItem('last_prolific_id', pid); } catch (err) {}

      if (!resumeMeta.canResume) {
        localStorage.removeItem(key);
        return;
      }

      const compactLog = JSON.parse(JSON.stringify(this.LOGS));

      compactLog.meta = compactLog.meta || {};
      compactLog.meta.resume_meta = resumeMeta;

      const currentRoundIdx = (compactLog.rounds?.length || 0) - 1;

      compactLog.rounds = (compactLog.rounds || []).map((r, idx) => {
        const round = { ...r };
        delete round._roundStartWallMs;
        round.state_log = [];

        if (idx !== currentRoundIdx) {
          round.action_log = { human: [], ai: [] };
          round.counter_log = [];
          delete round.initial_state;
        }

        return round;
      });

      compactLog.episodes = (compactLog.episodes || []).map(ep => ({
        ...ep,
        rounds: []
      }));

      localStorage.setItem(
        key,
        JSON.stringify({
          savedAtISO: new Date().toISOString(),
          reason,
          sessionId: STATE.sessionId ?? null,
          resume_meta: resumeMeta,
          log: compactLog
        })
      );
    } catch (err) {
      console.warn("Could not store local backup:", err);
    }
  },

  async saveProgressToServer(reason = 'autosave') {
    const payload = this.buildPayload();
    payload.meta = payload.meta || {};
    payload.meta.resume_meta = this._buildResumeMeta();

    this.persistLocalBackup(reason);

    try {
      const res = await fetch(`${SERVER_URL}/save_progress`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          log: payload,
          reason
        })
      });

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        console.warn("Autosave HTTP error:", res.status, text);
      }
    } catch (err) {
      console.warn("Autosave failed:", err);
    }
  },

  sendBeaconProgress(reason = 'pagehide') {
    try {
      const payload = this.buildPayload();
      payload.meta = payload.meta || {};
      payload.meta.resume_meta = this._buildResumeMeta();

      this.persistLocalBackup(reason);

      const blob = new Blob(
        [JSON.stringify({ log: payload, reason })],
        { type: 'application/json' }
      );

      navigator.sendBeacon(`${SERVER_URL}/save_progress`, blob);
    } catch (err) {
      console.warn("sendBeacon progress save failed:", err);
    }
  },

  async submitToServer() {
    const payload = this.buildPayload();

    const res = await fetch(`${SERVER_URL}/submit_log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ log: payload })
    });

    await fetch(`${SERVER_URL}/close_optimizer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prolificId: STATE.prolificId })
    });

    try {
      const pid = this.LOGS.prolificId || 'unknown';
      localStorage.removeItem(`overcooked_progress_${pid}`);
    } catch (err) {
      console.warn("Could not remove local backup:", err);
    }

    return await res.json();
  }
};