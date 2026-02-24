// static/js/dataManager.js

const DataManager = {
  // Top-level payload (backend expects `rounds` and optionally `prolificId` at top-level)
  LOGS: {
    prolificId: 'unknown',
    age: null,
    gender: null,
    experience: null,

    assignment: {
      condition: null,
      layout: null
    },

    meta: {
      version: 'dm_v2',
      startTimeISO: null,
      tick_ms: null,

      // experiment parameters (optional; set by controller if you want)
      rounds_per_episode: null,
      round_duration_sec: null,
      episode_break_sec: null,

      // optionally record reward scheme id/name here
      reward_scheme: null
    },

    // One entry per episode: policy used + survey after episode
    episodes: [],

    // One entry per round (ordered in time)
    rounds: []
  },

  // ----------------------
  // 1) Participant/session
  // ----------------------
  initUser(prolificId, age, gender, assigned, extraMeta = {}) {
    this.LOGS.prolificId = prolificId || 'unknown';
    this.LOGS.age = (age != null) ? age : null;
    this.LOGS.gender = (gender != null) ? gender : null;
    this.LOGS.experience = extraMeta.experience ?? null;
    this.LOGS.meta.startTimeISO = new Date().toISOString();

    if (typeof assigned === 'string') {
      this.LOGS.assignment.condition = assigned;
    } else if (assigned && typeof assigned === 'object') {
      this.LOGS.assignment = {
        ...this.LOGS.assignment,
        ...assigned
      };
    }

    if (extraMeta.tick_ms != null) this.LOGS.meta.tick_ms = extraMeta.tick_ms;
    if (extraMeta.rounds_per_episode != null) this.LOGS.meta.rounds_per_episode = extraMeta.rounds_per_episode;
    if (extraMeta.round_duration_sec != null) this.LOGS.meta.round_duration_sec = extraMeta.round_duration_sec;
    if (extraMeta.episode_break_sec != null) this.LOGS.meta.episode_break_sec = extraMeta.episode_break_sec;
    if (extraMeta.reward_scheme != null) this.LOGS.meta.reward_scheme = extraMeta.reward_scheme;
  },

  // ----------------------
  // 2) Episode + round
  // ----------------------
  _ensureEpisode(episode_index, episode_phase) {
    if (episode_index == null) return null;
    const idx = this.LOGS.episodes.findIndex(e => e.episode_index === episode_index);
    if (idx >= 0) {
      if (episode_phase != null) this.LOGS.episodes[idx].episode_phase = episode_phase;
      return this.LOGS.episodes[idx];
    }

    const ep = {
      episode_index,
      episode_phase: episode_phase ?? null,
      policy_id: null,
      mapTopology: null,
      startTimeISO: new Date().toISOString(),
      survey: null
    };
    this.LOGS.episodes.push(ep);
    return ep;
  },

  startNewRound(phase, configId, extraMeta = {}) {
    // extraMeta should contain: episode_index, round_in_episode, episode_phase, policyId, mapTopology
    const episode_index = extraMeta.episode_index ?? null;
    const episode_phase = extraMeta.episode_phase ?? null;
    const round_in_episode = extraMeta.round_in_episode ?? null;

    const ep = this._ensureEpisode(episode_index, episode_phase);
    if (ep) {
      if (extraMeta.policyId != null) ep.policy_id = extraMeta.policyId;
      if (extraMeta.mapTopology != null) ep.mapTopology = extraMeta.mapTopology;
    }

    const round = {
      // identifiers
      phase: phase ?? null,
      configId: configId ?? null,
      round_index_global: this.LOGS.rounds.length + 1,

      // episode structure
      episode_index,
      episode_phase,
      round_in_episode,

      // environment / policy info
      mapTopology: extraMeta.mapTopology ?? null,
      policy_id: extraMeta.policyId ?? null,

      startTimeISO: new Date().toISOString(),
      endTimeISO: null,

      // store the map/items ONCE (we fill this lazily on first tick)
      world: {
        map: null,
        items_initial: null
      },

      // Shared per-tick signals (reward, counters)
      rewards: [],

      // Separate agent streams
      agents: {
        human: {
          ticks: []
        },
        ai: {
          ticks: []
        }
      },

      // round summary (PER ROUND)
      summary: {
        final_score: 0,
        dishes_served: 0,
        human_steps: 0
      }
    };

    this.LOGS.rounds.push(round);
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
  // 3) Per-tick logging
  // ----------------------
  _packHolding(agent) {
    if (!agent) return null;
    const h = agent.holding;
    if (h == null) return null;

    if (typeof h === 'string') {
      return {
        item: h,
        containing: agent.holding_containing ?? null
      };
    }
    return h;
  },

  //Log ONE applied environment step (one tick).
  logStep(serverData, humanKey, timing = {}) {
    const r = this.getCurrentRound();
    if (!r || !serverData) return;

    const state = serverData.state || {};
    const agents = Array.isArray(state.agents) ? state.agents : [];

    // backend indexing: [0]=AI, [1]=Human
    const ai = agents[0] || {};
    const human = agents[1] || {};

    const t = (typeof state.cur_step === 'number') ? state.cur_step : null;

    // Capture the static world (map + initial items)
    if (!r.world.map && state.map) {
      r.world.map = state.map;
    }
    if (!r.world.items_initial && Array.isArray(state.items)) {
      r.world.items_initial = state.items;
    }

    // Update per-round summary directly from backend counters (per-round in backend)
    if (typeof serverData.cumulative_reward === 'number') r.summary.final_score = serverData.cumulative_reward;
    if (typeof serverData.dishes_served === 'number') r.summary.dishes_served = serverData.dishes_served;
    if (humanKey !== 'Stay') r.summary.human_steps += 1;

    // Human stream
    const humanTick = {
      t,
      action: humanKey,
      pos: (human.x != null && human.y != null) ? [human.x, human.y] : null,
      holding: this._packHolding(human)
    };
    if (this.options.captureTiming) {
      humanTick.human_press_ms = (typeof timing?.humanPressMs === 'number') ? timing.humanPressMs : null;
    }
    r.agents.human.ticks.push(humanTick);

    // AI stream
    const last = serverData.robot_last_action || {};
    const aiTick = {
      t,
      low_level_action: (last.low_level_action != null) ? last.low_level_action : null,
      macro_action: (last.ai_macro_action != null) ? last.ai_macro_action : null,
      arrow: last.arrow ?? null,
      pos: (ai.x != null && ai.y != null) ? [ai.x, ai.y] : null,
      holding: this._packHolding(ai)
    };
    r.agents.ai.ticks.push(aiTick);

    // Shared reward stream
    const rewardTick = {
      t,
      raw_reward: (typeof serverData.raw_reward === 'number') ? serverData.raw_reward : null,
      adjusted_reward: (typeof serverData.adjusted_reward === 'number') ? serverData.adjusted_reward : null,
      cumulative_reward: (typeof serverData.cumulative_reward === 'number') ? serverData.cumulative_reward : null,
      dishes_served: (typeof serverData.dishes_served === 'number') ? serverData.dishes_served : null,
      steps_left: (typeof serverData.steps_left === 'number') ? serverData.steps_left : null
    };
    if (this.options.captureTiming) {
      rewardTick.applied_ms = (typeof timing?.appliedMs === 'number') ? timing.appliedMs : null;
    }
    r.rewards.push(rewardTick);
  },

  // ----------------------
  // 4) Episode break survey
  // ----------------------
  saveEpisodeSurvey(episode_index, episode_phase, answers) {
    const ep = this._ensureEpisode(episode_index, episode_phase);
    if (!ep) return;
    ep.survey = {
      mental_effort: (answers?.mental_effort != null) ? answers.mental_effort : null,
      coordination_quality: (answers?.coordination_quality != null) ? answers.coordination_quality : null,
      submittedAtISO: new Date().toISOString()
    };
  },

  // ----------------------
  // 5) Submission
  // ----------------------
  async submitToServer() {
    console.log('Submitting Data...', this.LOGS);
    return await fetch(`${SERVER_URL}/submit_log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ log: this.LOGS })
    }).then(res => res.json());
  }
};
