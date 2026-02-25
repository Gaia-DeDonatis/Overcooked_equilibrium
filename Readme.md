# Overcooked Human–AI Equilibrium Study (Web App)

**Main components**
- **Backend (Python/Flask)**: hosts the Overcooked environment, selects and runs an AI policy, and exposes HTTP endpoints for stepping the game and saving logs.
- **Frontend (HTML + vanilla JS)**: shows consent/instructions/game pages, captures participant input, renders the grid, and logs all steps + survey answers to JSON.

---

## Repository structure

> The backend serves the frontend directly, so the expected structure is:

```
.
├── backend_new.py
├── policy_pool/
│   ├── <policy_name_1>/
│   │   └── model_500000.zip
│   ├── <policy_name_2>/
│   │   └── model_500000.zip
│   └── ...
├── submissions/                  # created automatically on first submission
├── Equilibrium_project/
│   ├── equilibrium_frontend.html
│   └── static/
│       ├── js/
│       │   ├── config.js
│       │   ├── controller.js
│       │   ├── practice.js
│       │   ├── view.js
│       │   └── dataManager.js
│       └── images/
│           ├── space.png
│           ├── counter.png
│           ├── agent-robot.png
│           └── ... (sprites used by view.js)
└── Overcooked - Experiment draft.pdf
```

### What each file does

#### Backend
- **`backend_new.py`**
  - Flask server + API endpoints (`/new_session`, `/reset`, `/key_event`, `/get_state`, `/submit_log`)
  - Session management (multiple concurrent participants; sessions expire after inactivity)
  - Environment creation for practice vs main experiment
  - Random policy selection from `policy_pool/`
  - Step logic: applies human key + AI action, updates score, returns state to frontend
  - Saves submitted logs to `submissions/*.json`

#### Frontend
- **`Equilibrium_project/equilibrium_frontend.html`**
  - Single-page HTML that contains all experiment “screens” (intro, consent, instructions, practice, gameplay, episode breaks, end)
  - Loads the JS modules below

- **`static/js/config.js`**
  - Experiment parameters (round length, #episodes, break duration, etc.)
  - Global `STATE` object used across scripts
  - `SERVER_URL` (important when hosting remotely)

- **`static/js/controller.js`**
  - Main controller:
    - creates session (`/new_session`)
    - starts practice and main rounds (`/reset`)
    - fixed-rate game stepping (`/key_event`)
    - page navigation + timers
    - comprehension checks / gating logic
    - triggers final submission (via `DataManager.submitToServer()`)

- **`static/js/practice.js`**
  - Practice-only helper: triggers `/reset` in practice mode and updates practice UI gating

- **`static/js/view.js`**
  - Rendering: preloads sprites and draws the grid/items/agents to a `<canvas>`
  - Tile ID → sprite mapping lives here (`TILE_MAP`)

- **`static/js/dataManager.js`**
  - Data logging format and helper functions
  - Stores participant metadata, episodes, rounds, per-tick actions/positions
  - Submits the final JSON to `/submit_log`

---

## How to run locally

### 1) Python environment
Create/activate a Python environment (recommended), then install dependencies. At minimum you need:
- `flask`, `flask-cors`
- `gym`
- `numpy`
- `torch`
- `stable-baselines3`
- `gym_macro_overcooked` **must be importable** (either installed or available inside `Equilibrium_project/`)

> The backend prepends `./Equilibrium_project` to `sys.path` so local packages inside that folder can be imported.

### 2) Add AI policies
The backend expects a directory named `policy_pool/` in the repo root. Each policy must be in its own subfolder and contain:

```
policy_pool/<policy_name>/model_500000.zip
```

During the main experiment, the backend will, for the moment, **randomly choose** a policy folder at the start of each episode.

### 3) Start the server
From the repo root:

```bash
python backend_new.py
```

Then open in a browser:

- `http://localhost:5000/`

---

## Experiment flow (what participants experience)

The HTML contains multiple screens (implemented as `<section id="page-...">` blocks). The controller shows/hides them in order:

1. **Intro / demographics** (Prolific ID, age, etc.)
2. **Consent**
3. **Instructions** (multiple instruction screens)
4. **Practice**  
   - backend config_id = `layout_practice`
   - AI stays still
   - participant must reach `CONFIG.PRACTICE_SCORE` to proceed
5. **Main task** (episodes × rounds)  
   - Each episode has `CONFIG.ROUNDS_PER_EPISODE` rounds  
   - Between episodes: timed break screen (`CONFIG.EPISODE_BREAK_SEC`)
6. **End / completion code**

---

## Configuration knobs

### Frontend (`static/js/config.js`)
- `SERVER_URL`  
  Change this when hosting on a different domain/port.

- `ROUND_DURATION_SEC`  
  Round duration (currently set to 10 for testing; comment indicates 45 for production).

- Episode structure:
  - `ROUNDS_PER_EPISODE`
  - `EPISODES_SEED`, `EPISODES_BO`, `EPISODES_STRESS`
  - `EPISODE_BREAK_SEC`

`CONFIG.TOTAL_EPISODES` is computed automatically.

### Backend (`backend_new.py`)
- `MAX_STEPS`  
  Maximum environment steps per round (the frontend also ends rounds by timer).

- Fixed main-task map:
  - `FIXED_MAP_TYPE` (currently `"cramped"`)
  - `FIXED_GRID_DIM` (currently `[5, 5]`)

- `rewardList`  
  Reward shaping / step penalty configuration used when creating envs.

---

## Backend API (for debugging + extensions)

All endpoints accept/return JSON.

### `POST /new_session`
Creates a new session.
- Response: `{ success: true, session_id: "<hex>" }`

### `POST /reset`
Creates a fresh environment for the session (practice or main task). Also used at episode boundaries.
- Request fields used by frontend:
  - `config_id`: `"layout_practice"` for practice; otherwise main task
  - `episode_index`, `round_in_episode`, `episode_phase`, `new_episode` (metadata; used to decide when to pick a new AI policy)

- Response includes:
  - `state`: full game state snapshot (map, items, agents, etc.)
  - `cumulative_reward`
  - `policy_id` / `chosen_policy_dir` / `chosen_ckpt`

### `POST /key_event`
Steps the environment once.
- Request: `{ key: "ArrowUp" | "ArrowDown" | "ArrowLeft" | "ArrowRight" | "Stay" }`
- Backend combines:
  - human key → agent 1 action
  - AI model prediction → macro action → low-level action for agent 0
- Response includes:
  - `state` (for rendering)
  - `cumulative_reward`
  - `dishes_served` (per-round counter)
  - `robot_last_action` (AI macro + low-level action)

### `GET` or `POST /get_state`
Returns the most recent state snapshot for a session.

### `POST /submit_log`
Receives the final experiment log and saves it to disk.
- Request: `{ log: <DataManager.LOGS> }`
- Saves to: `submissions/<prolificId>_<completion_code>.json`
- Returns: `{ success: true, completion_code: "..." }`

---

## Data logging format

The frontend builds a single JSON object `DataManager.LOGS` and POSTs it to `/submit_log`.

High-level shape:

```json
{
  "prolificId": "xxxx",
  "meta": {
    "prolificId": "xxxx",
    "age": "xx",
    "gender": "...",
    "experience": "...",
    "assignment": { "condition": null, "map": null },
    "startTimeISO": "..."
  },
  "episodes": [
    {
      "episode_index": 1,
      "episode_phase": "seed" | "bo" | "stress",
      "experiment_phase": 1 | 2 | null,
      "policy_id": "policy_folder_name",
      "optimal_policy_id": null,
      "feedback": {
        "mental_effort": null,
        "coordination_quality": null,
        "submittedAtISO": null
      },
      "rounds": [ /* same objects as in rounds[] */ ]
    }
  ],
  "rounds": [
    {
      "round_index_global": 1,
      "phase": 0 | 1,
      "episode_index": 1,
      "round_in_episode": 1,
      "policy_id": "...",
      "map": "...",
      "startTimeISO": "...",
      "endTimeISO": "...",
      "summary": {
        "final_score": 0,
        "dishes_served": 0,
        "human_steps": 0
      },
      "action_log": {
        "human": [
          { "t": 0, "wall_ms": 0, "action": "Up|Down|Left|Right|Stay", "pos": [x,y], "holding": {...} }
        ],
        "ai": [
          { "t": 0, "wall_ms": 0, "action": "Up|Down|Left|Right|Stay", "pos": [x,y], "holding": {...} }
        ]
      },
      "events": []
    }
  ]
}
```

Notes:
- `t` is the environment step index; `wall_ms` is elapsed real time since round start.
- AI and human positions come from backend `state.agents`, where `[0]=AI` and `[1]=Human`.
- `holding` is a compact representation of the item the agent is carrying (if any).

