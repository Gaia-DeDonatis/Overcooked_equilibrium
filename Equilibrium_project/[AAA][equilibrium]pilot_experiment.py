import pygame
import gym
import warnings
import sys
import numpy as np
import time
import os
import json
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO

# Optional video dependency:
#   pip install imageio imageio-ffmpeg
try:
    import imageio.v2 as imageio
except Exception:
    try:
        import imageio  # type: ignore
    except Exception:
        imageio = None

warnings.filterwarnings("ignore")

# --- COMPATIBILITY PATCH (kept from your pilot_experiment style) ---
try:
    sys.modules["numpy._core"] = np.core
    if hasattr(np.core, "numeric"):
        sys.modules["numpy._core.numeric"] = np.core.numeric
    if hasattr(np.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    import numpy.random._pickle  # type: ignore

    original_ctor = np.random._pickle.__bit_generator_ctor  # type: ignore

    def patched_ctor(bit_generator_name):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        return original_ctor(bit_generator_name)

    np.random._pickle.__bit_generator_ctor = patched_ctor  # type: ignore
except Exception:
    pass


# =========================
# SETTINGS (3 rounds x 45s)
# =========================
TOTAL_ROUNDS = 3
ROUND_DURATION = 45  # seconds
SECONDS_PER_STEP = 0.25  # 250ms tick (AI independent)
FPS = max(1, int(round(1.0 / SECONDS_PER_STEP)))

SAVE_VIDEO = True
VIDEO_DIR = "recordings"
LOG_DIR = "logs"
VIDEO_FORMAT = "mp4"  # "mp4" (needs ffmpeg) or fallback to "gif" if mp4 fails
SAVE_OBS_IN_CSV = True  # store obs_1/obs_2 as JSON strings (can be large)


# Primitive action mapping (same as your play_with_trained_agent)
# 0: right, 1: down, 2: left, 3: up, 4: still
KEYS_ACTIONS = {
    pygame.K_UP: 3,
    pygame.K_RIGHT: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
}
AGENT_KEYS = {pygame.K_1: 0, pygame.K_2: 1}
ACTION_LABELS = {0: "RIGHT", 1: "DOWN", 2: "LEFT", 3: "UP", 4: "STAY"}


def _jsonable(x):
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return str(x)


def safe_json_dumps(obj) -> str:
    try:
        return json.dumps(_jsonable(obj), ensure_ascii=False)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)


def get_macro_action_name(env, macro_id: int) -> str:
    # play_with_trained_agent.py used env.macroActionName[ai_action]
    try:
        names = getattr(env, "macroActionName", None)
        if names is not None and 0 <= macro_id < len(names):
            return str(names[macro_id])
    except Exception:
        pass
    return ""


class SingleAgentWrapper_accept_keyboard_action(gym.Wrapper):
    """
    Extract agent0 obs, but step with (agent0_action, agent1_action) primitive actions.
    """
    def __init__(self, env, agent_index):
        super().__init__(env)
        self.agent_index = agent_index
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.obs = None

    def reset(self):
        self.obs = self.env.reset()
        return self.obs[self.agent_index]

    def step(self, action_agent0, action_agent1):
        actions = [action_agent0, action_agent1]
        obs, rewards, dones, info = self.env.step(actions)

        # refresh macro obs (as your code did)
        self.obs = self.env._get_macro_obs()

        # play_with wrapper returned rewards[2]
        final_reward = rewards[2] if isinstance(rewards, list) and len(rewards) > 2 else rewards
        return self.obs[self.agent_index], final_reward, dones, info


def make_env():
    mac_env_id = "Overcooked-MA-equilibrium-v0"

    rewardList = [{
        "minitask finished": 0,
        "minitask failed": 0,
        "metatask finished": 0,
        "metatask failed": 0,
        "goodtask finished": 10,
        "goodtask failed": 0,
        "subtask finished": 20,
        "subtask failed": 0,
        "correct delivery": 200,
        "wrong delivery": -50,
        "step penalty": -1,
        "penalize using dirty plate": 0,
        "penalize using bad lettuce": -20,
        "pick up bad lettuce": -100
    }, {
        "minitask finished": 0,
        "minitask failed": 0,
        "metatask finished": 0,
        "metatask failed": 0,
        "goodtask finished": 10,
        "goodtask failed": 0,
        "subtask finished": 20,
        "subtask failed": 0,
        "correct delivery": 200,
        "wrong delivery": -50,
        "step penalty": -1,
        "penalize using dirty plate": 0,
        "penalize using bad lettuce": 0,
        "pick up bad lettuce": 0
    }]

    env_params = {
        "grid_dim": [5, 5],
        "task": ["lettuce salad"],
        "rewardList": rewardList,
        "map_type": "circle",
        "n_agent": 2,
        "obs_radius": 0,
        "mode": "vector",
        "debug": True
    }

    import gym_macro_overcooked  # needed for registration side effects
    base_env = gym.make(mac_env_id, **env_params)
    wrapped = SingleAgentWrapper_accept_keyboard_action(base_env, agent_index=0)
    return base_env, wrapped


def open_video_writer(path_mp4: str):
    """
    Try mp4 writer; return (writer, mode)
    mode: "mp4" if streaming writer, "frames" if fallback to frames list, "none" if not saving.
    """
    if not SAVE_VIDEO:
        return None, "none"
    if imageio is None:
        print("[Video] imageio not installed -> no video. Install: pip install imageio imageio-ffmpeg")
        return None, "none"

    # Try streaming mp4
    try:
        writer = imageio.get_writer(path_mp4, fps=FPS)
        return writer, "mp4"
    except Exception as e:
        print("[Video] Could not create mp4 writer, will fallback to GIF frames:", e)
        return None, "frames"


def save_gif_fallback(frames, gif_path: str):
    if not frames:
        return
    if imageio is None:
        return
    try:
        imageio.mimsave(gif_path, frames, fps=FPS)
        print("[GIF saved]", gif_path)
    except Exception as e:
        print("[GIF save failed]", e)


def run_one_round(round_num: int, base_env, env_agent_0, agent_model, session_ts: str):
    """
    Runs 45s round with 250ms tick, logs play_with-style data and saves video.
    Returns (df_log, video_out_path_or_none)
    """
    # Reset env for this round
    obs = env_agent_0.reset()

    # Controls
    selected_agent = 1            # human controls agent1 by default
    current_human_action = 4      # STAY unless key pressed
    step_idx = 0

    # Video
    video_out = None
    frames_fallback = []
    writer = None
    writer_mode = "none"

    if SAVE_VIDEO:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        mp4_path = os.path.join(VIDEO_DIR, f"session_{session_ts}_round_{round_num}.mp4")
        writer, writer_mode = open_video_writer(mp4_path)
        if writer_mode == "mp4":
            video_out = mp4_path
        else:
            # fallback will be gif
            video_out = os.path.join(VIDEO_DIR, f"session_{session_ts}_round_{round_num}.gif")

    # Capture initial frame
    try:
        frame0 = base_env.render(mode="rgb_array")
        if SAVE_VIDEO and frame0 is not None:
            if writer_mode == "mp4" and writer is not None:
                writer.append_data(frame0)
            elif writer_mode == "frames":
                frames_fallback.append(frame0)
        pygame.display.flip()
    except Exception:
        pass

    logs = []
    start_time = time.time()
    last_tick = time.time()

    running = True
    while running:
        elapsed = time.time() - start_time
        if elapsed >= ROUND_DURATION:
            running = False
            break

        # Input handling (does NOT step env)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return pd.DataFrame(logs), video_out, True  # aborted session
            if event.type == pygame.KEYDOWN:
                if event.key in AGENT_KEYS:
                    selected_agent = AGENT_KEYS[event.key]
                if event.key in KEYS_ACTIONS:
                    current_human_action = KEYS_ACTIONS[event.key]

        # Tick stepping
        now = time.time()
        if (now - last_tick) >= SECONDS_PER_STEP:
            last_tick = now
            step_idx += 1

            obs_before = obs

            # AI macro (policy output)
            ai_action_macro, _ = agent_model.predict(obs_before, deterministic=True)
            if isinstance(ai_action_macro, np.ndarray):
                ai_action_macro = int(ai_action_macro.item())
            else:
                ai_action_macro = int(ai_action_macro)

            ai_action_macro_name = get_macro_action_name(base_env, ai_action_macro)

            # macro -> primitive for agent0
            prim_pair, _ = base_env._computeLowLevelActions([ai_action_macro, 0])
            ai_action_primitive = int(prim_pair[0])

            # Build primitive action pair
            # If human controls agent1 (default): a0=AI, a1=human
            if selected_agent == 1:
                a0 = ai_action_primitive
                a1 = current_human_action
                human_action_used = a1
                ai_primitive_used = a0
            else:
                # If human selected agent0: human overrides agent0 primitive, agent1 stays still
                a0 = current_human_action
                a1 = 4
                human_action_used = a0
                ai_primitive_used = ""  # not applied

            # Step env ONCE
            obs_after, reward, dones, info = env_agent_0.step(a0, a1)

            logs.append({
                "round": round_num,
                "step": step_idx,
                "elapsed_sec": round(elapsed, 3),
                "selected_agent": selected_agent,

                # play_with-style "obs 1" and "obs 2"
                "obs_1": safe_json_dumps(obs_before.tolist()) if SAVE_OBS_IN_CSV else "",
                "obs_2": safe_json_dumps(obs_after.tolist()) if SAVE_OBS_IN_CSV else "",

                # AI actions (play_with prints macro name + uses primitive)
                "ai_macro_action": ai_action_macro,
                "ai_macro_name": ai_action_macro_name,
                "ai_primitive_action": ai_primitive_used,

                # Human action
                "human_action": human_action_used,
                "human_action_label": ACTION_LABELS.get(int(human_action_used), "UNKNOWN") if human_action_used != "" else "",

                # Extra debug fields that help reproduce behavior
                "reward": float(reward) if isinstance(reward, (int, float, np.number)) else str(reward),
                "done": str(dones),
                "info": safe_json_dumps(info),
            })

            obs = obs_after

            # Tap-per-tick behavior (like your pilot experiment): reset to STAY after applying
            current_human_action = 4

            # Render + video frame
            try:
                frame = base_env.render(mode="rgb_array")
                if SAVE_VIDEO and frame is not None:
                    if writer_mode == "mp4" and writer is not None:
                        writer.append_data(frame)
                    elif writer_mode == "frames":
                        frames_fallback.append(frame)
                pygame.display.flip()
            except Exception:
                pass

    # Close/save video
    if writer_mode == "mp4" and writer is not None:
        try:
            writer.close()
            print("[Video saved]", video_out)
        except Exception as e:
            print("[Video] writer.close failed:", e)

    if writer_mode == "frames" and SAVE_VIDEO:
        save_gif_fallback(frames_fallback, video_out)

    return pd.DataFrame(logs), video_out, False


def main():
    # ---- YOUR POLICY PATH ----
    model_path = r"C:\Users\dedong1\work\Overcooked_equilibrium\policy_pool\[equilibrium]agent0_a0sp_0_a1sp_3_helping_True_gamma0.95_0.9\model_500000.zip"

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    print("Model path:", model_path)
    print("Exists:", os.path.exists(model_path))

    # Pygame init
    pygame.init()
    pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Overcooked (3 rounds x 45s, tick=250ms)")

    # Env + model (load once)
    base_env, env_agent_0 = make_env()

    custom_objects = {
        "action_space": env_agent_0.action_space,
        "observation_space": env_agent_0.observation_space,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0
    }
    agent_model = PPO.load(model_path, env=env_agent_0, custom_objects=custom_objects)

    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_round_logs = []
    round_summary = []

    print(f"\n>>> START: {TOTAL_ROUNDS} rounds x {ROUND_DURATION}s  | tick={SECONDS_PER_STEP}s ({FPS} fps) <<<")

    aborted = False
    for r in range(1, TOTAL_ROUNDS + 1):
        pygame.display.set_caption(f"Overcooked Round {r}/{TOTAL_ROUNDS} (tick=250ms)")

        df_round, video_path, aborted = run_one_round(
            round_num=r,
            base_env=base_env,
            env_agent_0=env_agent_0,
            agent_model=agent_model,
            session_ts=session_ts
        )

        # Save per-round CSV
        round_log_path = os.path.join(LOG_DIR, f"play_with_style_session_{session_ts}_round_{r}.csv")
        df_round.to_csv(round_log_path, index=False)
        print("[Round log saved]", round_log_path)

        all_round_logs.append(df_round)
        round_summary.append({
            "round": r,
            "rows": int(len(df_round)),
            "video": video_path or ""
        })

        if aborted:
            print("Aborted by user (window closed).")
            break

        # small pause between rounds (optional)
        time.sleep(1.0)

    # Save combined CSV + summary
    if all_round_logs:
        df_all = pd.concat(all_round_logs, ignore_index=True)
        combined_path = os.path.join(LOG_DIR, f"play_with_style_session_{session_ts}_ALL.csv")
        df_all.to_csv(combined_path, index=False)
        print("[Combined log saved]", combined_path)

    df_sum = pd.DataFrame(round_summary)
    summary_path = os.path.join(LOG_DIR, f"play_with_style_session_{session_ts}_SUMMARY.csv")
    df_sum.to_csv(summary_path, index=False)
    print("[Summary saved]", summary_path)

    # Cleanup
    try:
        base_env.close()
    except Exception:
        pass
    pygame.quit()


if __name__ == "__main__":
    main()