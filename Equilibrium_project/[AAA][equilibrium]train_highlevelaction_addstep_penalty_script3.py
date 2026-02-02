import gym
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import matplotlib.pyplot as plt
from gym_macro_overcooked.items import Tomato, Lettuce, Onion, Plate, Knife, Delivery, Agent, Food, DirtyPlate
import random
import time
import torch
import os



# ====== 全局随机种子 ======
SEED = 42  # 你可以修改这个数字来改变随机性

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class SingleAgentWrapper(gym.Wrapper):
    """
    A wrapper to extract a single agent's perspective from a multi-agent environment.
    """
    def __init__(self, env, agent_index, step_penalty_agent0, step_penalty_agent1, other_agent_model=None):
        super(SingleAgentWrapper, self).__init__(env)
        self.agent_index = agent_index
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.other_agent_model = other_agent_model

        self.step_penalty_agent0 = step_penalty_agent0
        self.step_penalty_agent1 = step_penalty_agent1

        self.obs = None

    def reset(self):
        self.obs = self.env.reset()
        return self.obs[self.agent_index]

    def step(self, action):


        # 兼容 env 里可能有 self.agent 或 self.agents 两种命名
        if hasattr(self.env, "agent"):
            agents = self.env.agent
        elif hasattr(self.env, "agents"):
            agents = self.env.agents
        else:
            # 如果你的 env 把 agent 存在别的字段里，请在这里改
            agents = self.env.unwrapped.agent

        agent0_previous_location = [agents[0].x, agents[0].y]
        agent1_previous_location = [agents[1].x, agents[1].y]

        actions = [0, 0]

        # 另一个agent用其模型做动作（若未提供，则默认0）
        if self.other_agent_model is None:
            other_agent_action = (np.array([0]), None)
        else:
            other_agent_action = self.other_agent_model.predict(self.obs[1 - self.agent_index])

        actions[self.agent_index] = int(action)
        actions[1 - self.agent_index] = int(other_agent_action[0])

        # 宏动作 -> 底层动作
        primary_actions, _ = self.env._computeLowLevelActions(actions)

        self.obs, rewards, dones, info = self.env.step(primary_actions)
        self.obs = self.env._get_macro_obs()

        agent0_current_location = [agents[0].x, agents[0].y]
        agent1_current_location = [agents[1].x, agents[1].y]

        # ====== 这里把 step_penalty 加进去（每一步都扣）======
        # 你要的是 0/1/2/3/4 五档；这里按“额外step penalty”处理：reward -= step_penalty
        step_penalty = float(self.step_penalty_agent0 if self.agent_index == 0 else self.step_penalty_agent1)

        # ====== 保留你原来的“移动惩罚”逻辑 ======
        if self.agent_index == 0:
            if agent0_previous_location != agent0_current_location:
                total_reward = float(rewards[0] + rewards[1]) - step_penalty
            else:
                total_reward = float(rewards[0] + rewards[1])
            return self.obs[self.agent_index], total_reward, dones, info

        if self.agent_index == 1:
            if agent1_previous_location != agent1_current_location:
                total_reward = float(rewards[0] + rewards[1]) - step_penalty
            else:
                total_reward = float(rewards[0] + rewards[1])
            return self.obs[self.agent_index], total_reward, dones, info

        # # 不会到这里
        # return self.obs[self.agent_index], float(rewards[0] + rewards[1]) - step_penalty, dones, info


class EpisodeRewardCallback(BaseCallback):
    def __init__(self, save_path, save_freq=100000, verbose=0):
        super(EpisodeRewardCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.step_counter = 0
        self.episode_rewards = []
        self.current_episode_reward = 0.0

        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        self.step_counter += 1
        self.current_episode_reward += float(self.locals["rewards"][0])

        # Check if episode is done
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0

        # Save and plot every save_freq steps
        if self.step_counter % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"model_{self.step_counter}.zip")
            self.model.save(model_path)
            print(f"Step {self.step_counter}: Model saved at {model_path}")

            # Plot moving average of last 100 episode rewards
            if len(self.episode_rewards) >= 1:
                window = min(100, len(self.episode_rewards))
                moving_avg = [
                    sum(self.episode_rewards[max(0, i - window + 1): i + 1]) /
                    (i - max(0, i - window + 1) + 1)
                    for i in range(len(self.episode_rewards))
                ]

                plt.figure(figsize=(10, 5))
                plt.plot(moving_avg, label=f"Moving Avg (last {window} episodes)")
                plt.xlabel("Episode")
                plt.ylabel("Average Reward")
                plt.title("Training Progress")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_path, f"avg_reward_{self.step_counter}.png"))
                plt.close()

        return True


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}分{secs}秒"


def train_one_combo(step_penalty_agent0: int, step_penalty_agent1: int):
    # ====== 你的 rewardList / env_params ======
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
        "step penalty": -1,  # 这里保留你原来的；wrapper里会再额外扣 step_penalty_agent0/1
        "penalize using dirty plate": 0,
        "penalize using bad lettuce": 0,
        "pick up bad lettuce": 0
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

    mac_env_id = "Overcooked-MA-equilibrium-v0"
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

    # ====== 为每个组合单独建 log/save 目录，避免覆盖 ======
    combo_tag = f"a0sp_{step_penalty_agent0}_a1sp_{step_penalty_agent1}"

    log_dir = os.path.join("logs", combo_tag)
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["csv", "tensorboard"])  # 不输出 stdout

    save_dir_agent0 = os.path.join("final_trained_models", f"[equilibrium]agent0_{combo_tag}")
    save_dir_agent1 = os.path.join("final_trained_models", f"[equilibrium]agent1_{combo_tag}")

    reward_callback_0 = EpisodeRewardCallback(save_dir_agent0)
    reward_callback_1 = EpisodeRewardCallback(save_dir_agent1)

    # ====== Initialize shared environment ======
    shared_env = gym.make(mac_env_id, **env_params)
    shared_env.seed(SEED)
    shared_env.action_space.seed(SEED)
    shared_env.observation_space.seed(SEED)

    # 先用空 partner 占位创建 wrapper（后面每轮会 set_env 替换）
    env_agent_0 = SingleAgentWrapper(
        shared_env,
        agent_index=0,
        step_penalty_agent0=step_penalty_agent0,
        step_penalty_agent1=step_penalty_agent1,
        other_agent_model=None
    )
    env_agent_1 = SingleAgentWrapper(
        shared_env,
        agent_index=1,
        step_penalty_agent0=step_penalty_agent0,
        step_penalty_agent1=step_penalty_agent1,
        other_agent_model=None
    )

    # ====== PPO params ======
    ppo_params = {
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.95,
        "gae_lambda": 0.95,
        "clip_range": 0.3,
        "ent_coef": 0.02,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 0,
    }

    policy_kwargs = dict(net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])])

    # ====== Models (每个组合都从头训练一套 agent0 + agent1) ======
    model_agent_0 = PPO(
        "MlpPolicy",
        env_agent_0,
        verbose=1,
        policy_kwargs=policy_kwargs,
        seed=SEED
    )

    model_agent_1 = PPO(
        "MlpPolicy",
        env_agent_1,
        verbose=1,
        policy_kwargs=policy_kwargs,
        seed=SEED
    )

    model_agent_0.set_logger(new_logger)
    model_agent_1.set_logger(new_logger)

    # ====== Training configuration ======
    total_alternate_steps = 500_000
    alternate_interval = 10_000

    global_start_time = time.time()
    print(f"\n==================== Training combo: {combo_tag} ====================")

    # Alternate training loop
    for i in range(0, total_alternate_steps, alternate_interval):
        print(f"Training Agent 0 (Steps {i} to {i + alternate_interval})")
        env_agent_0 = SingleAgentWrapper(
            shared_env,
            agent_index=0,
            step_penalty_agent0=step_penalty_agent0,
            step_penalty_agent1=step_penalty_agent1,
            other_agent_model=model_agent_1
        )
        model_agent_0.set_env(env_agent_0)
        model_agent_0.learn(total_timesteps=alternate_interval, callback=reward_callback_0)

        print(f"Training Agent 1 (Steps {i} to {i + alternate_interval})")
        env_agent_1 = SingleAgentWrapper(
            shared_env,
            agent_index=1,
            step_penalty_agent0=step_penalty_agent0,
            step_penalty_agent1=step_penalty_agent1,
            other_agent_model=model_agent_0
        )
        model_agent_1.set_env(env_agent_1)
        model_agent_1.learn(total_timesteps=alternate_interval, callback=reward_callback_1)

        phase_end_time = time.time()
        total_duration = phase_end_time - global_start_time
        print(f"[🕒 累计训练时间] {format_time(total_duration)}")

    # # 最后再存一个最终模型（可选）
    # final_model0 = os.path.join(save_dir_agent0, "final_model.zip")
    # final_model1 = os.path.join(save_dir_agent1, "final_model.zip")
    # model_agent_0.save(final_model0)
    # model_agent_1.save(final_model1)
    # print(f"[✅ DONE] Saved final models:\n  - {final_model0}\n  - {final_model1}\n")


def main():
    # ====== 批处理：25 个组合，每个组合训练 agent0 + agent1 两个模型 => 2*25 个模型 ======
    step_penalty_list_agent0 = [20]
    step_penalty_list_agent1 = [0, 1, 10, 20, 50]

    for sp0 in step_penalty_list_agent0:
        for sp1 in step_penalty_list_agent1:
            if sp0 == 0 and sp1 == 0:
                continue
            train_one_combo(step_penalty_agent0=sp0, step_penalty_agent1=sp1)


if __name__ == "__main__":
    main()
