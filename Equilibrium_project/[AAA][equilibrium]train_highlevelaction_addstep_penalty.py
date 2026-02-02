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
    def __init__(self, env, agent_index, other_agent_model=None):
        super(SingleAgentWrapper, self).__init__(env)
        self.agent_index = agent_index
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.other_agent_model = other_agent_model


        self.obs = None


    def reset(self):
        self.obs = self.env.reset()

        return self.obs[self.agent_index]

    def step(self, action):

        agent0_moving_reward = 0
        agent1_moving_reward = 0

        agent0_previous_location = [self.agent[0].x, self.agent[0].y]
        agent1_previous_location = [self.agent[1].x, self.agent[1].y]
        
        actions = [0, 0]

        other_agent_action = self.other_agent_model.predict(self.obs[1 - self.agent_index])

        actions[self.agent_index] = action

        actions[1 - self.agent_index] = other_agent_action[0]

        primary_actions, _ = self.env._computeLowLevelActions(actions)

        self.obs, rewards, dones, info = self.env.step(primary_actions)

        self.obs = self.env._get_macro_obs()


        agent0_current_location = [self.agent[0].x, self.agent[0].y]
        agent1_current_location = [self.agent[1].x, self.agent[1].y]



        if self.agent_index == 0:
            if agent0_previous_location != agent0_current_location:
                # print('here')
                agent0_moving_reward -= 50
            return self.obs[self.agent_index], rewards[0] + rewards[1] + agent0_moving_reward, dones, info
        

        # 营造不喜欢移动的human
        if self.agent_index == 1:
            # 判断是否发生了位置的移动，如果发生了，则多扣一点step moving penalty
            # print(human_agent_previous_location, human_agent_current_location)
            if agent1_previous_location != agent1_current_location:
                # print('here')
                agent1_moving_reward -= 50

            return self.obs[self.agent_index], rewards[0] + rewards[1] + agent1_moving_reward, dones, info





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
        self.current_episode_reward += self.locals['rewards'][0]

        # Check if episode is done
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0

        # Save and plot every save_freq steps
        if self.step_counter % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f'model_{self.step_counter}.zip')
            self.model.save(model_path)
            print(f"Step {self.step_counter}: Model saved at {model_path}")

            # Plot moving average of last 100 episode rewards
            if len(self.episode_rewards) >= 1:
                window = min(100, len(self.episode_rewards))
                moving_avg = [sum(self.episode_rewards[max(0, i - window + 1):i + 1]) / (i - max(0, i - window + 1) + 1)
                              for i in range(len(self.episode_rewards))]

                plt.figure(figsize=(10, 5))
                plt.plot(moving_avg, label=f"Moving Avg (last {window} episodes)")
                plt.xlabel("Episode")
                plt.ylabel("Average Reward")
                plt.title("Training Progress")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_path, f'avg_reward_{self.step_counter}.png'))
                plt.close()

        return True





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
    "penalize using bad lettuce": 0,
    "pick up bad lettuce": 0
},{
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


mac_env_id = 'Overcooked-MA-equilibrium-v0'
env_params = {
    'grid_dim': [5, 5],
    'task': ["lettuce salad"],
    'rewardList': rewardList,
    'map_type': "circle",
    'n_agent': 2,
    'obs_radius': 0,
    'mode': "vector",
    'debug': True
}

new_logger = configure('./logs/', ["csv", "tensorboard"])  # Remove "stdout" to prevent console logging


# Initialize shared environment
shared_env = gym.make(mac_env_id, **env_params)
shared_env.seed(SEED)
shared_env.action_space.seed(SEED)
shared_env.observation_space.seed(SEED)


# Wrap each agent
env_agent_0 = SingleAgentWrapper(shared_env, agent_index=0)
env_agent_1 = SingleAgentWrapper(shared_env, agent_index=1)

ppo_params = {
    'learning_rate': 3e-4,
    'n_steps': 256,
    'batch_size': 128,
    'n_epochs': 10,
    'gamma': 0.95,
    'gae_lambda': 0.95,
    'clip_range': 0.3,
    'ent_coef': 0.02,
    # 'ent_coef': 0.05,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'verbose': 0,
}


policy_kwargs = dict(
    net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])]
)


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


# layout_v1：物品各自在两边
reward_callback_0 = EpisodeRewardCallback('final_trained_models/[equilibrium]agent0_step_penalty_5_vs_5')
reward_callback_1 = EpisodeRewardCallback('final_trained_models/[equilibrium]agent1_step_penalty_5_vs_5')


# agent0_step_penalty_1_vs_1, agent1_step_penalty_1_vs_1
# agent0_step_penalty_1_vs_2, agent1_step_penalty_1_vs_2
# agent0_step_penalty_1_vs_3, agent1_step_penalty_1_vs_3
# agent0_step_penalty_1_vs_4, agent1_step_penalty_1_vs_4
# agent0_step_penalty_1_vs_5, agent1_step_penalty_1_vs_5


# agent0_step_penalty_2_vs_1, agent1_step_penalty_2_vs_1
# agent0_step_penalty_2_vs_2, agent1_step_penalty_2_vs_2
# agent0_step_penalty_2_vs_3, agent1_step_penalty_2_vs_3
# agent0_step_penalty_2_vs_4, agent1_step_penalty_2_vs_4
# agent0_step_penalty_2_vs_5, agent1_step_penalty_2_vs_5


# agent0_step_penalty_3_vs_1, agent1_step_penalty_3_vs_1
# agent0_step_penalty_3_vs_2, agent1_step_penalty_3_vs_2
# agent0_step_penalty_3_vs_3, agent1_step_penalty_3_vs_3
# agent0_step_penalty_3_vs_4, agent1_step_penalty_3_vs_4
# agent0_step_penalty_3_vs_5, agent1_step_penalty_3_vs_5


# agent0_step_penalty_4_vs_1, agent1_step_penalty_4_vs_1
# agent0_step_penalty_4_vs_2, agent1_step_penalty_4_vs_2
# agent0_step_penalty_4_vs_3, agent1_step_penalty_4_vs_3
# agent0_step_penalty_4_vs_4, agent1_step_penalty_4_vs_4
# agent0_step_penalty_4_vs_5, agent1_step_penalty_4_vs_5


# agent0_step_penalty_5_vs_1, agent1_step_penalty_5_vs_1
# agent0_step_penalty_5_vs_2, agent1_step_penalty_5_vs_2
# agent0_step_penalty_5_vs_3, agent1_step_penalty_5_vs_3
# agent0_step_penalty_5_vs_4, agent1_step_penalty_5_vs_4
# agent0_step_penalty_5_vs_5, agent1_step_penalty_5_vs_5





# Training configuration
total_alternate_steps = 5000000  # Total training steps
alternate_interval = 10000  # Each agent trains for this many steps before switching


global_start_time = time.time()  # 记录整个训练开始的时间

def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}分{secs}秒"




# Alternate training loop
for i in range(0, total_alternate_steps, alternate_interval):
    print(f"Training Agent 0 (Steps {i} to {i + alternate_interval})")
    env_agent_0 = SingleAgentWrapper(shared_env, agent_index=0, other_agent_model=model_agent_1)  # Agent 1 is fixed
    model_agent_0.set_env(env_agent_0)
    model_agent_0.learn(total_timesteps=alternate_interval, callback=reward_callback_0)

    print(f"Training Agent 1 (Steps {i} to {i + alternate_interval})")
    env_agent_1 = SingleAgentWrapper(shared_env, agent_index=1, other_agent_model=model_agent_0)  # Agent 0 is fixed
    model_agent_1.set_env(env_agent_1)
    model_agent_1.learn(total_timesteps=alternate_interval, callback=reward_callback_1)

    phase_end_time = time.time()
    total_duration = phase_end_time - global_start_time
    print(f"[🕒 累计训练时间] {format_time(total_duration)}")

