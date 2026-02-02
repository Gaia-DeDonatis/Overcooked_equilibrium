import gym
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import matplotlib.pyplot as plt
import time
import random
from gym_macro_overcooked.items import Tomato, Lettuce, Onion, Plate, Knife, Delivery, Agent, Food, DirtyPlate



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

        # human_agent_previous_location = [self.agent[1].x, self.agent[1].y]

        actions = [0, 0]

        other_agent_action = self.other_agent_model.predict(self.obs[1 - self.agent_index])

        actions[self.agent_index] = action

        actions[1 - self.agent_index] = other_agent_action[0]

        primary_actions, _ = self.env._computeLowLevelActions(actions)

        self.obs, rewards, dones, info = self.env.step(primary_actions)

        self.obs = self.env._get_macro_obs()

        # human_agent_current_location = [self.agent[1].x, self.agent[1].y]


        # print(human_agent_previous_location, human_agent_current_location)


        return self.obs[self.agent_index], rewards[2], dones, info




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


# Initialize shared environment
shared_env = gym.make(mac_env_id, **env_params)




# Wrap each agent
env_agent_0 = SingleAgentWrapper(shared_env, agent_index=0)
env_agent_1 = SingleAgentWrapper(shared_env, agent_index=1)


"""旧的layout且没有加moving penalty"""
# model_agent_0 = PPO.load("final_trained_models/[equilibrium]agent0_highlevelaction\model_700000", env=env_agent_0)
# model_agent_1 = PPO.load("final_trained_models/[equilibrium]agent1_highlevelaction\model_700000", env=env_agent_1)


"""新的layout但是没有加movivng penalty"""
# model_agent_0 = PPO.load("final_trained_models/[equilibrium]agent0_highlevelaction_layout_v1\model_700000", env=env_agent_0)
# model_agent_1 = PPO.load("final_trained_models/[equilibrium]agent1_highlevelaction_layout_v1\model_700000", env=env_agent_1)


"""新的layout而且加上了moving penalty"""
# model_agent_0 = PPO.load("final_trained_models/[equilibrium]agent0_highlevelaction_step_penalty_layout_v1\model_600000", env=env_agent_0)
# model_agent_1 = PPO.load("final_trained_models/[equilibrium]agent1_highlevelaction_step_penalty_layout_v1\model_600000", env=env_agent_1)


"""新的layout，partitial_obs"""
# model_agent_0 = PPO.load("final_trained_models/[equilibrium]agent0_highlevelaction_partitial_obs_layout_v1\model_3400000", env=env_agent_0)
# model_agent_1 = PPO.load("final_trained_models/[equilibrium]agent1_highlevelaction_partitial_obs_layout_v1\model_3400000", env=env_agent_1)





# model_agent_0 = PPO.load("../final_trained_models/[equilibrium]agent0_step_penalty_1_vs_1/model_300000", env=env_agent_0)
# model_agent_1 = PPO.load("../final_trained_models/[equilibrium]agent1_step_penalty_1_vs_1/model_300000", env=env_agent_1)

# model_agent_0 = PPO.load("../final_trained_models/[equilibrium]agent0_a0sp_0_a1sp_0/model_500000", env=env_agent_0)
# model_agent_1 = PPO.load("../final_trained_models/[equilibrium]agent1_a0sp_0_a1sp_0/model_500000", env=env_agent_1)

# model_agent_0 = PPO.load("../final_trained_models/[equilibrium]agent0_step_penalty_5_vs_5/model_500000", env=env_agent_0)
# model_agent_1 = PPO.load("../final_trained_models/[equilibrium]agent1_step_penalty_5_vs_5/model_500000", env=env_agent_1)



















# model_agent_0 = PPO.load("../final_trained_models/[equilibrium]agent0_a0sp_0_a1sp_0/model_500000", env=env_agent_0)
# model_agent_1 = PPO.load("../final_trained_models/[equilibrium]agent1_a0sp_0_a1sp_0/model_500000", env=env_agent_1)


model_agent_0 = PPO.load("../final_trained_models/[equilibrium]agent0_a0sp_0_a1sp_1/model_500000", env=env_agent_0)
model_agent_1 = PPO.load("../final_trained_models/[equilibrium]agent1_a0sp_0_a1sp_1/model_500000", env=env_agent_1)


# model_agent_0 = PPO.load("../final_trained_models/[equilibrium]agent0_a0sp_0_a1sp_10/model_500000", env=env_agent_0)
# model_agent_1 = PPO.load("../final_trained_models/[equilibrium]agent1_a0sp_0_a1sp_10/model_500000", env=env_agent_1)


# model_agent_0 = PPO.load("../final_trained_models/[equilibrium]agent0_a0sp_0_a1sp_20/model_500000", env=env_agent_0)
# model_agent_1 = PPO.load("../final_trained_models/[equilibrium]agent1_a0sp_0_a1sp_20/model_500000", env=env_agent_1)


# model_agent_0 = PPO.load("../final_trained_models/[equilibrium]agent0_a0sp_0_a1sp_50/model_500000", env=env_agent_0)
# model_agent_1 = PPO.load("../final_trained_models/[equilibrium]agent1_a0sp_0_a1sp_50/model_500000", env=env_agent_1)


"""=================================================="""

# model_agent_0 = PPO.load("../final_trained_models/[equilibrium]agent0_a0sp_1_a1sp_0/model_500000", env=env_agent_0)
# model_agent_1 = PPO.load("../final_trained_models/[equilibrium]agent1_a0sp_1_a1sp_0/model_500000", env=env_agent_1)


# model_agent_0 = PPO.load("../final_trained_models/[equilibrium]agent0_a0sp_1_a1sp_1/model_500000", env=env_agent_0)
# model_agent_1 = PPO.load("../final_trained_models/[equilibrium]agent1_a0sp_1_a1sp_1/model_500000", env=env_agent_1)


model_agent_0 = PPO.load("../final_trained_models/[equilibrium]agent0_a0sp_1_a1sp_10/model_500000", env=env_agent_0)
model_agent_1 = PPO.load("../final_trained_models/[equilibrium]agent1_a0sp_1_a1sp_10/model_500000", env=env_agent_1)


# model_agent_0 = PPO.load("../final_trained_models/[equilibrium]agent0_a0sp_1_a1sp_20/model_500000", env=env_agent_0)
# model_agent_1 = PPO.load("../final_trained_models/[equilibrium]agent1_a0sp_1_a1sp_20/model_500000", env=env_agent_1)


# model_agent_0 = PPO.load("../final_trained_models/[equilibrium]agent0_a0sp_1_a1sp_50/model_500000", env=env_agent_0)
# model_agent_1 = PPO.load("../final_trained_models/[equilibrium]agent1_a0sp_1_a1sp_50/model_500000", env=env_agent_1)



"""=================================================="""

model_agent_0 = PPO.load("../final_trained_models/[equilibrium]agent0_a0sp_10_a1sp_0/model_500000", env=env_agent_0)
model_agent_1 = PPO.load("../final_trained_models/[equilibrium]agent1_a0sp_10_a1sp_0/model_500000", env=env_agent_1)


# gamma, reward (helping)


# Test the trained models
obs = shared_env.reset()

import time
import cv2
import numpy as np

reward_this = 0

# ===== 先 render 第一帧（非常关键）=====
frame0 = shared_env.render(mode="rgb_array")  # RGB, HxWx3
h, w, _ = frame0.shape

fps = 10  # 与 sleep(0.1) 对齐
video_path = "rollout_1st.mp4"

# fourcc: mp4 常用编码
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

# OpenCV 需要 BGR
video_writer.write(cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR))

for step in range(200):
    action_0, _states_0 = model_agent_0.predict(obs[0])
    action_1, _states_1 = model_agent_1.predict(obs[1])

    total_action = [action_0, action_1]


    # print('agent action: ', shared_env.macroActionName[action_0])


    human_agent_previous_location = [shared_env.agent[1].x, shared_env.agent[1].y]


    total_action, real_execute_macro_actions = shared_env._computeLowLevelActions(
        total_action
    )

    obs, rewards, dones, info = shared_env.step(total_action)
    reward_this += rewards[0] + rewards[1]

    print("---------")
    obs = shared_env._get_macro_obs()


    human_agent_current_location = [shared_env.agent[1].x, shared_env.agent[1].y]


    print(human_agent_previous_location, human_agent_current_location)    
    


    frame = shared_env.render(mode="rgb_array")

    # 尺寸保护（极少数 env 会抖）
    if frame.shape[0] != h or frame.shape[1] != w:
        frame = frame[:h, :w]

    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(reward_this)
    time.sleep(0.1)

    if isinstance(dones, (list, tuple, np.ndarray)):
        if any(dones):
            break
    else:
        if dones:
            break

# ===== 释放资源 =====
video_writer.release()
print(f"Saved video to: {video_path}")
