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


        # Gaia, please see here
        # ====================
        self.firsttime_down_go_to_counter = True
        self.firsttime_up_get_counter_lettuce = True
        # ====================


    def reset(self):
        self.obs = self.env.reset()

        # Gaia, please see here
        # ====================
        self.firsttime_down_go_to_counter = True
        self.firsttime_up_get_counter_lettuce = True
        # ====================

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



        return self.obs[self.agent_index], rewards[2], dones, info






# Gaia, please see here
# ====================
ITEMNAME = ["space", "counter", "agent", "tomato", "lettuce", "plate", "knife", "delivery", "onion", "dirtyplate", "badlettuce"]

macroActionDict = {"stay": 0, "get lettuce 1": 1, "get lettuce 2": 2, "get plate 1": 3, "get plate 2": 4, "go to knife 1": 5, "deliver 1": 6, "chop": 7, "go to counter": 8, "right": 9, "down": 10, "left": 11, "up": 12}


def check_action_benevolence(env, action_up, action_down, firsttime_down_go_to_counter, firsttime_up_get_counter_lettuce):

    agent_up = env.agent[0]
    agent_down = env.agent[1]


    counter1_x = 2
    counter1_y = 2

    counter1 = ITEMNAME[env.map[counter1_x][counter1_y]]

    reward_shaping_bonus = 0
    total_reward_bonus = 0


    reward_bonus_up = 0
    reward_bonus_down = 0

    counters = [counter1]

    if any(counter in ("lettuce") for counter in counters):
        best_action = intelligently_find_item_number(env, agent_up, "get lettuce")

        if firsttime_up_get_counter_lettuce == True:
            reward_shaping_bonus = check_benevolence(env, best_action, action_up)
            if reward_shaping_bonus == 20:
                total_reward_bonus += reward_shaping_bonus
                reward_bonus_up = 100
                firsttime_up_get_counter_lettuce = False


    if all(counter not in ("lettuce") for counter in counters):

        if agent_down.holding and isinstance(agent_down.holding, Lettuce):
            best_action = "go to counter"

            if firsttime_down_go_to_counter == True:

                reward_shaping_bonus = check_benevolence(env, best_action, action_down)

                if reward_shaping_bonus == 20:
                    total_reward_bonus += reward_shaping_bonus
                    reward_bonus_down = 100
                    firsttime_down_go_to_counter = False

    return reward_bonus_up, reward_bonus_down, firsttime_down_go_to_counter, firsttime_up_get_counter_lettuce




def find_best_reachable_index(can_reach_1, can_reach_2, can_reach_3, distance_1, distance_2, distance_3):
    reachable_indices = []
    distances = []
    
    if can_reach_1 != 4:
        reachable_indices.append(0)
        distances.append(distance_1)
    if can_reach_2 != 4:
        reachable_indices.append(1)
        distances.append(distance_2)
    if can_reach_3 != 4:
        reachable_indices.append(2)
        distances.append(distance_3)
    
    if not reachable_indices:
        return False
    
    if len(reachable_indices) == 1:
        return reachable_indices[0]
    
    min_distance_index = reachable_indices[distances.index(min(distances))]
    return min_distance_index



def intelligently_find_item_number(env, agent_item, raw_name):

    if raw_name == "get plate":
        target_x_1, target_y_1 = env._findPOitem(agent_item, macroActionDict[raw_name + " 1"])
        can_reach_1 = env._navigate(agent_item, target_x_1, target_y_1)
        distance_1 = env._calDistance(target_x_1, target_y_1, agent_item.x, agent_item.y)

        target_x_2, target_y_2 = env._findPOitem(agent_item, macroActionDict[raw_name + " 2"])
        can_reach_2 = env._navigate(agent_item, target_x_2, target_y_2)
        distance_2 = env._calDistance(target_x_2, target_y_2, agent_item.x, agent_item.y)

        target_x_3, target_y_3 = env._findPOitem(agent_item, macroActionDict["get dirty plate"])
        can_reach_3 = env._navigate(agent_item, target_x_3, target_y_3)
        distance_3 = env._calDistance(target_x_3, target_y_3, agent_item.x, agent_item.y)


        best_action = "stay"

        min_distance_index = find_best_reachable_index(can_reach_1, can_reach_2, can_reach_3, distance_1, distance_2, distance_3)

        if min_distance_index == 0:
            best_action = raw_name + " 1"
        if min_distance_index == 1:
            best_action = raw_name + " 2"
        if min_distance_index == 2:
            best_action = "get dirty plate"

        return best_action

    

    target_x_1, target_y_1 = env._findPOitem(agent_item, macroActionDict[raw_name + " 1"])
    can_reach_1 = env._navigate(agent_item, target_x_1, target_y_1)
    distance_1 = env._calDistance(target_x_1, target_y_1, agent_item.x, agent_item.y)

    target_x_2, target_y_2 = env._findPOitem(agent_item, macroActionDict[raw_name + " 2"])
    can_reach_2 = env._navigate(agent_item, target_x_2, target_y_2)
    distance_2 = env._calDistance(target_x_2, target_y_2, agent_item.x, agent_item.y)

    best_action = "stay"
    if can_reach_1 == 4 and can_reach_2 != 4:
        best_action = raw_name + " 2"

    if can_reach_1 != 4 and can_reach_2 == 4:
        best_action = raw_name + " 1"

    if can_reach_1 != 4 and can_reach_2 != 4:
        if distance_1 <= distance_2:
            best_action = raw_name + " 1"

        else:
            best_action = raw_name + " 2"
    
    return best_action



def check_benevolence(env, best_action, action):
    env.reward = 0
    if action == macroActionDict[best_action] and macroActionDict[best_action] != 0:
        env.reward += 20
    return env.reward



import re

def parse_policy_id(policy_id: str):
    """
    Parse step_penalty (AAA) and cooperation_bonus (CCC) from policy_id

    policy_id format:
    [equilibrium]agent0_a0sp_AAA_a1sp_BBB_helping_CCC_gammaDDD_EEE
    
    """

    pattern = r"a0sp_([^_]+).*?helping_([^_]+)"
    match = re.search(pattern, policy_id)

    if not match:
        raise ValueError(f"Invalid policy_id format: {policy_id}")

    step_penalty = match.group(1)
    cooperation_bonus = match.group(2)

    return step_penalty, cooperation_bonus
# ====================


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



ai_policy_id = "[equilibrium]agent0_a0sp_0_a1sp_0_helping_True_gamma0.9_0.9"
human_policy_id = "[equilibrium]agent1_a0sp_0_a1sp_0_helping_True_gamma0.9_0.9"


model_agent_0 = PPO.load("../policy_pool_using_gamma/" + ai_policy_id + "/model_500000", env=env_agent_0)
model_agent_1 = PPO.load("../policy_pool_using_gamma/" + human_policy_id + "/model_500000", env=env_agent_1)




# gamma, reward (helping)


# Test the trained models
obs = shared_env.reset()

import time
import cv2
import numpy as np


ai_reward = 0
human_reward = 0



for step in range(200):
    action_0, _states_0 = model_agent_0.predict(obs[0])
    action_1, _states_1 = model_agent_1.predict(obs[1])

    total_action = [action_0, action_1]


    # print('agent action: ', shared_env.macroActionName[action_0])



    # Gaia, please see here
    # ====================
    """1. Get AI agent previous location before running step()"""
    ai_agent_previous_location = [shared_env.agent[0].x, shared_env.agent[0].y]
    # ====================
    
    total_action, real_execute_macro_actions = shared_env._computeLowLevelActions(
        total_action
    )


    # Gaia, please see here
    # ====================
    """2. Calculate cooperation bonus based on the second returned actions from _computeLowLevelActions()"""
    benevolence_reward_up, benevolence_reward_down, env_agent_0.firsttime_down_go_to_counter, env_agent_0.firsttime_up_get_counter_lettuce = check_action_benevolence(shared_env, real_execute_macro_actions[0], real_execute_macro_actions[1], env_agent_0.firsttime_down_go_to_counter, env_agent_0.firsttime_up_get_counter_lettuce)
    # ====================


    obs, rewards, dones, info = shared_env.step(total_action)

    """This is human reward, just the team reward, shown to participants."""
    human_reward += float(rewards[0] + rewards[1])

    print("---------")
    obs = shared_env._get_macro_obs()



    # Gaia, please see here
    # ====================
    """3. After step(), calculate the reward"""
    ai_agent_current_location = [shared_env.agent[0].x, shared_env.agent[0].y]


    step_penalty, cooperation_bonus = parse_policy_id(ai_policy_id)

    print("step_penalty:", step_penalty)
    print("cooperation_bonus:", cooperation_bonus)

    print(ai_agent_previous_location)
    print(ai_agent_current_location)

    
    """This is AI reward, for Bayes Optimization"""
    if cooperation_bonus == True:
        if ai_agent_previous_location != ai_agent_current_location:
            ai_reward += float(rewards[0] + rewards[1]) - float(step_penalty) + benevolence_reward_up
        else:
            ai_reward += float(rewards[0] + rewards[1]) + benevolence_reward_up

    else:
        if ai_agent_previous_location != ai_agent_current_location:
            ai_reward += float(rewards[0] + rewards[1]) - float(step_penalty)
        else:
            ai_reward += float(rewards[0] + rewards[1])
    # ====================



    print('AI reward: ', ai_reward)
    print('Human reward: ', human_reward)


    frame = shared_env.render(mode="rgb_array")

    time.sleep(1)

    if isinstance(dones, (list, tuple, np.ndarray)):
        if any(dones):
            break
    else:
        if dones:
            break

# ===== release =====
# video_writer.release()
# print(f"Saved video to: {video_path}")