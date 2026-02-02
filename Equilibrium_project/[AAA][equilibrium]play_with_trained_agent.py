import pygame
import gym
from gym.envs.registration import register
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper
from stable_baselines3 import PPO
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

# Define keys for actions
# 0: right, 1: down, 2: left, 3: up, 4: still
KEYS_ACTIONS = {
    pygame.K_UP: 3,       # Up
    pygame.K_RIGHT: 0,    # Right
    pygame.K_DOWN: 1,     # Down
    pygame.K_LEFT: 2,     # Left
}

# Define agent selection keys
AGENT_KEYS = {
    pygame.K_1: 0,
    pygame.K_2: 1,
    pygame.K_3: 2
}



class SingleAgentWrapper_accept_keyboard_action(gym.Wrapper):
    """
    A wrapper to extract a single agent's perspective from a multi-agent environment.
    """
    def __init__(self, env, agent_index):
        super(SingleAgentWrapper_accept_keyboard_action, self).__init__(env)
        self.agent_index = agent_index
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.obs = None


    def reset(self):
        self.obs = self.env.reset()

        return self.obs[self.agent_index]



    def step(self, action, keyboard_action):
        
        actions = [action, keyboard_action]

        self.obs, rewards, dones, info = self.env.step(actions)

        self.obs = self.env._get_macro_obs()

        
        return self.obs[self.agent_index], rewards[2], dones, info





from PIL import Image


def main():

    frames = []


    # pygame.init()
    # screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Overcooked Control")

    mac_env_id = 'Overcooked-MA-equilibrium-v0'


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


    env = gym.make(mac_env_id, **env_params)

    shared_env = gym.make(mac_env_id, **env_params)

    env_agent_0 = SingleAgentWrapper_accept_keyboard_action(shared_env, agent_index=0)





    # Load the agent model
    agent_model = PPO.load("final_trained_models/[equilibrium]agent0_highlevelaction_layout_v1\model_700000", env=env_agent_0)
    
    obs = env_agent_0.reset()
    env.reset()


    # For saving videos.
    frame = env.render(mode='rgb_array')  # Capture the initial frame
    frames.append(frame)  # Append the first frame


    # Render the initial state of the environment
    # env.render()

    # Select an agent to play
    selected_agent = 1

    # Main game loop
    running = True

    step = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            

            if event.type == pygame.KEYDOWN:
                # Handle agent selection
                if event.key in AGENT_KEYS:
                    selected_agent = AGENT_KEYS[event.key]
                
                # Handle actions
                if event.key in KEYS_ACTIONS:
                    step += 1
                    action = [4] * env.n_agent  # Default action is 'stay' for all agents
                    action[selected_agent] = KEYS_ACTIONS[event.key]

                    if selected_agent != 0:

                        print('obs 1: ', obs)
                        ai_action, _states = agent_model.predict(obs)

                        print('Agent action: ', env.macroActionName[ai_action])
                    
                        primitive_action, _ = env._computeLowLevelActions([ai_action, 0])

                        action[0] = primitive_action[0]

                    
                    env.step(action)

                    print(action[0])
                    print(action[1])
                    obs, rewards, dones, info = env_agent_0.step(action[0], action[1])

                    print('obs 2: ', obs)


                    frame = env.render(mode='rgb_array')  # Get the rendered frame as an array
                    frames.append(frame)



        # screen.fill((0, 0, 0))
        pygame.display.flip()



    # time.sleep(0.5)

    # Save the frames as a video

    # output_path = "Low_Integrity.mp4"
    # imageio.mimsave(output_path, frames, fps=2)  # Save the video at 10 frames per second
    # print(f"Video saved to {output_path}")




    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
    # import random
    # print(random.randint(1, 2))

