import pygame
import gym
import warnings
import sys
import numpy
import time
import pandas as pd
import shimmy
from datetime import datetime
from stable_baselines3 import PPO

# --- COMPATIBILITY PATCH ---
try:
    sys.modules["numpy._core"] = numpy.core
    if hasattr(numpy.core, "numeric"):
        sys.modules["numpy._core.numeric"] = numpy.core.numeric
    if hasattr(numpy.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
    import numpy.random._pickle
    original_ctor = numpy.random._pickle.__bit_generator_ctor
    def patched_ctor(bit_generator_name):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        return original_ctor(bit_generator_name)
    numpy.random._pickle.__bit_generator_ctor = patched_ctor
except Exception as e:
    print(f"Patch Warning: {e}")

warnings.filterwarnings("ignore")

# --- EXPERIMENT SETTINGS ---
TOTAL_ROUNDS = 10
ROUND_DURATION = 45
SECONDS_PER_STEP = 0.2

ACTION_LABELS = {0: "RIGHT", 1: "DOWN", 2: "LEFT", 3: "UP", 4: "STAY"}
KEYS_ACTIONS = {
    pygame.K_UP: 3, 
    pygame.K_RIGHT: 0, 
    pygame.K_DOWN: 1, 
    pygame.K_LEFT: 2
}

# --- WRAPPER ---
class SingleAgentWrapper_accept_keyboard_action(gym.Wrapper):
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
        step_result = self.env.step(actions)
        self.obs = self.env._get_macro_obs()
        return self.obs[self.agent_index], step_result[1], step_result[2], step_result[3]

# --- HELPER: Is Agent Holding Dish? ---
def is_holding_dish(agent):
    if not agent.holding: return False
    obj = agent.holding
    if "Plate" in type(obj).__name__:
        if hasattr(obj, 'containing') and obj.containing is not None:
            return True
    return False

# --- RUN ROUND ---
def run_single_round(round_num, model_path, prev_stats=None):
    print(f"\n=== ROUND {round_num} INITIALIZING ===")
    
    # 1. SETUP ENV
    mac_env_id = 'Overcooked-MA-equilibrium-v0'
    rewardList = [{
        "minitask finished": 0, "minitask failed": 0, "metatask finished": 0,
        "metatask failed": 0, "goodtask finished": 10, "goodtask failed": 0,
        "subtask finished": 20, "subtask failed": 0, "correct delivery": 200,
        "wrong delivery": -50, "step penalty": -1, "penalize using dirty plate": 0,
        "penalize using bad lettuce": -20, "pick up bad lettuce": -100
    },{
        "minitask finished": 0, "minitask failed": 0, "metatask finished": 0,
        "metatask failed": 0, "goodtask finished": 10, "goodtask failed": 0,
        "subtask finished": 20, "subtask failed": 0, "correct delivery": 200,
        "wrong delivery": -50, "step penalty": -1, "penalize using dirty plate": 0,
        "penalize using bad lettuce": 0, "pick up bad lettuce": 0
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
    
    import gym_macro_overcooked 
    base_env = gym.make(mac_env_id, **env_params)
    env = SingleAgentWrapper_accept_keyboard_action(base_env, agent_index=0)
    
    # 2. LOAD MODEL
    custom_objects = {
        "action_space": env.action_space,
        "observation_space": env.observation_space,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0
    }
    agent_model = PPO.load(model_path, env=env, custom_objects=custom_objects)
    
    obs = env.reset()
    
    # 3. SETUP PYGAME & FONTS
    if not pygame.get_init():
        pygame.init()
    
    pygame.display.set_caption(f"Round {round_num}/{TOTAL_ROUNDS}")
    font = pygame.font.SysFont('Arial', 24, bold=True)
    
    # Try to get screen surface
    try:
        base_env.render(mode='rgb_array')
        screen = pygame.display.get_surface() # Get the window gym created
    except:
        screen = pygame.display.set_mode((600,600))

    start_time = time.time()
    last_tick = time.time()
    current_human_action = 4
    
    # Round Statistics
    dishes_served = 0
    human_steps_taken = 0 # Counter for actions
    telemetry = []
    
    human_was_holding = False
    ai_was_holding = False
    
    running = True
    
    while running:
        elapsed = time.time() - start_time
        if elapsed > ROUND_DURATION:
            print(f"Time Up! Final Dishes: {dishes_served}")
            running = False
            break
            
        # Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None, None
            if event.type == pygame.KEYDOWN:
                if event.key in KEYS_ACTIONS:
                    current_human_action = KEYS_ACTIONS[event.key]
        
        # Step Logic
        if time.time() - last_tick > SECONDS_PER_STEP:
            last_tick = time.time()
            timestamp = "{:.3f}".format(elapsed)
            
            # Count Action (If human is not staying still)
            if current_human_action != 4:
                human_steps_taken += 1

            # Get Agent Positions
            try:
                agents = base_env.unwrapped.agent
                human_pos = (agents[1].x, agents[1].y)
                ai_pos = (agents[0].x, agents[0].y)
                human_is_holding = is_holding_dish(agents[1])
                ai_is_holding = is_holding_dish(agents[0])
            except:
                human_pos, ai_pos = (-1,-1), (-1,-1)
                human_is_holding, ai_is_holding = False, False

            # Predict & Step
            ai_raw, _ = agent_model.predict(obs)
            ai_action = int(ai_raw.item()) if isinstance(ai_raw, numpy.ndarray) else int(ai_raw)
            prim_pair, _ = base_env._computeLowLevelActions([ai_action, 0])
            ai_micro = prim_pair[0]
            
            finals = [ai_micro, current_human_action]
            obs, reward_info, done, info = env.step(finals[0], finals[1])
            
            # Scoring Logic
            step_reward = reward_info[2] if isinstance(reward_info, list) and len(reward_info) > 2 else reward_info if not isinstance(reward_info, list) else reward_info[0]

            human_lost_dish = human_was_holding and not human_is_holding
            ai_lost_dish = ai_was_holding and not ai_is_holding
            is_handoff_to_ai = human_lost_dish and (ai_is_holding and not ai_was_holding)
            is_handoff_to_human = ai_lost_dish and (human_is_holding and not human_was_holding)
            
            if step_reward > 100:
                dishes_served += 1
            elif human_lost_dish and not is_handoff_to_ai and (human_pos[1] >= 3 or human_pos[0] >= 3):
                 dishes_served += 1
            elif ai_lost_dish and not is_handoff_to_human and (ai_pos[1] >= 3 or ai_pos[0] >= 3):
                 dishes_served += 1

            human_was_holding = human_is_holding
            ai_was_holding = ai_is_holding
            
            telemetry.append({
                "Round": round_num,
                "Timestamp": timestamp,
                "Score": dishes_served,
                "Human_Steps": human_steps_taken,
                "AI_X": ai_pos[0], "AI_Y": ai_pos[1],
                "Human_X": human_pos[0], "Human_Y": human_pos[1],
                "Action_Human": ACTION_LABELS.get(current_human_action, "UNKNOWN")
            })
            
            current_human_action = 4
            
            # --- RENDER WITH TEXT ---
            try:
                base_env.render(mode='rgb_array') # Draws the game
                
                # Create HUD Text
                # 1. Current Stats (Green)
                curr_text = f"ROUND {round_num} | Dishes: {dishes_served} | Actions: {human_steps_taken}"
                curr_surf = font.render(curr_text, True, (0, 255, 0), (0, 0, 0)) # Green text, Black BG
                screen.blit(curr_surf, (10, 10))
                
                # 2. Previous Round Stats (Yellow)
                if prev_stats:
                    prev_text = f"PREV ROUND | Dishes: {prev_stats['dishes']} | Actions: {prev_stats['actions']}"
                    prev_surf = font.render(prev_text, True, (255, 200, 0), (0, 0, 0)) # Yellow text
                    screen.blit(prev_surf, (10, 40))
                
                pygame.display.flip()
            except: 
                pass
            
    env.close()
    
    # Return current stats to be used as 'prev' in next round
    current_stats = {"dishes": dishes_served, "actions": human_steps_taken}
    return dishes_served, telemetry, current_stats

def main():
    model_path = r"final_trained_models/[equilibrium]agent0_highlevelaction_layout_v1\model_700000"
    
    all_summaries = []
    all_telemetry = []
    last_round_stats = None # Starts empty
    
    print(f"\n>>> STARTING STUDY: {TOTAL_ROUNDS} Rounds x {ROUND_DURATION} Seconds <<<")
    
    for i in range(1, TOTAL_ROUNDS + 1):
        # Pass last_round_stats into the function
        score, log, last_round_stats = run_single_round(i, model_path, prev_stats=last_round_stats)
        
        if score is None: 
            print("Experiment Aborted.")
            break
            
        all_summaries.append({"Round": i, "Score": score, "Actions": last_round_stats['actions']})
        all_telemetry.extend(log)
        
        print(f"Round {i} Finished. Score: {score}, Actions: {last_round_stats['actions']}")
        time.sleep(1)
        
    print("\nSaving Data...")
    df_sum = pd.DataFrame(all_summaries)
    df_sum.to_csv("pilot_summary_GRAPH_THIS.csv", index=False)
    
    df_tel = pd.DataFrame(all_telemetry)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_tel.to_csv(f"experiment_log_{timestamp_str}.csv", index=False)
    
    pygame.quit()

if __name__ == "__main__":
    main()