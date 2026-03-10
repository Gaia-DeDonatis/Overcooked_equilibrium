import json

def count_actions(round_data, agent='human'):
    """
    This function counts the number of actions (excluding "stay") for the given agent in a round.
    
    Parameters:
        round_data (dict): The data for a round (assumed to contain actions data).
        agent (str): The agent whose actions are to be counted ('human' or 'ai').

    Returns:
        int: The number of actions taken by the agent excluding 'stay'.
    """
    actions_count = 0
    
    # Extract the action log for the specified agent
    action_log = round_data.get('action_log', {}).get(agent, [])
    
    for action in action_log:
        # For human, exclude 'Stay' action
        if agent == 'human' and action.get('action') != 'STAY':
            actions_count += 1
        # For AI, exclude 'stay' based on 'low' or 'macro' action
        elif agent == 'ai' and action.get('arrow') != 'STAY':
            actions_count += 1
    
    return actions_count

def process_rounds(file_path):
    """
    This function processes all the rounds in the provided data file and calculates the number of actions
    for both human and AI, excluding 'stay' actions.

    Parameters:
        file_path (str): Path to the JSON file containing the round data.
    
    Returns:
        list of dict: A list containing the calculated results for each round.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Store the calculated results
    results = []
    
    # Loop through the rounds in the data
    for round_data in data.get('rounds', []):
        round_info = round_data.get('summary', {})
        
        # Extract the episode and round number directly from round_data
        episode = round_data.get('episode_index', 'N/A')  # Or any other field that stores episode info
        round_number = round_data.get('round_index_global', 'N/A')  # Or other relevant field for round info
        
        # Calculate actions for both human and AI
        human_actions = count_actions(round_data, 'human')
        ai_actions = count_actions(round_data, 'ai')
        
        # Append the results
        results.append({
            'episode': episode,
            'round': round_number,
            'human_actions': human_actions,
            'ai_actions': ai_actions,
            'dishes_served': round_info.get('dishes_served', 0)
        })
    
    return results

def save_results(results, output_file):
    """
    This function saves the calculated results to a JSON file.

    Parameters:
        results (list of dict): The results to be saved.
        output_file (str): The path to the output file.
    """
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

# Example usage:
input_file_path = r'C:\Users\dedong1\work\Overcooked_equilibrium\submissions\Pilot_01_09_03\final_result.json'  # Replace with the path to your file
output_file_path = 'calculated_actions.json'  # Path where the results will be saved

# Process the rounds and calculate actions
round_results = process_rounds(input_file_path)

# Save the results to a file
save_results(round_results, output_file_path)

print(f"Results saved to {output_file_path}")