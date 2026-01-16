# First, My Previous Overcooked User Study Codebase

## How to Run This User Study Codebase

Under this path, you will find the code I used in my previous project to run user studies. I built a simple Python server using **Flask**, and the front end was written in **HTML + JavaScript**. The setup is not complicated. The version of Python is 3.11.

## Start the Backend Server

`backend.py` is the server code. Start it with:

```bash
python backend.py
```

## Run the Frontend

Once the server is running, open:

- `UserStudy/frontend.html`

This will start the study interface and begin data collection.

## Notes

Some comments in the code are written in Chinese. If they appear as garbled text on your side, you can safely ignore them.

For now, both the backend and frontend can run locally. If you later want to deploy everything on a real server, I can guide you through that. For the moment, please start by running everything locally so that you can understand and debug the code first.

## Folder and File Overview

- **utils.py**  
  Contains some utility functions. You can ignore this for now.

- **gym_macro_overcooked/**  
  Includes the core code for the environment.

- **userstudy_models/**  
  Contains the trained agent models for four different Overcooked map layouts.  
  There are four agents for each layout, so in total **16 models (4 × 4)**.  
  Once `backend.py` is running, it will load a specific model file based on the layout ID and model ID sent from the front end.  
  This allows users to play Overcooked with one of these agent partners.

## Final Reminder

Please note that the repository I shared with you is a fairly complex project. It contains the full codebase from my previous project and is much more complicated than what you are currently working on. As a result, some parts may be difficult to understand at first. Try to go through it step by step and see what you can make sense of first.



# Second, Our Human-AI Equilibrium Project!

Above is the introduction to the frontend and backend components of the user study from my previous project.  
Next, I will introduce our current **Human–AI Equilibrium** project.

---

## Code Structure

### `Equilibrium_project/`
This directory contains the current codebase of the equilibrium project.

### Please go into this directory. Then, you will see the following:

### `gym_macro_overcooked/`
This folder contains the environment code.  
If you would like to modify the layouts or change the number and positions of vegetables, plates, and other items in the environment, you should make the changes here.

### `final_trained_models/`
This folder stores the models I have trained so far.

---

## How to Run the Code

### Play the Game Manually
`[AAA][equilibrium]play.py` allows you to play Overcooked manually.

Run:
```bash
python [AAA][equilibrium]play.py
```

A game window will pop up.  
- Press **1** or **2** to select different agents.  
- Use the **arrow keys** to control your actions.

---

## Training Scripts

The following scripts are used to train agents:

- `[AAA][equilibrium]train_highlevelaction_addstep_penalty.py`  
- `[AAA][equilibrium]train_highlevelaction_partitial_obs.py`  
- `[AAA][equilibrium]train_highlevelaction.py`  

Each script corresponds to a different training setting or observation configuration.

---

## Testing Trained Models

`[AAA][equilibrium]test.py` is used to load the trained models and evaluate their performance.

Run this script to see how the trained agents behave in the environment.
