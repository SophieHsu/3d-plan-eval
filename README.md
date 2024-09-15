# Integrating Human Field of View in Human-Aware Collaborative #

## Project Overview ##

### Problem ###
Most research on human-AI collaboration assumes humans have full knowledge of their surroundings, 
which is unrealistic.

### Goal ###
Adapt to humans' changing subtask intentions while considering their limited FOV.

### Approach ###
- Integrate FOV into a human-aware probabilistic planning framework.
- Develop a hierarchical online planner to handle large state spaces efficiently.
- Enable the AI agent to explore actions that enter the human's FOV to influence their intended subtask.

### Validation ###
- Conducted a user study using a 2D cooking domain.
- Extended findings to a VR kitchen environment.

### Results ###
- The FOV-aware planner reduced interruptions and redundant actions during collaboration.
- Similar collaborative behaviors were observed in both 2D and VR environments.

### Significance ###
This research addresses a gap in human-AI collaboration by accounting for humans' perceptual 
limitations, potentially leading to more natural and efficient teamwork between humans and AI
agents.

## Installation Instructions ## 

## Usage ##

## Project Structure ##
The most relevant components in the project are described below.
```
├── 3d_plan_eval_main.py
├── igibson/
├── lsi_3d/
│   ├── agents/
│   ├── config/
│   ├── environment/
│   ├── logs/
│   ├── mdp
│   ├── motion_controllers/
│   ├── planners/
│   ├── utils/
│   └── overcooked_state_dump.json
├── README.md
└── utils.py
```

### Main Scripts ###
- `3d_plan_eval_main.py` is the main entrypoint of the project that defines the necessary `RUNNER` class to 
    run the project.
### `iGibson` ###
- The `iGibson` directory contains the core components for the iGibson simulation framework. Please refer 
    [here](https://github.com/StanfordVL/iGibson) for more details.

### `lsi_3d` Components ###
- `agents` - This directory contains various files to define the agent classes that are responsible for low and high 
    level control of both the AI agents and the human player.
- `config` - This directory contains the files to set up configurations of the various components of this 
  project, namely, agent, algorithm, experiment and map. These configs are defined in
  [`toml`](https://toml.io/en/) files. 
- `environment` - This directory consists of files defining the elements of the environment. `vision_limit_env`, 
  `tracking_env` and `lsi_env` files contain implementations of different kinds of environments. The `kitchen` file
  defines the main class used to tie all components (VR environment, iGibson, planners, etc.) together. Other files, 
  `objects`, `object_configs` and `actions` define the objects, their configs within the environments and the actions
  that can be applied on them in the environment respectively.
- `logs` - Directory to store logs from experimental runs of the project.
- `mdp` - Implementation of out mdp solver.
- `planners` - Different planner implementations for different environment typer for different agents (human vs AI).
- `utils` - Common utility functions.

## Results ##

## Citation ##
```
Coming soon
```

## Contact ##
```
For any questions, reach out to: yachuanh at usc dot edu
```