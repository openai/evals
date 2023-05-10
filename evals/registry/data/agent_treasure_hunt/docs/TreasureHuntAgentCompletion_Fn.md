# Documentation for Treasure Hunt Agent Module

This module contains classes and functions that implement an agent for playing a Treasure Hunt game.
## Classes
### `TreasureHuntAgentCompletionFn`

This class extends the `AgentCompletionFn` class from the `evals.completion_fns.agent` module. It implements a completion function for the Treasure Hunt game.
#### Methods 
- `check_for_end_context()`: This method checks if the game has ended by checking if the `state` attribute of the `agent_context` is equal to `"game_over"`. Returns `True` if the game is over and `False` otherwise. 
- `setup_agent(prompt)`: This method sets up the agent for the game. It takes in a prompt and extracts the starting position of the agent and the location of the treasure from the prompt. It then creates a new `TreasureHunt` game with the extracted positions and sets up the `TreasureHuntAgentContext` and `TreasureHuntAgent` objects for the game.
### `TreasureHuntAgentContext`

This class represents the context of the Treasure Hunt game for the agent. It contains the following attributes: 
- `state`: A string representing the current state of the game. Can be one of `"game_on"`, `"game_over"`, or `"win"`. 
- `steps_taken`: An integer representing the number of steps taken by the agent so far.
### `TreasureHuntAgent`

This class represents the agent for the Treasure Hunt game. It takes in a `TreasureHuntAgentContext` object and a `TreasureHunt` object during initialization. It contains the following methods: 
- `get_action(observation)`: This method takes in an observation of the game state and returns an action for the agent to take. The action can be one of `"north"`, `"east"`, `"south"`, `"west"`, or `"dig"`. 
- `update_context(observation)`: This method updates the `TreasureHuntAgentContext` object with the current state of the game.
### `TreasureHunt`

This class represents the Treasure Hunt game. It takes in the starting position of the agent and the location of the treasure during initialization. It contains the following methods: 
- `step(action)`: This method takes in an action and updates the state of the game. Returns an observation of the new game state. 
- `get_state()`: This method returns the current state of the game. 
- `get_reward()`: This method returns the current reward for the agent based on the state of the game.
## Functions
### `parse_positions(s)`

This function takes in a string `s` containing the starting positions of the agent and the location of the treasure and returns a dictionary containing the positions. The starting position of the agent is represented by the letter `"A"`, and the location of the treasure is represented by the letter `"T"`. The positions are represented as tuples of integers in the format `(x, y)`.
