# Eval Summary
We created this evaluation as a template for creating future evaluations that deal with agent-tool types of problems. To achieve this, we have introduced a new module (evals\completion_fns\agent.py), which provides base classes that we can use to quickly create customized completion functions. These functions will evaluate iterations of a series of responses from an LLM, where each step modifies some context to pursue a goal.

As an example:
we have created an example evaluation called "agent_treasure_hunt," which involves an agent searching for a treasure in a grid of cells. The agent can move in four directions: north, south, east, and west. Additionally, the agent can use a compass to determine the direction of the treasure. Once the agent reaches the cell with the treasure, they can dig to find the treasure.

To run this eval you must set the following environment variables:
EVALS_THREAD_TIMEOUT = 600
PYTHONPATH = PATH/TO/evals/registry/data/agent_treasure_hunt


# Documentation Summary
This document provides a summary of the new modules, classes, and methods provided for the base `Agent` classes and the `Treasure Hunt` game.
## Agent Completion Function Module 
``` evals\completion_fns\agent.py ```

This module provides interfaces to create a completion function and an agent that can take actions based on a provided context.
### Main Interfaces 
- `AgentContext` 
- `IAgentState` 
- `Tool` 
- `Agent` 
- `AgentState` 
- `AgentCompletionResult` 
- `AgentCompletionFn`
## Treasure Hunt Agent Module 
``` evals\registry\data\agent_treasure_hunt\treasure_hunt_agent.py ```

This module provides an implementation of an agent that plays the `Treasure Hunt` game. The agent uses a set of tools to navigate and search for treasure in the game.
### Main Classes 
- `TreasureHuntAgentContext` 
- `DigTool` 
- `MoveTool` 
- `CompassTool` 
- `ExplorationAgentState` 
- `DiggingAgentState` 
- `TreasureHuntAgent`
## Treasure Hunt Agent Completion Function Module
``` evals\registry\data\agent_treasure_hunt\treasure_hunt_completion_fn.py ```

This module contains classes and functions that implement an agent for playing a `Treasure Hunt` game.
### Main Classes 
- `TreasureHuntAgentCompletionFn` 
- `TreasureHuntAgentContext` 
- `TreasureHuntAgent` 
- `TreasureHunt`
### Main Functions
- `parse_positions(s)`

## Treasure Hunt Module
``` evals\registry\data\agent_treasure_hunt\treasure_hunter_game.py ```

The `TreasureHunt` module provides a class that represents a treasure hunt game.
### Main Class 
- `TreasureHunt`

For more detailed information about each module, class, and method, please refer to the full documentation.