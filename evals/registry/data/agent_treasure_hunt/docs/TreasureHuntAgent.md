# Treasure Hunt Agent Module Documentation

This module provides an implementation of an agent that plays the Treasure Hunt game. The agent uses a set of tools to navigate and search for treasure in the game.
## Classes
## `TreasureHuntAgentContext`

This class implements the context of the `TreasureHuntAgent`. It stores information about the current state of the game, the number of steps taken, the current coordinate of the agent, the history of the agent's actions, the last action feedback, and the available next actions.

The class has the following methods:
#### `get_context_string()`

This method returns a string representation of the context in YAML format.
## `DigTool`

This class implements a tool that allows the agent to dig at the current location in the treasure hunt game. The class has the following methods:
#### `get_tool_instructions()`

This method returns a string with instructions on how to use the tool.
#### `use(**kwargs)`

This method uses the tool to perform an action on the context.
## `MoveTool`

This class implements a tool that allows the agent to move one step in a chosen direction (North, South, East, or West). The class has the following methods:
#### `get_tool_instructions()`

This method returns a string with instructions on how to use the tool.
#### `use(**kwargs)`

This method uses the tool to perform an action on the context.
## `CompassTool`

This class implements a tool that points to the treasure in the treasure hunt game. The class has the following methods:
#### `get_tool_instructions()`

This method returns a string with instructions on how to use the tool.
#### `use(**kwargs)`

This method uses the tool to perform an action on the context.
## `ExplorationAgentState`

This class represents a toolbox containing a set of tools for the exploration phase of the game. The class has the following methods:
#### `choose_next_tool(command:str)`

This method parses the provided command to choose the next tool.
#### `check_transition_state()`

This method decides if we should transition to a different agent state based on the context.
## `DiggingAgentState`

This class represents a toolbox containing a set of tools for the digging phase of the game. The class has the following methods:
#### `choose_next_tool(command:str)`

This method parses the provided command to choose the next tool.
#### `check_transition_state()`

This method decides if we should transition to a different agent state based on the context.
## `TreasureHuntAgent`

This class represents the agent that plays the Treasure Hunt game. The class has the following methods:
#### `update_context()`

This method updates the context after an action has been taken.
#### `do_next_action()`

This method executes the next action chosen by the agent.

