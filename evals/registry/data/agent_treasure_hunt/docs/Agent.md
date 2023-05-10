oaieval treasure-hunt agent-treasure-hunt
# Module

This module provides an abstract framework for agents and their tools. An agent can be in different states, each state with its own set of tools. The agent performs actions in its context, and the state of the agent can change based on the context.
___
## Table of contents
- AgentContext
- IAgentState
- Tool
- Agent
- AgentState
___
## AgentContext
### Description

An abstract class representing the context of an agent.
### Methods 
- `get_context_string()`: Retrieve the context string of the agent.
___
## IAgentState
### Description

An abstract interface for agent state to break cyclic dependency between Agent and AgentState.
### Methods 
- `do_next_action()`: Perform the next action for the chosen tool. 
- `get_tools_instructions() -> list[str]`: Get a list of the instructions for the available tools in this state. 
- `choose_next_tool(command: str)`: Parse the provided command to choose the next tool. 
- `use_tool(**kwargs) -> str`: Use a tool from the toolbox to perform an action on the context. 
- `check_transition_state()`: Decide if we should transition to a different agent state based on the context.
___
## Tool
### Description

An abstract base class for defining the interface for a tool.
### Properties 
- `agent_state`: The agent state in which the tool is used.
### Methods 
- `get_tool_instructions() -> str`: Returns a string explaining the tool's interface to the agent. 
- `use(**kwargs) -> str`: Uses the tool to perform an action on the context.
___
## Agent
### Description

An abstract class representing an agent.
### Properties 
- `agent_context`: The agent's context. 
- `agent_state`: The agent's state.
### Methods 
- `do_next_action(command: str)`: Perform the next action of the agent using the chosen tool and its arguments. 
- `update_context()`: Update the AgentContext after an action has been taken.
___
## AgentState
### Description

An abstract base class representing the state of an agent.
### Properties 
- `agent`: The agent the state belongs to. 
- `chosen_tool`: The chosen tool for the agent's next action. 
- `chosen_tool_kwargs`: The chosen tool's arguments. 
- `tools`: List of available tools for this state.
### Methods 
- `get_tools_instructions() -> list[str]`: Returns a list of strings explaining the toolbox's interface to the agent. 
- `do_next_action()`: Perform the next action of the agent using the chosen tool and its arguments. 
- `choose_next_tool(command: str)`: Parse the provided command to choose the next tool. 
- `use_tool(**kwargs) -> str`: Use a tool from the toolbox to perform an action on the context. 
- `check_transition_state()`: Decide if we should transition to a different agent state based on the context.