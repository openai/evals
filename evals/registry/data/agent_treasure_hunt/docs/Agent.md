oaieval treasure-hunt agent-treasure-hunt
# Agent Completion Function Module

This module provides interfaces to create a completion function and an agent that can take actions based on a provided context. The agent's state and available tools change depending on the context. This module provides abstract classes for defining the agent and its tools and states. 
## Interfaces

This module defines several abstract classes to define the interfaces for agents, states, and tools.
## `AgentContext`

This abstract class represents the context of an agent. It defines a single abstract method:
#### `get_context_string() -> str`

This method returns the string representation of the context.
## `IAgentState`

This abstract class represents a state of the agent. It defines the following abstract methods:
#### `do_next_action()`

This method performs the next action for the chosen tool.
#### `get_tools_instructions() -> list[str]`

This method returns a list of instructions for the available tools in this state.
#### `choose_next_tool(command:str)`

This method parses the provided command to choose the next tool.
#### `check_transition_state() -> _type_`

This method decides if we should transition to a different agent state based on the context.
## `Tool`

This abstract class represents a tool in the agent's toolbox. It defines the following abstract methods:
#### `get_tool_instructions() -> str`

This method returns a string explaining the tool's interface to the agent.
#### `use(**kwargs) -> str`

This method uses the tool to perform an action on the context.
## `Agent`

This abstract class represents an agent. It defines the following method:
#### `do_next_action()`

This method performs the next action of the agent using the chosen tool and its arguments.
#### `update_context()`

This method updates the `AgentContext` after an action has been taken.
## `AgentState`

This base class represents the state of an agent. It defines the following abstract methods:
#### `get_tools_instructions() -> list[str]`

This method returns a list of strings explaining the toolbox's interface to the agent.
#### `choose_next_tool(command:str)`

This method parses the command provided by the agent to choose the next tool.
#### `check_transition_state() -> _type_`

This method decides if we should transition to a different agent state based on the context.
## `AgentCompletionResult` Class

This class is a subclass of `CompletionResult` that returns the response from the agent as a single item list. It has the following method:
#### `get_completions() -> list[str]`

This method returns a list with the response from the agent as a single item.
## `AgentCompletionFn` Class

This class is a subclass of `CompletionFn` and is used to create an agent that can take actions based on a provided context. It has the following methods:
### `__init__(self, agent_template: str = DEFAULT_AGENT_TEMPLATE, completion_fn: str = None, max_agent_steps : int = 25, registry: Registry = None, registry_path: str = None, **kwargs) -> None:`

This method initializes the `AgentCompletionFn` object with the given parameters.
### `setup_agent(self,prompt)`

This method sets up the agent for the given prompt.
### `check_for_end_context() -> bool:`

This method checks if the agent has reached the end of the context.
### `__call__(self, prompt, **kwargs) -> AgentCompletionResult:`

This method takes in a prompt and returns the response of the agent as an instance of `AgentCompletionResult`. It creates an instance of the chosen `CompletionFn` and uses it to generate the agent's response. It also keeps track of the current step in the agent's actions and checks if the agent
