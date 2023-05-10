"""
This module provides an interface to create an agent that can take actions based on a provided context. The agent's state and available tools change depending on the context. This module provides abstract classes for defining the agent and its tools and states.
"""
from abc import ABC, abstractmethod
from typing import Optional
import copy
from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import ChatCompletionPrompt
from evals.record import record_sampling
from evals.registry import Registry

#######################################################
# Declaration of interfaces
#######################################################

# An agent is provided a context. 
# Depending on the context, the agent will transition to a new AgentState.
# Each AgentState has Tools where each tool represents a possible action the agent can take in the
# given context. 
# The agent chooses a tool based on the context and provides the necessary arguments. 
# The action is taken which modifies the context.

class AgentContext(ABC):
    """
    Class representing the context of an agent.
    """
    @abstractmethod
    def get_context_string(self):
        pass
    
    
class IAgentState(ABC):
    """Purely abstract interface for agent state to break cyclic dependency between Agent and AgentState."""
    
    @abstractmethod      
    def do_next_action(self):
        """Perform the next action for the chosen tool.
        """
    @abstractmethod
    def get_tools_instructions(self) -> list[str]:
        """get a list of the instructions for the available tools in this state."""

    @abstractmethod
    def choose_next_tool(self, command:str):
        """Parse the provided command to choose the next tool.

        Args:
            command (str): _description_
        """

    @abstractmethod
    def check_transition_state(self):
        """Decide if we should transition to a different agent state based on the context.

        Returns:
            _type_: _description_
        """
        
class Tool(ABC):
    """
    Abstract base class for defining the interface for a tool.
    """

    def __init__(self, agent_state: IAgentState):
        self.agent_state = agent_state

    @abstractmethod
    def get_tool_instructions(self) -> str:
        """
        Returns a string explaining the tool's interface to the agent.
        """

    @abstractmethod
    def use(self, **kwargs) -> str:
        """
        Uses the tool to perform an action on the context.
        """

class Agent(ABC):
    """
    Class representing an agent.
    """

    def __init__(self, agent_context:AgentContext):
        self.agent_context = agent_context
        self.agent_state: Optional[AgentState] = None

    def do_next_action(self):
        """
        Performs the next action of the agent using the chosen tool and its arguments.
        """
        self.agent_state.do_next_action()
        self.update_context()
        self.agent_state.check_transition_state()
        
    @abstractmethod
    def update_context(self):
        """Updates the AgentContext after an action has been taken.
        """

class AgentState(IAgentState):
    """
    Abstract base class representing the state of an agent.
    """

    def __init__(self, agent: Agent):
        self.last_action = ""
        self.agent = agent
        self.chosen_tool = None
        self.chosen_tool_kwargs = {}
        self.tool_response = ""
        self.tools = []

    def get_tools_instructions(self) -> list[str]:
        """
        Returns a list of strings explaining the toolbox's interface to the agent.
        """
        instructions = []
        for tool in self.tools:        
            instructions.append(tool.get_tool_instructions())
        return instructions
    
    def do_next_action(self):
        """
        Performs the next action of the agent using the chosen tool and its arguments.
        """
        if self.chosen_tool is None:
            self.tool_response = "Invalid action. Please choose a valid action."
            return
        
        self.tool_response = self.chosen_tool.use(**self.chosen_tool_kwargs)
        
    @abstractmethod
    def choose_next_tool(self, command:str):
        """Parse the command provided by the agent to choose the next tool.

        Args:
            command (str): _description_
        """

    @abstractmethod
    def check_transition_state(self):
        """Decide if we should transition to a different agent state based on the context.

        Returns:
            _type_: _description_
        """

DEFAULT_AGENT_TEMPLATE = """
You are playing a game where your goal is to find the treasure.

The map grid goes from 0 to 5 in X and Y.
The map is divided in to 2x2 cells. When you enter the digging state, the treasure is in your current cell.

Below, you will find yaml formatted context of the current state of the game after triple hashtags.

Your task is to reason about your next choice in a step by step manner, and then choose a next action from "6 Available Next Actions"
the form of the items in this list is: "syntax"-"description"

Hint: after moving, use the compass to find the treasure until you are in the same cell as the treasure. Then dig in all the squares of the cell.

Examples of action syntax:
COMPASS
MOVE N
DIG

The format of your response is shown below surrounded by <<<angle brackets>>>:
<<<
Reasoning:
Your reasoning about choosing your next action in a step by step manner

Choice:
Your choice an action using the syntax for your choice with no explanation.
>>>

Context:
###

"""

class AgentCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class AgentCompletionFn(CompletionFn):
    def __init__(
        self,
        agent_template: str = DEFAULT_AGENT_TEMPLATE,
        completion_fn: str = None,
        max_agent_steps : int = 25,
        registry: Registry = None,
        registry_path: str = None,
        **kwargs
    ) -> None:
        
        self.registry = Registry() if not registry else registry
        
        if registry_path:
            self.registry.add_registry_paths(registry_path)

        self.completion_fn = "gpt-3.5-turbo" if not completion_fn else completion_fn

        self.agent_template = agent_template
        
        
        self.agent : Agent = None
        self.agent_context : AgentContext = None
        self.max_agent_steps = max_agent_steps
        self.current_agent_step = 0
        print("CompletionFnMade")

    def setup_agent(self,prompt):
        pass

    def check_for_end_context(self) -> bool:
        return  False

    def __call__(self, prompt, **kwargs) -> AgentCompletionResult:
        # Ensure it is in string format
        
        completion_fn_instance = self.registry.make_completion_fn(self.completion_fn)
        prompt = ChatCompletionPrompt(prompt).to_formatted_prompt()

        self_copy = copy.deepcopy(self)
        self_copy.setup_agent(prompt)
        
        is_agent_done = False
        while self_copy.current_agent_step < self.max_agent_steps:
            
            context = self_copy.agent_context.get_context_string()
            agent_prompt = [{"role": "system", "content": "you are a helpful gamer"},{"role": "user", "content": self_copy.agent_template + context}]
            
            answer = completion_fn_instance(prompt=agent_prompt, **kwargs).get_completions()[0]
            record_sampling(prompt=agent_prompt, sampled=answer)

            action = answer.split("\n")[-1]
            self_copy.agent.agent_state.choose_next_tool(action)
            
            self_copy.agent.do_next_action()
            self_copy.current_agent_step += 1
            
            is_agent_done = self_copy.check_for_end_context()
            if is_agent_done:
                break

        if is_agent_done:
            completion_result_str = "Victory"
            
        else:
            completion_result_str = "Defeat"

        return AgentCompletionResult(completion_result_str)
    
