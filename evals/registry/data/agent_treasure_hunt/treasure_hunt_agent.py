from evals.completion_fns.agent import AgentContext, AgentState, Agent, Tool
from treasure_hunter_game import TreasureHunt
import yaml 

class TreasureHuntAgentContext(AgentContext):
    
    def __init__(self):
        self.steps_taken = 0
        self.current_coord = (0,0)
        self.coord_history = []
        
        self.state = "Exploring"
        self.tool_directions = []
        self.last_tool_response = ""
        
    def get_context_string(self):
        """Override abstract get_content_string"""
        data = {}
        data['Current State'] = self.state
        data['Steps Taken'] = self.steps_taken
        data['Current Coordinate'] = f"X: {self.current_coord[0]} Y: {self.current_coord[1]}"
        if len(self.coord_history) > 0:
            data['History'] = [f"Step: {c[0]} - X: {c[1]} Y: {c[2]} -> Action: {c[3]} -> Response: {c[4]}" for c in self.coord_history]
        data['Last Action Feedback'] = self.last_tool_response
        data['Available Next Actions'] = self.tool_directions
        
        yaml_str = yaml.safe_dump(data)
        
        return yaml_str


class DigTool(Tool):
    """
    A tool to dig at the current location in a treasure hunt game.
    """

    def __init__(self,agent_state:AgentState, game: TreasureHunt):
        super().__init__(agent_state)
        
        self.game = game
        
    def get_tool_instructions(self) -> str:
        return "DIG: Dig at the current location"

    def use(self, **kwargs) -> str:
        return self.game.dig()
        
class MoveTool(Tool):
    """
    A tool to dig at the current location in a treasure hunt game.
    """

    def __init__(self,agent_state:AgentState, game: TreasureHunt):
        super().__init__(agent_state)
        self.game = game
        
        
    def get_tool_instructions(self) -> str:
        return "MOVE (N|S|E|W): Move one step in the chosen direction. You cannot move on diagonals."

    def use(self, **kwargs):
        """
        Uses the tool to perform an action on the context.
        """
        return self.game.move(kwargs['direction'])
        
class CompassTool(Tool):
    """
    A tool that points to the treasure in a treasure hunt game.
    """

    def __init__(self,agent_state:AgentState, game: TreasureHunt):
        super().__init__(agent_state)
        
        self.game = game
          
    def get_tool_instructions(self) -> str:
        return "COMPASS: 'last action feedback' will show the direction of the treasure relative to your current position"

    def use(self, **kwargs):
        """
        Uses the tool to perform an action on the context.
        """
        return self.game.use_compass()
            
class ExplorationAgentState(AgentState):
    """
    Class representing a toolbox containing a set of tools.
    """

    def __init__(self, agent:Agent, game: TreasureHunt):
        super().__init__(agent)
        
        self.game = game
        
        self.compass_tool = CompassTool(self,game)
        self.move_tool = MoveTool(self,game)
        
        self.tools.append(self.compass_tool)
        self.tools.append(self.move_tool)
        

    def choose_next_tool(self, command:str):
        """Parse the provided command to choose the next tool.

        Args:
            command (str): _description_
        """
        self.last_action = command
        words = command.split()
        if len(words) == 0:
            return
        
        if words[0] == 'COMPASS':
            self.chosen_tool = self.compass_tool
        elif words[0] == 'MOVE':
            if len(words) < 2:
                return 
            self.chosen_tool = self.move_tool
            direction = words[1]
            if direction == 'N':
                self.chosen_tool_kwargs = {'direction': 'north'}
            elif direction == 'S':
                self.chosen_tool_kwargs = {'direction': 'south'}
            elif direction == 'E':
                self.chosen_tool_kwargs = {'direction': 'east'}
            elif direction == 'W':
                self.chosen_tool_kwargs = {'direction': 'west'}
        else:
            self.chosen_tool = None


    def check_transition_state(self):
        """Decide if we should transition to a different agent state based on the context.

        Returns:
            _type_: _description_
        """
        if self.agent.agent_context.state == "digging":
            new_state = DiggingAgentState(self.agent, self.game)
            
            self.agent.agent_state = new_state
            self.agent.update_context()
        
class DiggingAgentState(AgentState):
    """
    Class representing a toolbox containing a set of tools.
    """

    def __init__(self, agent: Agent, game: TreasureHunt):
        super().__init__(agent)
        
        self.game = game
        
        self.dig_tool = DigTool(self,game)
        self.move_tool = MoveTool(self,game)
        
        self.tools.append(self.dig_tool)
        self.tools.append(self.move_tool)

    def choose_next_tool(self, command:str):
        self.last_action = command
        words = command.split()
        if words[0] == 'DIG':
            self.chosen_tool = self.dig_tool
        elif words[0] == 'MOVE':
            self.chosen_tool = self.move_tool
            direction = words[1]
            if direction == 'N':
                self.chosen_tool_kwargs = {'direction': 'north'}
            elif direction == 'S':
                self.chosen_tool_kwargs = {'direction': 'south'}
            elif direction == 'E':
                self.chosen_tool_kwargs = {'direction': 'east'}
            elif direction == 'W':
                self.chosen_tool_kwargs = {'direction': 'west'}
        else: 
            self.chosen_tool == None

    def check_transition_state(self):
        """Decide if we should transition to a different agent state based on the context.  
        """
        if self.agent.agent_context.state == "exploration":
            new_state = ExplorationAgentState(self.agent, self.game)
            
            self.agent.agent_state = new_state
            self.agent.update_context()


class TreasureHuntAgent(Agent):
    """_summary_

    Args:
        Agent (_type_): _description_
    """

    def __init__(self, agent_context:TreasureHuntAgentContext, game: TreasureHunt):
        super().__init__(agent_context)
        self.game = game
        
        #initialize the starting state
        self.agent_state = ExplorationAgentState(self,self.game)
        
        self.last_step_for_history = None
        
        self.update_context()

        
               
    def update_context(self):
        """Updates the context after an action has been taken.
        """          
        self.agent_context.current_coord = self.game.agent_position

        self.agent_context.state = self.game.state
        
        self.agent_context.tool_directions = self.agent_state.get_tools_instructions()
        self.agent_context.last_tool_response = self.agent_state.tool_response
        
        if self.last_step_for_history != None:
            self.last_step_for_history += (self.agent_state.tool_response,)
        
            self.agent_context.coord_history.insert(0,self.last_step_for_history)
        
        if len(self.agent_context.coord_history) > 5:
            self.agent_context.coord_history.pop()
            
    def do_next_action(self):
                
        self.agent_context.steps_taken += 1
        self.last_step_for_history = (self.agent_context.steps_taken,)
        self.last_step_for_history += self.agent_context.current_coord
        self.last_step_for_history += (self.agent_state.last_action,)

        return super().do_next_action()
    
    
        
                
if __name__ == "__main__":
    
    treasure_hunt_game = TreasureHunt(agent_position=(0,0), treasure_position=(2,0))
    treasure_hunt_context = TreasureHuntAgentContext()
    treasure_hunt_agent = TreasureHuntAgent(treasure_hunt_context,treasure_hunt_game)
    
    while treasure_hunt_context.state != "game_over":
        
        print("Context: ")
        print(treasure_hunt_context.get_context_string())
    
        action = input("Choose an action from the context:")

        treasure_hunt_agent.agent_state.choose_next_tool(action)
        treasure_hunt_agent.do_next_action()
    
    
    print("Done!")