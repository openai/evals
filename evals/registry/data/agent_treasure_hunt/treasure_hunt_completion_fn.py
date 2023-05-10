from evals.completion_fns.agent import AgentCompletionFn
from treasure_hunt_agent import TreasureHuntAgent, TreasureHuntAgentContext
from treasure_hunter_game import TreasureHunt
import re

def parse_positions(s):
    coordinates = {}
    for item in s.split():
        if item.isalpha():
            current_key = item
        else:
            current_value = tuple(map(int, item.strip('()').split(',')))
            coordinates[current_key] = current_value
    return coordinates


class TreasureHuntAgentCompletionFn(AgentCompletionFn):
    
    def check_for_end_context(self) -> bool:
        return self.agent_context.state == "game_over"

    def setup_agent(self,prompt):
        #setup agent must assign agent_context and agent.
        
        positions = parse_positions(prompt[0]['content'])
        
        self.treasure_hunt_game = TreasureHunt(agent_position=positions['A'], treasure_position=positions['T'])
        
        self.agent_context = TreasureHuntAgentContext()
        self.agent = TreasureHuntAgent(self.agent_context,self.treasure_hunt_game)
    


if __name__=="__main__":
    
    treasure_hunt_completion_fn = TreasureHuntAgentCompletionFn()
    treasure_hunt_completion_fn.__call__("Respond to me!")