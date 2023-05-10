from evals.completion_fns.agent import AgentCompletionFn
from treasure_hunt_agent import TreasureHuntAgent, TreasureHuntAgentContext
from treasure_hunter_game import TreasureHunt

class TreasureHuntAgentCompletionFn(AgentCompletionFn):
    
    def check_for_end_context(self) -> bool:
        print(self.agent_context.get_context_string())
        print(self.agent_context.state)
        return self.agent_context.state == "game_over"

    def setup_agent(self,prompt):
        #setup agent must assign agent_context and agent.
        self.treasure_hunt_game = TreasureHunt(agent_position=(0,0), treasure_position=(2,0))
        
        self.agent_context = TreasureHuntAgentContext()
        self.agent = TreasureHuntAgent(self.agent_context,self.treasure_hunt_game)
    


if __name__=="__main__":
    
    treasure_hunt_completion_fn = TreasureHuntAgentCompletionFn()
    treasure_hunt_completion_fn.__call__("Respond to me!")