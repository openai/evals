# `TreasureHunt` 
class representing a simple game of finding treasure on a 6x6 grid. It has the following properties and methods:

Properties: 
- `agent_position`: a tuple representing the current position of the agent. 
- `treasure_position`: a tuple representing the position of the treasure. 
- `state`: a string representing the state of the game, either "exploration" or "digging".

Methods: 
- `__init__(self, agent_position=(0, 0), treasure_position=(0, 0))`: Initializes the game with the given agent and treasure positions. 
- `validate_position(self, position)`: Validates that the given position is valid. 
- `cell_distance(pos1, pos2)`: Computes the Manhattan distance between two positions. 
- `use_compass(self)`: Uses the compass to determine the direction of the treasure. 
- `move(self, direction)`: Moves the agent in the given direction. 
- `dig(self)`: Digs at the agent's current position.