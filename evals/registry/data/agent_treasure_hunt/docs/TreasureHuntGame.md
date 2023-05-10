# TreasureHunt Module Documentation

The `TreasureHunt` module provides a class that represents a treasure hunt game. The game involves an agent searching for a treasure in a grid of cells. The agent can move in four directions: north, south, east, and west. The agent can also use a compass to determine the direction of the treasure. Once the agent reaches the cell with the treasure, they can dig to find the treasure.
# Class: TreasureHunt

This is the main class in the `TreasureHunt` module. It represents a treasure hunt game.
# Methods
## init (self, agent_position=(0, 0), treasure_position=(0, 0))

Initializes the game with the given agent and treasure positions. 
- `agent_position` (tuple): The starting position of the agent. 
- `treasure_position` (tuple): The position of the treasure.

Raises a `ValueError` if the agent and treasure start in the same cell or if the positions are out of bounds.
## validate_position(self, position)

Validates that the given position is valid. 
- `position` (tuple): The position to validate.

Raises a `ValueError` if the position is not a tuple with 2 elements or if the coordinates are out of bounds.
## cell_distance(pos1, pos2)

Computes the Manhattan distance between two positions. 
- `pos1` (tuple): The first position. 
- `pos2` (tuple): The second position.

Returns the Manhattan distance between the two positions.
## use_compass(self)

Uses the compass to determine the direction of the treasure.

Returns the direction of the treasure relative to the agent.
## move(self, direction)

Moves the agent in the given direction. 
- `direction` (str): The direction to move in.

Returns a message indicating the result of the move.

Raises a `ValueError` if an invalid direction is given.
## dig(self)

Digs at the agent's current position.

Returns a message indicating the result of the dig.
# Attributes
## agent_position

The current position of the agent.
## treasure_position

The position of the treasure.
## state

The state of the game. Can be 'exploration', 'digging', or 'game_over'.
# Example Usage

```python

from TreasureHunt import TreasureHunt

# Create a new game with the agent starting at (0, 0) and the treasure at (3, 3)
game = TreasureHunt(agent_position=(0, 0), treasure_position=(3, 3))

# Move the agent to the east
result = game.move('east')

# Use the compass to determine the direction of the treasure
direction = game.use_compass()

# Dig at the agent's current position
result = game.dig()
```
