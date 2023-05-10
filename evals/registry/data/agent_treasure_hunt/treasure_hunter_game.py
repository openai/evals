class TreasureHunt:
    """
    Class representing a treasure hunt game.
    """

    def __init__(self, agent_position=(0, 0), treasure_position=(0, 0)):
        """
        Initializes the game with the given agent and treasure positions.

        Args:
            agent_position (tuple): The starting position of the agent.
            treasure_position (tuple): The position of the treasure.
        """
        self.validate_position(agent_position)
        self.validate_position(treasure_position)

        if self.cell_distance(agent_position, treasure_position) == 0:
            raise ValueError("Agent and treasure must start in different cells.")

        self.agent_position = agent_position
        self.treasure_position = treasure_position
        self.state = "exploration"

    def validate_position(self, position):
        """
        Validates that the given position is valid.

        Args:
            position (tuple): The position to validate.

        Raises:
            ValueError: If the position is not a tuple with 2 elements or if the coordinates are out of bounds.
        """
        if len(position) != 2:
            raise ValueError("Position must be a tuple with 2 elements.")

        x, y = position
        if not (0 <= x < 6 and 0 <= y < 6):
            raise ValueError("Position must have coordinates between 0 and 5 inclusive.")

    @staticmethod
    def cell_distance(pos1, pos2):
        """
        Computes the Manhattan distance between two positions.

        Args:
            pos1 (tuple): The first position.
            pos2 (tuple): The second position.

        Returns:
            int: The Manhattan distance between the two positions.
        """
        cell1 = (pos1[0] // 2, pos1[1] // 2)
        cell2 = (pos2[0] // 2, pos2[1] // 2)
        return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])

    def use_compass(self):
        """
        Uses the compass to determine the direction of the treasure.

        Returns:
            str: The direction of the treasure relative to the agent.
        """
        if self.state == "exploration":
            dx, dy = (
                self.treasure_position[0] - self.agent_position[0],
                self.treasure_position[1] - self.agent_position[1],
            )
            direction = ""
            if dx > 0:
                direction = "east"
            elif dx < 0:
                direction = "west"
            if dy > 0:
                direction = "north"
            elif dy < 0:
                direction = "south"
            return direction
        else:
            return "Can't use the compass in the digging state."

    def move(self, direction):
        """
        Moves the agent in the given direction.

        Args:
            direction (str): The direction to move in.

        Returns:
            str: A message indicating the result of the move.
        """

        x, y = self.agent_position
        if direction == "north":
            y = min(y + 1, 5)
        elif direction == "south":
            y = max(y - 1, 0)
        elif direction == "east":
            x = min(x + 1, 5)
        elif direction == "west":
            x = max(x - 1, 0)
        else:
            return "Invalid direction. Use 'N, 'S', 'E', or 'W'."

        self.agent_position = (x, y)

        if self.cell_distance(self.agent_position, self.treasure_position) == 0:
            self.state = "digging"
            return f"Moved {direction}. Treasure in this cell"
        else:
            if self.state == "digging":
                self.state = "exploration"
                return f"Moved {direction}. You left the cell with treasure"

        return f"Moved {direction}."

    def dig(self):
        """
        Digs at the agent's current position.

        Returns:
            str: A message indicating the result of the dig.
        """
        if self.state == "digging":
            if self.agent_position == self.treasure_position:
                self.state = "game_over"
                return "Congratulations! You found the treasure!"
            return "You dug but found nothing. Try a different spot."
        else:
            return "You can't dig in the exploration state."
