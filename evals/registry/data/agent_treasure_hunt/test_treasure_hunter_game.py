import unittest

from treasure_hunter_game import TreasureHunt

class TestTreasureHunt(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(ValueError):
            
            TreasureHunt(agent_position=(0, 0), treasure_position=(0, 0))

        with self.assertRaises(ValueError):
            TreasureHunt(agent_position=(-1, 0), treasure_position=(2, 2))

        with self.assertRaises(ValueError):
            TreasureHunt(agent_position=(0, 6), treasure_position=(2, 2))

        with self.assertRaises(ValueError):
            TreasureHunt(agent_position=(0, 0), treasure_position=(6, 5))

        th = TreasureHunt(agent_position=(0, 0), treasure_position=(2, 2))
        self.assertEqual(th.agent_position, (0, 0))
        self.assertEqual(th.treasure_position, (2, 2))

    def test_validate_position(self):
        th = TreasureHunt(agent_position=(0, 0), treasure_position=(2, 2))

        with self.assertRaises(ValueError):
            th.validate_position((0, 6))

        with self.assertRaises(ValueError):
            th.validate_position((6, 0))

        with self.assertRaises(ValueError):
            th.validate_position((-1, 0))

        with self.assertRaises(ValueError):
            th.validate_position((0, -1))

        with self.assertRaises(ValueError):
            th.validate_position((0, 0, 0))

        self.assertIsNone(th.validate_position((0, 0)))

    def test_cell_distance(self):
        self.assertEqual(TreasureHunt.cell_distance((0, 0), (2, 2)), 2)
        self.assertEqual(TreasureHunt.cell_distance((0, 0), (0, 2)), 1)
        self.assertEqual(TreasureHunt.cell_distance((2, 4), (4, 4)), 1)
        self.assertEqual(TreasureHunt.cell_distance((5, 5), (5, 0)), 2)

    def test_use_compass(self):
        th = TreasureHunt(agent_position=(0, 0), treasure_position=(2, 2))
        self.assertEqual(th.use_compass(), 'eastnorth')

        th.agent_position = (5, 5)
        self.assertEqual(th.use_compass(), 'westsouth')

        th.state = 'digging'
        self.assertEqual(th.use_compass(), "You can't use the compass in the digging state.")

    def test_move(self):
        th = TreasureHunt(agent_position=(0, 0), treasure_position=(2, 2))

        self.assertEqual(th.move('north'), "Moved north.")
        self.assertEqual(th.agent_position, (0, 1))

        self.assertEqual(th.move('westy'), "Invalid direction. Use 'north', 'south', 'east', or 'west'.")
        self.assertEqual(th.agent_position, (0, 1))

        th.state = 'digging'
        self.assertEqual(th.move('south'), "You can't move in the digging state.")

    def test_dig(self):
        th = TreasureHunt(agent_position=(0, 0), treasure_position=(2, 2))

        self.assertEqual(th.dig(), "You can't dig in the exploration state.")

        th.state = 'digging'
        self.assertEqual(th.dig(), "You dug but found nothing.")

        th.agent_position = (2, 2)
        self.assertEqual(th.dig(), "Congratulations! You found the treasure!")

if __name__ == '__main__':
    unittest.main()
