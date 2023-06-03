"""
Module: NxN Maze Evaluation Generator

Module Overview:
This module is responsible for generating random mazes, finding solutions with the shortest path algorithm,
and exporting these maze examples to JSON-line formatted files, primarily for use in OpenAI's evaluation platform.
The mazes consist of four key elements: pathway (0), wall (1), start position (2), and exit (3).
The primary objective of this script is to generate a multitude of maze examples,
aiding in the evaluation of GPT-3's and GPT-4's proficiency in 2D spatial reasoning via maze solving.

The generated maze examples are simultaneously formatted and added to two distinct JSON-line formatted files.
One file contains examples assessing the correct initial move to solve the maze,
while the other focuses on the entire sequence of steps required to reach the exit.

Maze generation employs a recursive backtracker, ensuring a solvable maze is generated
every time without extensive computational demands during generation.

Example JSON Line Item "Pairs":
1.  Single Step Evaluation - 3x3-mazes-singlemove.jsonl:
    {"input": [{"role": "system", "content": "Task: You will be given a maze, determine the shortest sequence of moves
     to reach the end without walking through walls. Respond with only the first move and refrain from providing
      additional information. The maze elements are: pathways (0), walls (1), your position (2), and exit (3).
       Moves outside the maze are deemed invalid. Valid moves: [\"right\", \"left\", \"up\", \"down\"]"},
        {"role": "user", "content": "[0,0,3]\n[1,1,0]\n[2,0,0]\n"}], "ideal": "right"}

2. Multi-Step Evaluation - 3x3-mazes.jsonl:
    {"input": [{"role": "system", "content": "Task: You will be given a maze, determine the shortest sequence of moves
                to reach the end without walking through walls. Respond only with a comma-separated list of moves
                (Example output: down, right, up, left, left) and avoid providing additional information.
                Maze elements are: pathways (0), walls (1), your position (2), and exit (3).
                Moves outside the maze are deemed invalid. Valid moves: [\"right\", \"left\", \"up\", \"down\"]"},
                {"role": "user", "content": "[0,0,3]\n[1,1,0]\n[2,0,0]\n"}], "ideal": "right, right, up, up"}

Module Functions:
    recursive_backtracker: Implements the recursive backtracker algorithm for maze generation.
    generate_maze: Produces a maze using the recursive backtracker algorithm.
    random_outer_pos: Retrieves a random position on the maze's outer edge.
    generate_start_end: Determines random start and end positions on the maze's outer boundary.
    build_graph: Constructs a graph representation of a given maze.
    generate_example_files: Produces a defined number of maze examples and exports them to JSON-line formatted files.
    create_example: Crafts a single maze example.
    create_move_files: Develops JSON-line formatted files featuring the moves necessary to solve the maze.
    plot_maze: Illustrates a maze using matplotlib.
"""

import concurrent.futures
import json
import random
import threading
import os
from numpy import ndarray
from tqdm import tqdm
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Union


# Dictionary mapping relative maze positions to their corresponding move names
move_map = {(1, 0): "down", (-1, 0): "up", (0, 1): "right", (0, -1): "left"}

# Define custom types for better readability
ProgressBarItem = Dict[str, Union[str, List[Dict[str, str]]]]
ProgressUpdate = Tuple[ProgressBarItem, ProgressBarItem, ndarray]


def recursive_backtracker(maze: np.ndarray, pos: Tuple[int, int]) -> None:
    """
    Recursive backtracker algorithm for maze generation.

    Args:
        maze (np.ndarray): The maze represented as a NumPy array.
        pos (Tuple[int, int]): Current position in the maze.
    """
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    random.shuffle(directions)
    for direction in directions:
        new_pos = pos[0] + direction[0] * 2, pos[1] + direction[1] * 2
        if 1 <= new_pos[0] < maze.shape[0] - 1 and 1 <= new_pos[1] < maze.shape[1] - 1:
            if maze[new_pos] == 1:
                maze[tuple(np.add(pos, direction))] = 0
                maze[new_pos] = 0
                recursive_backtracker(maze, new_pos)


def generate_maze(width: int, height: int) -> np.ndarray:
    """
    Generates a maze using the recursive backtracker algorithm.

    Args:
        width (int): The width of the maze.
        height (int): The height of the maze.

    Returns:
        np.ndarray: A NumPy array representing the generated maze.
    """
    maze = np.ones((height + 2, width + 2), dtype=np.uint8)
    start_pos = (1, 1)
    maze[start_pos] = 0
    recursive_backtracker(maze, start_pos)
    return maze[1:-1, 1:-1]


def random_outer_pos(maze: np.ndarray) -> Tuple[int, int]:
    """
    Returns a random position on the outer edge of the maze.

    Args:
        maze (np.ndarray): A NumPy array representing the maze.

    Returns:
        Tuple[int, int]: A tuple containing the random outer position's row and column indices.
    """
    height, width = maze.shape
    edge_pos = [
        (x, y)
        for x, row in enumerate(maze)
        for y, cell in enumerate(row)
        if (x in (0, width - 1) or y in (0, height - 1)) and cell == 0
    ]
    return random.choice(edge_pos)


def generate_start_end(maze: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Returns a random start and end position on the outer edge of the maze.

    Args:
        maze (np.ndarray): A NumPy array representing the maze.

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: A tuple containing the start and end positions.
    """
    start = random_outer_pos(maze)
    end = random_outer_pos(maze)
    while end == start:
        end = random_outer_pos(maze)
    return start, end


def build_graph(maze: np.ndarray) -> nx.Graph:
    """
    Builds a graph from a maze.

    Args:
        maze (np.ndarray): A NumPy array representing the maze.

    Returns:
        nx.Graph: A NetworkX graph object representing the maze.
    """
    graph = nx.Graph()
    for x, row in enumerate(maze):
        for y, cell in enumerate(row):
            if cell == 0:
                graph.add_node((x, y))
                if maze[x - 1, y] == 0:
                    graph.add_edge((x, y), (x - 1, y))
                if maze[x, y - 1] == 0:
                    graph.add_edge((x, y), (x, y - 1))
    return graph


def generate_example_files(
    num_examples: int,
    maze_width: int,
    maze_height: int,
    output_directory: str = ".",
    show_plot: bool = False,
    verbose: bool = False,
) -> None:
    """
    Generates a specified number of maze examples and exports them to JSON-line formatted files.

    Args:
        num_examples (int): Number of maze examples to generate.
        maze_width (int): The width of the maze.
        maze_height (int): The height of the maze.
        output_directory (str, optional): The directory to export the maze examples to. Defaults to ".".
        show_plot (bool, optional): Whether to display the maze plot using matplotlib. Defaults to False.
        verbose (bool, optional): Whether to print as threads finish. Defaults to False.
    """

    def create_example(progress_bar: tqdm) -> ProgressUpdate:
        """
        Create an example maze with a solution and update the progress bar.

        Generates a maze using provided width and height values, calculates the shortest path from
        the start to end positions, and returns a progress update containing the original maze,
        single move line, and all moves line.

        :param progress_bar: a tqdm progress bar instance to track progress
        :return: a tuple containing single move line, all moves line, and the generated maze
                 - single_move_line: a dictionary representing the maze with only the first move made
                 - all_moves_line: a dictionary representing the maze with all moves made
                 - maze: a 2D ndarray representing the generated maze
        """
        thread_name = threading.current_thread().name
        if verbose:
            print(f"Thread {thread_name} started")

        maze = generate_maze(maze_width, maze_height)
        start, end = generate_start_end(maze)

        graph = build_graph(maze)
        path = nx.shortest_path(graph, start, end)
        moves = [
            move_map[(x2 - x1, y2 - y1)] for (x1, y1), (x2, y2) in zip(path, path[1:])
        ]

        maze[start] = 2
        maze[end] = 3

        maze_repr = (
            "\n".join(["[" + ",".join(str(cell) for cell in row) + "]" for row in maze])
            + "\n"
        )

        single_move_line = create_move_line(maze_repr, moves, first_move_only=True)
        all_moves_line = create_move_line(maze_repr, moves, first_move_only=False)
        if verbose:
            print(f"Thread {thread_name} completed")

        with progress_bar.get_lock():
            progress_bar.update(1)

        return single_move_line, all_moves_line, maze

    single_move_filename = os.path.join(
        output_directory, f"{maze_width}x{maze_height}-mazes-singlemove.jsonl"
    )
    all_moves_filename = os.path.join(
        output_directory, f"{maze_width}x{maze_height}-mazes.jsonl"
    )

    with open(single_move_filename, "w") as single_move_file, open(
        all_moves_filename, "w"
    ) as all_moves_file:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            with tqdm(total=num_examples) as progress_bar:
                futures = [
                    executor.submit(create_example, progress_bar)
                    for _ in range(num_examples)
                ]

                for future in concurrent.futures.as_completed(futures):
                    single_move_line, all_moves_line, maze = future.result()
                    single_move_file.write(json.dumps(single_move_line) + "\n")
                    all_moves_file.write(json.dumps(all_moves_line) + "\n")

    if show_plot:
        for future in concurrent.futures.as_completed(futures):
            _, _, maze = future.result()
            plot_maze(maze)


def create_move_line(
    maze_repr: str, moves: List[str], first_move_only: bool
) -> Dict[str, Union[List[Dict[str, str]], str]]:
    """
    Creates a JSON-line formatted line for a maze example.

    Args:
        maze_repr (str): String representation of the maze.
        moves (List[str]): List of moves in the shortest path.
        first_move_only (bool): Whether to return only the first move or all moves.

    Returns:
        Dict[str, Union[List[Dict[str, str]], str]]: A dictionary containing the move line.
    """
    task_description = (
        f"Task: You will be given a maze, determine the shortest sequence of"
        f" moves to reach the end without walking through walls. Respond only with the "
        f"{'first move' if first_move_only else 'a comma seperated list of moves (example_output:down, right, up, left, left)'} "
        f"and do not include any additional information. The maze elements are:"
        f" pathways (0), walls (1), your position (2), and exit (3). "
        f'Moves that go outside the maze are illegal. valid_moves:["right", "left", "up", "down"]'
    )

    return {
        "input": [
            {"role": "system", "content": task_description},
            {"role": "user", "content": maze_repr},
        ],
        "ideal": (moves[0] if first_move_only else ", ".join(moves))
        if moves
        else "No valid path",
    }


def plot_maze(maze: np.ndarray) -> None:
    """
    Plots a maze using matplotlib.

    Args:
        maze (np.ndarray): A NumPy array representing the maze.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap

    # Set custom colormap for maze
    path_color = "white"
    wall_color = "black"
    start_color = "blue"
    end_color = "green"
    cmap = ListedColormap([path_color, wall_color, start_color, end_color])

    # Create legend
    legend_elements = [
        Patch(facecolor=path_color, edgecolor="black", label="Path"),
        Patch(facecolor=wall_color, edgecolor="black", label="Wall"),
        Patch(facecolor=start_color, edgecolor="black", label="Start"),
        Patch(facecolor=end_color, edgecolor="black", label="End"),
    ]

    plt.imshow(maze, cmap=cmap)
    plt.legend(handles=legend_elements)
    plt.show()


if __name__ == "__main__":
    num_examples = 2000
    maze_width = 5
    maze_height = 5
    output_directory = ""
    show_plot = False
    verbose = False
    generate_example_files(
        num_examples,
        maze_width,
        maze_height,
        output_directory=output_directory,
        show_plot=show_plot,
        verbose=verbose,
    )
