import json

from solver import solve_diagram
from wave_function_collapse import ContradictionException, generate_collapsed_wave, print_wave

samples = [
    [3, 3],
    [4, 4],
    [4, 4],
    [4, 4],
    [4, 4],
    [5, 5],
    [5, 5],
    [5, 5],
    [5, 5],
    [6, 6],
    [6, 6],
    [7, 7],
    [8, 8],
    [9, 9],
    [10, 10],
]

PROMPT = """Below is a diagram of a small, simple physics simulator.
Here's the ruleset for the physics engine:
The direction of up is the first line. The direction of down is represented by the last line.
The ball, represented by ◦ will fall down due to gravity. 
The ball begins with only downward momentum.
The right ramp, represented by ◣ will cause the ball to roll 1) down to its own level and 2) an indefinite number of spaces to the right until it is stopped by another object.
The left ramp, represented by ◢ will cause the ball to roll 1) down to its own level and 2) an indefinite number of spaces to the left until it is stopped by another object.
The ball will continue rolling left or right even after it falls one or more levels.
If the ball should be both falling and traveling left or right, it will always fall downward though its horizontal momentum will be converted to the next level.
If a ball is traveling horizontally while it encounters a ramp, it will come to a stop. It will not roll up any ramp.
The block, represented by ■, prevents the ball from falling further down, or from rolling further left or right.
The ball can fall off a block to a lower level.
The walls and floor are also represented by ■ characters.
Air is represented by the ▯ character. It has no effect on the movement of the ball.
The ball will always come to rest in an air space. It will not take the place of any other object.
Friction and air resistance are not present in this simulator.

Try reasoning through the movement of the ball step by step.
Provide your answer by reproducing the full, final state of the simulation using the character key provided.
"""


def create_row(initial_state, ideal_state):
    return {
        "input": [
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": "Where will the ball come to a rest in the following diagram?\n"
                + initial_state,
            },
        ],
        "ideal": ideal_state,
    }


def format_wave_as_string(wave):
    out = ""
    for i in range(len(wave)):
        for j in range(len(wave[i])):
            out += wave[i][j][0]
        out += "\n"
    return out


def generate_samples():
    with open("samples.jsonl", "w") as outfile:
        for height, width in samples:
            # Just retry generating until you get a fully collapsed tileset
            while True:
                try:
                    initial_diagram = generate_collapsed_wave(height, width)
                    print_wave(initial_diagram)
                    initial_state = format_wave_as_string(initial_diagram)

                    solved_diagram = solve_diagram(initial_diagram)
                    print_wave(solved_diagram)
                    ideal_state = format_wave_as_string(solved_diagram)

                    sample = create_row(initial_state, ideal_state)
                    json.dump(sample, outfile)
                    outfile.write("\n")

                    break
                except ContradictionException:
                    pass

            print("")


generate_samples()
