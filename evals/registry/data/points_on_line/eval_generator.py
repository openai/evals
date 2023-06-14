import json
import random

# Constants
COMP_MIN: int = -10
COMP_MAX: int = 10
N_DECIMALS: int = 2
EVAL_SAMPLES: int = 100
OUTPUT_FILE: str = "./line_coords.jsonl"


def tuple_to_string(float_tuple: tuple, n_decimals: int) -> str:
    """Convert a tuple of floats to a string with specified decimal places."""
    formatted_string = ", ".join("{:.{}f}".format(x, n_decimals) for x in float_tuple)
    return f"({formatted_string})"


def comp_float_mul(vector: tuple, n_decimals: int, factor: float = 0.1) -> tuple:
    """Applies a random multiplicative factor to each component of the vector.

    Parameters:
        vector: A tuple of three floats.
        factor: A float representing the multiplicative factor.
        n_decimals: The number of decimal places to use in the output vector.

    Returns:
        A tuple of three floats, each component is the original component multiplied by
        the factor and rounded to the specified number of decimal places.
    """
    new_vector = (
        float(vector[0]) + random.uniform(-factor, factor),
        float(vector[1]) + random.uniform(-factor, factor),
        float(vector[2]) + random.uniform(-factor, factor),
    )
    new_vector = (
        round(new_vector[0], n_decimals),
        round(new_vector[1], n_decimals),
        round(new_vector[2], n_decimals),
    )
    return new_vector


def random_divisible_line(comp_min: int, comp_max: int, n_decimals: int) -> (str, str, str):
    """Generates three points on a straight line in 3D space.

    The 'line_start' variable represents the position that a line begins from.
    After the start point is generated, the 'line_add' variable represents the
    mid point movement away from the start point.

    Parameters:
        comp_min: The minimum start position for any component of the points.
        comp_max: The maximum start position for any component of the points.
        n_decimals: The number of decimal places to use in the output points.

    Returns:
        A tuple of three strings. Each string represents a point on the line.
    """
    line_start = (
        random.randint(comp_min, comp_max),
        random.randint(comp_min, comp_max),
        random.randint(comp_min, comp_max),
    )

    line_add = (
        random.randint(comp_min, comp_max),
        random.randint(comp_min, comp_max),
        random.randint(comp_min, comp_max),
    )

    line_start = comp_float_mul(line_start, n_decimals=n_decimals)
    line_add = comp_float_mul(line_add, n_decimals=n_decimals)

    line_center = (
        line_start[0] + (line_add[0]),
        line_start[1] + (line_add[1]),
        line_start[2] + (line_add[2]),
    )

    line_end = (
        line_center[0] + (line_add[0]),
        line_center[1] + (line_add[1]),
        line_center[2] + (line_add[2]),
    )

    line_start = tuple_to_string(line_start, n_decimals)
    line_center = tuple_to_string(line_center, n_decimals)
    line_end = tuple_to_string(line_end, n_decimals)

    return line_start, line_center, line_end


def construct_messages(start: tuple, end: tuple) -> list[dict]:
    """Constructs the input messages for the line midpoint calculation task."""
    system_msg = {
        "role": "system",
        "content": "You will be provided with the end points of a line in 3 dimensions. Please calculate and return only the midpoint of this line, in this format: (x, y, z)",
    }
    user_msg = {"role": "user", "content": f"{start}, {end}"}
    return [system_msg, user_msg]


def assemble_test_format(n_samples: int) -> list[dict]:
    """Generates the test format for the line midpoint calculation task.

    Parameters:
        n_samples: The number of eval json entries to generate.

    Returns:
        A list of dictionaries. Each dictionary represents an eval.
    """
    results = []
    for i in range(n_samples):
        start, center, end = random_divisible_line(
            comp_min=COMP_MIN, comp_max=COMP_MAX, n_decimals=N_DECIMALS
        )
        result = {
            "input": construct_messages(start, end),
            "ideal": f"{center}",
        }
        results.append(result)
    return results


if __name__ == "__main__":
    dict_entries = assemble_test_format(n_samples=EVAL_SAMPLES)

    with open(OUTPUT_FILE, "w") as writefile:
        for entry in dict_entries:
            writefile.write(json.dumps(entry) + "\n")
