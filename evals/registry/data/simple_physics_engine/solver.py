from wave_function_collapse import (
    AIR,
    BALL,
    BLOCK,
    LEFT_RAMP,
    RIGHT_RAMP,
    get_below_tile,
    get_left_tile,
    get_right_tile,
)

DOWN = "down"
LEFT = "left"
RIGHT = "right"


def solve_diagram(diagram):
    # Track the direction of the movement of the ball
    ball_direction = DOWN
    ball_position = [0, 0]
    ball_can_move = True

    # Find the initial j position of the ball
    for j in range(len(diagram[0])):
        if diagram[0][j][0] == BALL:
            ball_position[1] = j

    # Write over the initial ball position with AIR
    diagram[ball_position[0]][ball_position[1]] = [AIR]

    while ball_can_move:
        i, j = ball_position
        if ball_direction == DOWN:
            below_shape = get_below_tile(diagram, i, j)[0]
            if below_shape == AIR:
                i += 1
            elif below_shape == LEFT_RAMP:
                i += 1
                j -= 1
                ball_direction = LEFT
            elif below_shape == RIGHT_RAMP:
                i += 1
                j += 1
                ball_direction = RIGHT
            elif below_shape == BLOCK:
                ball_can_move = False
            else:
                print("Unhandled case in DOWN")

        elif ball_direction == LEFT:
            left_shape = get_left_tile(diagram, i, j)[0]
            if left_shape == AIR:
                below_shape = get_below_tile(diagram, i, j)[0]
                if below_shape == AIR:
                    # We always precedence descent, so we have to handle multiple drops at once.
                    tiles_to_drop = 1
                    [i + 1, j]
                    while get_below_tile(diagram, i + tiles_to_drop, j)[0] == AIR:
                        tiles_to_drop += 1
                    i += tiles_to_drop
                    if get_left_tile(diagram, i, j)[0] == AIR:
                        j -= 1
                elif below_shape == BLOCK:
                    j -= 1
                elif below_shape == LEFT_RAMP or below_shape == RIGHT_RAMP:
                    ball_direction = DOWN

            elif left_shape == LEFT_RAMP:
                ball_direction = DOWN
            elif left_shape == RIGHT_RAMP:
                ball_direction = DOWN
            elif left_shape == BLOCK:
                ball_direction = DOWN
            else:
                print("Unhandled case in LEFT")

        elif ball_direction == RIGHT:
            right_shape = get_right_tile(diagram, i, j)[0]
            if right_shape == AIR:
                below_shape = get_below_tile(diagram, i, j)[0]
                if below_shape == AIR:
                    # We always precedence descent, so we have to handle multiple drops at once.
                    tiles_to_drop = 1
                    [i + 1, j]
                    while get_below_tile(diagram, i + tiles_to_drop, j)[0] == AIR:
                        tiles_to_drop += 1
                    i += tiles_to_drop
                    if get_right_tile(diagram, i, j)[0] == AIR:
                        j += 1
                elif below_shape == BLOCK:
                    j += 1
                elif below_shape == LEFT_RAMP or below_shape == RIGHT_RAMP:
                    ball_direction = DOWN

            elif right_shape == LEFT_RAMP:
                ball_direction = DOWN
            elif right_shape == RIGHT_RAMP:
                ball_direction = DOWN
            elif right_shape == BLOCK:
                ball_direction = DOWN
            else:
                print("Unhandled case in LEFT")

        ball_position = [i, j]

    # Write the ball in its final resting position
    diagram[ball_position[0]][ball_position[1]] = [BALL]
    return diagram
