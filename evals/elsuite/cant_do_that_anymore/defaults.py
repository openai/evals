TASK_DESCRIPTION = """You are playing a game of chess. Play the best move available to you.

You must indicate which move you want to make by outputting the move using algebraic notation. Do not output anything else.
- Always write the full start and end position, for example, "a7b6"
- To castle, write the start and end position of the king, for example, "e8g8"
- Do not write "x" for capturing moves"""

TASK_DESCRIPTION_VARIANT = """You are playing a game of a variant of chess. Play the best move available to you.

You must indicate which move you want to make by outputting the move using algebraic notation. Do not output anything else.
- Always write the full start and end position, for example, "a7b6"
- To castle, write the start and end position of the king, for example, "e8g8"
- Do not write "x" for capturing moves

This variant of chess does not follow the same rules as chess. All bishops now move like knights do. DO NOT move bishops diagonally, this is now considered an illegal move. All other pieces don't have their rules changed."""
