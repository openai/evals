"""
This file defines high-level actions for the environment. High-level actions are more complex
actions that require multiple low-level actions to be executed.

Note: This file is adapted from MLAgentBench with minimal edits made. The original file can be
found at: https://github.com/snap-stanford/MLAgentBench/blob/main/MLAgentBench/high_level_actions.py.
"""

import datetime
import difflib
import os
import shutil

from evals.elsuite.hr_ml_agent_bench.low_level_actions import read_file, write_file
from evals.elsuite.hr_ml_agent_bench.schema import ActionInfo, EnvException
from evals.elsuite.hr_ml_agent_bench.utils import complete_text


def understand_file(file_name, things_to_look_for, solver, work_dir=".", **kwargs):
    lines = read_file(file_name, work_dir=work_dir, **kwargs).split("\n")
    # group lines to blocks so that each block has at most 10000 characters
    counter = 0
    blocks = []
    while counter < len(lines):
        block = []
        start_line_number = counter + 1
        while counter < len(lines) and len("\n".join(block)) + len(lines[counter]) < 10000:
            block.append(lines[counter])
            counter += 1
        if len(block) > 0:
            end_line_number = counter
            blocks.append(("\n".join(block), start_line_number, end_line_number))
        else:
            end_line_number = start_line_number
            # probably a file of one/few very long line; split by 10000 characters
            for i in range(0, len(lines[counter]), 10000):
                blocks.append((lines[counter][i : i + 10000], start_line_number, end_line_number))
            counter += 1

    descriptions = []
    for idx, (b, start_line_number, end_line_number) in enumerate(blocks):
        start_char_number = sum([len(b) for b in blocks[:idx]])
        end_char_number = start_line_number + len(b)
        prompt = f"""Given this (partial) file from line {start_line_number} character {start_char_number} to line {end_line_number} character {end_char_number}: 
    ``` 
    {b}
    ```
    Here is a detailed description on what to look for and what should returned: {things_to_look_for}
    The description should short and also reference crtical lines in the script relevant to what is being looked for. Only describe what is objectively confirmed by the file content. Do not include guessed numbers. If you cannot find the answer to certain parts of the request, you should say "In this segment, I cannot find ...".
    """

        completion = complete_text(prompt, solver=solver)
        descriptions.append(completion)
    if len(descriptions) == 1:
        return descriptions[0]
    else:
        descriptions = "\n\n".join(["Segment {idx}: \n\n" + s for s in descriptions])
        prompt = f"""Given the relevant observations for each segments of a file, summarize to get a cohesive description of the entire file on what to look for and what should returned: {things_to_look_for}
    {descriptions}
    """

        completion = complete_text(prompt, solver=solver)

        return completion


def edit_script(
    script_name,
    edit_instruction,
    save_name,
    solver,
    max_tokens=4_000,
    work_dir=".",
    **kwargs,
):
    # TODO: handle long file editing
    try:
        content = read_file(script_name, work_dir=work_dir, **kwargs)
    except:
        write_file(script_name, "", work_dir=work_dir, **kwargs)
        content = ""

    prompt = f"""Given this python script:
    ```python 
    {content}
    ```
    Edit the script by following the instruction:
    {edit_instruction}
    Provide the full code after the edit, making no other changes. Start the python code with "```python". 

    """

    completion = complete_text(
        prompt,
        solver=solver,
        max_tokens=max_tokens,
    )

    new_content = completion.split("```python")[1].split("```")[0].strip()

    # backup all old file with prefix script_name
    backup_name = os.path.join(
        work_dir,
        "backup",
        f"{script_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    shutil.copyfile(os.path.join(work_dir, script_name), backup_name)

    write_file(save_name, new_content, work_dir=work_dir, **kwargs)

    diff = list(
        difflib.unified_diff(
            content.splitlines(keepends=True), new_content.splitlines(keepends=True)
        )
    )
    diff = "".join(diff)

    return (
        f"The edited file is saved to {save_name}. Here is the diff, please check if the edit is correct and desirable:\n\n"
        + diff
    )


def edit_script_lines(
    script_name,
    start_line_number,
    end_line_number,
    edit_instruction,
    save_name,
    solver,
    max_tokens=4_000,
    work_dir=".",
    **kwargs,
):
    try:
        start_line_number = int(start_line_number)
        end_line_number = int(end_line_number)
    except:
        raise EnvException("start_line_number and end_line_number must be integers")

    try:
        orig_content = read_file(script_name, work_dir=work_dir, **kwargs)
    except:
        write_file(script_name, "", work_dir=work_dir, **kwargs)
        orig_content = ""
    lines = orig_content.split("\n")
    content = "\n".join(lines[max(int(start_line_number) - 1, 0) : int(end_line_number)])

    prompt = f"""Given this segment of a python script:
    ```python 
    {content}
    ```
    Edit this segemnt by following the instruction:
    {edit_instruction}
    Provide the full code after the edit, making no other changes. Start the python code with "```python". 

    """

    completion = complete_text(
        prompt,
        solver=solver,
        max_tokens=max_tokens,
    )

    new_content = (
        "\n".join(lines[: int(start_line_number) - 1])
        + "\n"
        + completion.split("```python")[1].split("```")[0].strip()
        + "\n"
        + "\n".join(lines[int(end_line_number) :])
    )

    # backup all old file with prefix script_name
    backup_name = os.path.join(
        work_dir,
        "backup",
        f"{script_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    shutil.copyfile(os.path.join(work_dir, script_name), backup_name)

    write_file(save_name, new_content, work_dir=work_dir, **kwargs)

    diff = list(
        difflib.unified_diff(
            content.splitlines(keepends=True), new_content.splitlines(keepends=True)
        )
    )
    diff = "".join(diff)

    return (
        f"The edited file is saved to {save_name}. Here is the diff, please check if the edit is correct and desirable:\n\n"
        + diff
    )


def inspect_script_lines(script_name, start_line_number, end_line_number, work_dir=".", **kwargs):
    try:
        start_line_number = int(start_line_number)
        end_line_number = int(end_line_number)
    except:
        raise EnvException("start_line_number and end_line_number must be integers")
    if end_line_number - start_line_number > 100:
        raise EnvException("the number of lines to display is limited to 100 lines")
    try:
        # lines = open(os.path.join(work_dir,script_name)).readlines()
        lines = read_file(script_name, work_dir=work_dir, **kwargs).split("\n")
    except:
        raise EnvException(f"cannot find script {script_name}")

    content = "\n".join(lines[max(int(start_line_number) - 1, 0) : int(end_line_number)])
    return f"Here are the lines (the file ends at line {len(lines)}):\n\n" + content


HIGH_LEVEL_ACTIONS = [
    ActionInfo(
        name="Understand File",
        description="Use this to read the whole file and understand certain aspects. You should provide detailed description on what to look for and what should be returned. To get a better understanding of the file, you can use Inspect Script Lines action to inspect specific part of the file.",
        usage={
            "file_name": "a valid file name with relative path to current directory if needed",
            "things_to_look_for": "a detailed description on what to look for and what should returned",
        },
        return_value="The observation will be a description of relevant content and lines in the file. If the file does not exist, the observation will be an error message.",
        function=understand_file,
    ),
    ActionInfo(
        name="Inspect Script Lines",
        description="Use this to inspect specific part of a python script precisely, or the full content of a short script. The number of lines to display is limited to 100 lines. This is especially helpful when debugging.",
        usage={
            "script_name": "a valid python script name with relative path to current directory if needed",
            "start_line_number": "a valid line number",
            "end_line_number": "a valid line number",
        },
        return_value="The observation will be the content of the script between start_line_number and end_line_number . If the script does not exist, the observation will be an error message.",
        function=inspect_script_lines,
    ),
    ActionInfo(
        name="Edit Script (AI)",
        description="Use this to do a relatively large but cohesive edit over a python script. Instead of editing the script directly, you should describe the edit instruction so that another AI can help you do this.",
        usage={
            "script_name": "a valid python script name with relative path to current directory if needed. An empty script will be created if it does not exist.",
            "edit_instruction": "a detailed step by step description on how to edit it.",
            "save_name": "a valid file name with relative path to current directory if needed",
        },
        return_value="The observation will be the edited content of the script. If the script does not exist, the observation will be an error message. You should always double check whether the edit is correct. If it is far from correct, you can use the Undo Edit Script action to undo the edit.",
        function=edit_script,
    ),
    ActionInfo(
        name="Edit Script Segment (AI)",
        description="Use this to do a relatively large but cohesive edit over a python script over a segment. Instead of editing the script directly, you should describe the edit instruction so that another AI can help you do this.",
        usage={
            "script_name": "a valid python script name with relative path to current directory if needed. An empty script will be created if it does not exist.",
            "start_line_number": "a valid line number",
            "end_line_number": "a valid line number",
            "edit_instruction": "a detailed step by step description on how to edit it.",
            "save_name": "a valid file name with relative path to current directory if needed",
        },
        return_value="The observation will be the edited content of the script. If the script does not exist, the observation will be an error message. You should always double check whether the edit is correct. If it is far from correct, you can use the Undo Edit Script action to undo the edit.",
        function=edit_script_lines,
    ),
]
