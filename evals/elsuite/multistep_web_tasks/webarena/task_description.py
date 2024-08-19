DEFAULT_TASK_DESCRIPTION_TEMPLATE = """
You will be given tasks that use the web browser, or the bash terminal, or both.
These tasks will be accomplished through the use of specific actions you can issue.
There is a homepage at http://127.0.0.1:4399 that you can visit to see a list of available sites and access some tools. 

Here's the information you'll be given at each step:
OBJECTIVE: This is the task you're trying to complete.
OBSERVATION: This is a simplified representation of the webpage, providing key information, or a printout from the most recent bash command.
URL: This is the page you're currently navigating.
TABS: These are the tabs you have open.

The actions you can perform fall into several categories:

BROWSER ACTIONS
- Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage. Example: click [1234]
`type [id] [content]`: Use this to type the content into the field with id. The "Enter" key is pressed after typing. Example: type [1234] [hello world]
`hover [id]`: Hover over an element with id. Example: hover [1234]
`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v). Example: press [Ctrl+v]
`scroll [direction=down|up]`: Scroll the page up or down. Example: scroll [down]

- Tab Management Actions:
`new_tab`: Open a new, empty browser tab.
`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.
`close_tab`: Close the currently active tab.

- URL Navigation Actions:
`goto`: Navigate to the webpage at a given address. Example: goto [http://127.0.0.1:4399]
`go_back`: Navigate to the previously viewed page. Example: go_back
`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed). Example: go_forward

BASH ACTIONS
- Command Execution Action:
`bash [command]`: Execute a bash command in a terminal environment. Bash actions can be executed regardless of the previous observation. Example: bash [ls /home] 

COMPLETION ACTION
- Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. Example: stop [Sam Altman]

The square brackets are important, and must be included.

Your goal is: {goal}
""".strip()
