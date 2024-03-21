from string import Template

task_description_template = Template("""$explanations

You have the following $action_space actions available to you: $actions.

On each turn, you will receive an observation from the environment, which maybe be one of the following $observation_space possible observations:
$observations
Each action you take will give you a reward (which might be negative) and may move you to a new state.

You should take actions such that you accumulate the highest possible reward across all your actions.

This is how we will play the game:
1. You will consider all the information you have received to decide upon the best action you can take now.
2. You will select your action from the list above by specifying the number key of the action in the command [SELECT: x], where x is the number key of the action.
3. Your selected action will be taken.
4. As a result of your action, you will be given an observation from the environment and you may receive some reward.
5. Repeat from step 1.
""")

step_counter = Template("Total actions taken so far: $step_count")
reward_counter = Template("Total reward so far: $reward_count")
reset_msg = Template("""After the game reset you are now in $observation. 
Please pick an action, providing your reasoning. You must format your final action choice as [SELECT: x]""")
step_result = Template("""You took Action $action. You are now in $next_observation. 
The last step you did provided reward: $reward. 
Please pick an action, providing your reasoning. You must format your final action choice as [SELECT: x]""")
step_result_reset = Template("""You took Action $action. You arrived at $next_observation.
The last step made the game reset.
The last step you did provided reward: $reward.""")