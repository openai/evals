from dataclasses import dataclass
from typing import List

from evals.task_state import Message, TaskState


@dataclass
class Interaction:
    #   All messages we've seen (except for the task_description)
    messages: List[Message]

    #   IDs of the CoT private internal messages
    private_messages_ids: List[int]


class PersistentMemoryCache:
    def __init__(
        self,
        interaction_length: int,
    ):
        self.private_interaction_length = interaction_length
        self.last_interaction = None

    def save_private_interaction(self, task_state: TaskState):
        #   Save the interaction
        interaction_messages = task_state.messages
        num_interaction_messages = len(interaction_messages)
        private_messages_ids = (
            [] if self.last_interaction is None else self.last_interaction.private_messages_ids
        )
        private_messages_ids += list(
            range(
                num_interaction_messages - self.private_interaction_length - 1,
                num_interaction_messages - 1,
            )
        )
        self.last_interaction = Interaction(interaction_messages, private_messages_ids)

    def load_private_interaction(self, task_state: TaskState) -> List[Message]:
        if self.last_interaction is None:
            return task_state.messages

        #   Check if task_state matches our last interaction
        interaction = self.last_interaction
        task_state_message_ix = 0
        for our_message_ix in range(0, len(interaction.messages)):
            if our_message_ix in interaction.private_messages_ids:
                continue
            else:
                if (
                    task_state.messages[task_state_message_ix]
                    != interaction.messages[our_message_ix]
                ):
                    raise ValueError(
                        (
                            f"task_state message {task_state_message_ix} different than the corresponding message "
                            "in the interaction history.\n"
                            f"task_state.messages:\n{task_state.messages}\n"
                            f"interaction.messages:\n{interaction.messages}\n"
                        )
                    )
                task_state_message_ix += 1

        return interaction.messages + task_state.messages[task_state_message_ix:]
