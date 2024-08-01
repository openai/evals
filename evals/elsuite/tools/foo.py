from datasets import load_dataset


def print_dataset_messages():
    # Load the dataset
    dataset = load_dataset("fixie-ai/tools-audio")

    # Iterate through each row in the dataset
    for row in dataset["train"]:
        # Extract the messages from the current row
        messages = row["messages"]

        # Print each message on a single line
        for message in messages:
            role = message["role"]
            content = message["content"]
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    func = tool_call["function"]
                    content += f" !{func['name']}({func['arguments']})"
            print(f"{role}: {content}")

        # Print a separator between conversations
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    print_dataset_messages()
