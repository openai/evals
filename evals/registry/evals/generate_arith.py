import json


def to_binary(num):
    """Converts a decimal number to binary representation"""
    return bin(num)


def get_template(q, a):
    return {
        "input": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": q},
        ],
        "ideal": a,
    }


def generate_question(num1, num2):
    """Generates a question asking for the sum of two numbers in binary representation"""
    binary1 = to_binary(num1)
    binary2 = to_binary(num2)
    question = f"What is the binary representation of the sum of {binary1} and {binary2}?"
    return question


def generate_questions(output_path, max_num=1000):
    """Generates a list of questions asking for the sum of two numbers in binary representation"""

    questions = []
    for num1 in range(max_num):
        for num2 in range(max_num):
            question = generate_question(num1, num2)  # Generate question for sum of 5 and 3
            questions.append(get_template(question, to_binary(num1 + num2)))
    with open(output_path, "w") as f:
        for question in questions:
            json.dump(question, f)
            f.write("\n")


generate_questions("evals/registry/data/binary_arithmatic/binary_addition.jsonl", 300)
