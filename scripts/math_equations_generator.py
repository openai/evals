import json
import random


def generate_equations(num_equations):
    """Generate random addition, subtraction, multiplication and division equations with 10 numbers."""
    for i in range(num_equations):
        # Generate 10 random integers between -999 and 999
        nums = [random.randint(-999, 999) for _ in range(10)]

        # Choose randomly whether to add, subtract, multiply, or divide each pair of numbers
        operators = [random.choice(["+", "-", "*", "/"]) for _ in range(9)]

        # Combine the numbers and operators into an equation string
        equation = str(nums[0])
        for j in range(9):
            equation += " " + operators[j] + " " + str(nums[j + 1])

        # Evaluate the equation and calculate the answer
        answer = eval(equation)

        # Choose randomly whether the response is correct or incorrect
        is_correct = random.choice([True, False])

        # Generate the response based on whether it's correct or incorrect
        response = "Correct" if is_correct else "Incorrect"

        # Create a JSON object with the equation and response
        data = {"equation": equation, "response": response, "answer": answer}

        # Yield the JSON object as a line in JSONL format
        yield json.dumps(data)


# Example usage: Generate 100 equations and print them as JSONL
equations = generate_equations(250)
for equation in equations:
    print(equation)
