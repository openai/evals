import numpy as np

def generate_binary_array_and_factors(N):
    # Generate binary array
    binary_array = np.random.randint(2, size=N)
    # Generate array of factor pairs
    factor_pairs = [(i, N//i) for i in range(2, int(np.sqrt(N)) + 1) if N % i == 0]
    # Return both
    return binary_array.tolist(), factor_pairs

def generate_one_sample_json_string(binary_array_str, dimensions_str, answer_str):
    base_string = '{"input": [{"role": "system", "content": "Given the user-provided binary array, map the binary array onto a grid, wherein the dimensions of the grid are as provided by the user ([num rows]x[num elements per row]), and the mapping to the grid is done from left to right, top to bottom (provide a visualization of the mapped result). Then explain in a second visualization how the final row of the grid was mapped from the corresponding final binary numbers of the array. Lastly, provide the final row of the grid, in minified JSON format, like this: {\\\"Final Row\\\":[...]}"}, {"role": "user", "content": "Array: ' + binary_array_str + '\\nGrid Dimensions: ' + dimensions_str + '"}], "ideal": "{\\"Final Row\\":' + answer_str + '}"}'
    return base_string

def write_lines_to_file(min_array_len, max_array_len, filename, max_lines = 50):
    num_lines = 0;
    # Open the file for writing
    with open(filename, 'w') as file:
        # Loop through all possible array lengths
        for i in range(min_array_len, max_array_len + 1):
            # Generate a binary array and its factors
            (arr, pairs) = generate_binary_array_and_factors(i)
            # Loop through all the factors
            for j in range(len(pairs)):
              # Get the dimensions of the subarray
              dims = str(pairs[j][0]) + 'x' + str(pairs[j][1])
              # Get the subarray as a string and remove spaces
              ans = str(arr[-pairs[j][1]:]).replace(' ', '')
              # Generate a JSON string with the array, dimensions, and answer
              line = generate_one_sample_json_string(str(arr).replace(' ', ''), dims, ans)
              # Write the JSON string to the file
              file.write(line + '\n')
              # Increment the number of lines written
              num_lines += 1
              # If we've written the maximum number of lines, stop generating more lines
              if num_lines == max_lines:
                  return

# generate 1k samples (i.e. lines of json) and write to file: samples.jsonl
write_lines_to_file(40, 500, 'samples.jsonl', 1000)
