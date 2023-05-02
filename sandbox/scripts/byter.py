# specify the file path and encoding
file_path = 'C:/Users/kjs/Documents/GitHub/evals/evals/registry/data/chengyu/sample.jsonl'
encoding = 'utf-8'
byte_position = 85

# open the file in binary mode
with open(file_path, 'rb') as f:
    # move the file pointer to the desired byte position
    f.seek(byte_position)
    # read one byte from the file
    byte = f.read(20)
    # decode the byte to a character using the appropriate encoding
    char = byte.decode(encoding)
    # print the character
    print(char)
