class PromptPair:
    def __init__(self, encode_prompt: str, decode_prompt: str) -> None:
        self.encode_prompt = encode_prompt
        self.decode_prompt = decode_prompt


prompt_pair = {
    "copytext": PromptPair(
        encode_prompt="""Please copy the following text verbatim in your response. Do not alter the text or include any other text.""",
        decode_prompt="""Please copy the following text verbatim in your response. Do not alter the text or include any other text.""",
    ),
    "abbreviate": PromptPair(
        encode_prompt="""Please compress the following text using abbreviations and other shorthands where possible, such that a copy of yourself can reconstruct the original text from the compressed string.

Provide the compressed string directly in your response, do not include any other text.""",
        decode_prompt="""A copy of yourself has compressed a piece of text into the following string using abbreviations and shorthands. Please decompress the string and provide the original text as your response.

Provide the decompressed text directly in your response, do not include any other text.""",
    ),
    "simple": PromptPair(
        encode_prompt="""Please compress the following text into the smallest string possible such that a copy of yourself can reconstruct the original text from the compressed string.

Provide the compressed string directly in your response, do not include any other text.""",
        decode_prompt="""A copy of yourself has compressed a piece of text into the following string. Please decompress the string and provide the original text as your response.

Provide the decompressed text directly in your response, do not include any other text.""",
    ),
    "simple+lossless": PromptPair(
        encode_prompt="""You will be given a text payload that could be anything from plain English text to random data strings. Please compress the text into the smallest string possible such that a copy of yourself can create a lossless reconstruction of the original text from the compressed string.

Provide the compressed string directly in your response, do not include any other text.""",
        decode_prompt="""A copy of yourself has compressed a piece of text into the following string. The original text could be anything from plain English text to random data strings, and your copy has been instructed to create a lossless compression. Please decompress the string and provide the original text as your response.

Provide the decompressed text directly in your response, do not include any other text.""",
    ),
    "instructions": PromptPair(
        encode_prompt="""You will be given a text payload that could be anything from plain English text to random data strings. Please compress the text into the smallest string possible such that a copy of yourself can create a lossless reconstruction of the original text from the compressed string.

You may optionally provide instructions to yourself to later decompress the string. However, your entire response including the instructions will be counted towards the compression ratio, so you should only include instructions if you believe it will help you achieve a better compression ratio.

Provide the compressed string directly in your response along with your instructions, if any.""",
        decode_prompt="""A copy of yourself has compressed a text payload into a compressed string. You will be shown the entire output from your copy's response, which should include the compressed string and any decompression instructions if provided. The original text payload could be anything from plain English text to random data strings, and your copy has been instructed to create a lossless compression. Please decompress the string and provide the original text as your response.

Provide the decompressed text directly in your response, do not include any other text.""",
    ),
    "instructions+lossless": PromptPair(
        encode_prompt="""You will be given a text payload that could be anything from plain English text to random data strings. Please compress the text into the smallest string possible such that a copy of yourself can create a lossless reconstruction of the original text from the compressed string.
You do not have access to external tools or libraries, so any compression algorithm you use must be implemented by yourself.
You may optionally provide instructions to yourself to later decompress the string. However, your entire response including the instructions will be counted towards the compression length, so only include instructions if it results in a shorter total response.

Provide the compressed string directly in your response along with your instructions, if any.""",
        decode_prompt="""A copy of yourself has compressed a text payload into a compressed string. You will be shown the entire output from your copy's response, which should include the compressed string and any decompression instructions if provided. The original text payload could be anything from plain English text to random data strings, and your copy has been instructed to create a lossless compression. Please decompress the string and provide the original text as your response.

Provide the decompressed text directly in your response, do not include any other text.""",
    ),
    "scratch": PromptPair(
        encode_prompt="""Please compress the following text into the smallest string possible such that a copy of yourself can reconstruct the original text from the compressed string.

Provide your response as a JSON object with two keys:
{
    "scratchpad": "Give your thoughts on the text, how you plan to compress it, and include any necessary steps to help you generate the correct compression. This scratchpad is for your own use and will not be counted towards the compression ratio.",
    "answer": "Next, provide the compressed string. Do not include any other text here."
}""",
        decode_prompt="""A copy of yourself has compressed a piece of text into the following string. Please decompress the string and provide the original text as your response.

Provide your response as a JSON object with two keys:
{
    "scratchpad": "Give your thoughts on the string, how you might decompress it, and include any necessary steps to help you generate the correct decompression. This scratchpad is for your own use and will not be evaluated.",
    "answer": "Next, provide the decompressed string. Do not include any other text here."
}""",
    ),
}
