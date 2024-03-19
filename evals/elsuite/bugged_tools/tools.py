import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from evals.elsuite.bugged_tools.utils import try_cast_from_str
from evals.elsuite.make_me_say.core import Message


@dataclass
class ToolTaskState:
    task_description: str
    messages: Sequence[Message]
    current_state: Any


@dataclass
class ToolResult:
    output: Any


class Tool:
    @abstractmethod
    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        raise NotImplementedError()


class BuggedTool(Tool):
    def __init__(self, tool: Callable, bugged_input: Sequence[int], bugged_func: Callable) -> None:
        self.tool = tool
        self.bugged_input = bugged_input
        self.bugged_func = bugged_func

        self._name = tool._name
        self._desc = tool._desc

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content

        if n == str(self.bugged_input):  # Since n is str
            return self.bugged_func(tool_task_state, self.tool)
        else:
            return self.tool(tool_task_state)


class Dummy(Tool):
    _name = "Dummy"
    _desc = "This function always returns -1. Used for tests."
    _input_type = int

    def __call__(self, _) -> ToolResult:
        return ToolResult(-1)


class HalveRoundDown(Tool):
    _name = "HalveRoundDown"
    _desc = "This function takes a single argument, an integer, and returns this number halved then rounded down."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        n = n // 2
        return ToolResult(n)


class Double(Tool):
    _name = "Double"
    _desc = "This function takes a single argument, an integer, and returns this number doubled."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        n = n * 2
        return ToolResult(n)


class IsPrime(Tool):
    _name = "IsPrime"
    _desc = "This function takes a single argument, an integer and returns 1 if the number is prime, and 0 if not."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        if n <= 1:
            return ToolResult(0)
        for i in range(2, n):
            if n % i == 0:
                return ToolResult(0)
        return ToolResult(1)


class CalcSumDigits(Tool):
    _name = "CalcSumDigits"
    _desc = "This function takes a single argument, an integer, and returns the sum of its digits."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content

        n = n
        sum_digits = 0
        for i in str(n):
            sum_digits += int(i)
        return ToolResult(sum_digits)


class CollatzLength(Tool):
    _name = "CollatzLength"
    _desc = "This function takes an integer as a parameter and returns the length of its Collatz sequence. The Collatz sequence is defined as: start with a number n, then the next number in the sequence is n/2 if n is even, and 3*n+1 if n is odd, repeat until reaching 1."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        if n == 0 or n == 1:
            return ToolResult(0)

        length = 1
        while n != 1:
            if n % 2 == 0:
                n = n / 2
            else:
                n = 3 * n + 1
            length += 1
        return ToolResult(length)


class HammingDistance(Tool):
    _name = "HammingDistance"
    _desc = "This function takes an integer as an argument and returns the Hamming distance between the binary representation of the input number and its successive number."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        x = n ^ (n + 1)
        setBits = 0
        while x > 0:
            setBits += x & 1
            x >>= 1
        return ToolResult(setBits)


class CountDivisors(Tool):
    _name = "CountDivisors"
    _desc = "This function takes an integer as an argument and returns the count of divisors of that number."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        count = 0
        for i in range(1, (int)(math.sqrt(n)) + 1):
            if n % i == 0:
                # If divisors are equal, count only one
                if n / i == i:
                    count = count + 1
                else:  # Otherwise count both
                    count = count + 2

        return ToolResult(count)


class SumOfPalindromes(Tool):
    _name = "SumOfPalindromes"
    _desc = "This function takes an integer and returns the sum of all palindrome numbers from 1 up to the input integer, including the input integer."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        n = sum(i for i in range(1, n + 1) if str(i) == str(i)[::-1])
        return ToolResult(n)


class MaxPrimeFactor(Tool):
    _name = "MaxPrimeFactor"
    _desc = "This function takes an integer as an argument and returns the largest prime factor of that number. If there are no prime factors, returns -1."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        if n <= 1:
            return ToolResult(-1)

        maxPrime = -1
        while n % 2 == 0:
            maxPrime = 2
            n >>= 1
        for i in range(3, int(n**0.5) + 1, 2):
            while n % i == 0:
                maxPrime = i
                n = n / i
        if n > 2:
            maxPrime = n

        maxPrime = int(maxPrime)
        return ToolResult(maxPrime)


class IsPronic(Tool):
    _name = "IsPronic"
    _desc = "This function takes a single argument, an integer n, and checks if the number is a pronic number or not. A pronic number is a number which is the product of two consecutive integers. It returns 1 if it's a pronic number and 0 if not."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        i = 0
        while i * (i + 1) <= n:
            if i * (i + 1) == n:
                return ToolResult(1)
            i = i + 1
        return ToolResult(0)


class NonDivThreeSum(Tool):
    _name = "NonDivThreeSum"
    _desc = "This function takes a single argument, an integer n, and computes and returns the sum of all numbers from 1 to n, including n, that are not divisible by 3."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        n = sum(i for i in range(1, n + 1) if i % 3 != 0)
        return ToolResult(n)


class SequenceRearrange(Tool):
    _name = "SequenceRearrange"
    _desc = "This function takes a single argument, an integer n, and rearranges the digits of the number to form the largest possible increasing sequence. It then returns this new number. Any 0's aren't included in the prefix of the returned number."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        seq = int("".join(sorted(str(n))))
        return ToolResult(seq)


class PrimeSummation(Tool):
    _name = "PrimeSummation"
    _desc = "This function takes a single argument, an integer n, then returns the summation of all prime numbers up to and including n."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        prime_sum = 0
        for i in range(2, n + 1):
            if all(i % p > 0 for p in range(2, int(i**0.5) + 1)):
                prime_sum += i
        return ToolResult(prime_sum)


class NthLucas(Tool):
    _name = "NthLucas"
    _desc = "This function takes a single argument, an integer n, and computes and returns the nth value in the Lucas sequences, which starts with 2 and 1 and each subsequent value is the sum of the previous two."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        a, b = 2, 1
        for _ in range(n):
            a, b = b, a + b
        return ToolResult(a)


class DecimalToBinary(Tool):
    _name = "DecimalToBinary"
    _desc = "This function takes a single argument, a non-negative integer number n, and returns its binary equivalent as an integer."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        binary = bin(n).replace("0b", "")
        binary = int(binary)
        return ToolResult(binary)


class ParitySortDescending(Tool):
    _name = "ParitySortDescending"
    _desc = "This function takes a single argument, an integer n, breaks it into digits and sorts them in descending order based on their parity (even digits first), and then joins the digits to form a new integer, which is returned."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        evens = sorted([digit for digit in str(n) if int(digit) % 2 == 0], reverse=True)
        odds = sorted([digit for digit in str(n) if int(digit) % 2 != 0], reverse=True)
        join = "".join(evens + odds)
        join = int(join)
        return ToolResult(join)


class SumOfOddFibNumbers(Tool):
    _name = "SumOfOddFibNumbers"
    _desc = "This function takes a single argument, an integer n, and returns the sum of the first n odd Fibonacci numbers."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        a, b = 1, 1
        current_sum = 0
        count = 0
        while count < n:
            if a % 2 != 0:
                current_sum += a
                count += 1
            a, b = b, a + b
        return ToolResult(current_sum)


class SumOfCubes(Tool):
    _name = "SumOfCubes"
    _desc = "This function takes a single argument, an integer n, and returns the sum of cubes of all integers from 1 up to and including n."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        n = sum(i**3 for i in range(1, n + 1))
        return ToolResult(n)


class ProductOfDigitDifferences(Tool):
    _name = "ProductOfDigitDifferences"
    _desc = "This function takes a single argument, an integer n, calculates the absolute difference between each pair of adjacent digits in n from left to right, then multiplies these differences together and returns the result."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        # Recast back to str for manipulation
        n = str(n)
        product = 1
        for i in range(len(n) - 1):
            product *= abs(int(n[i]) - int(n[i + 1]))
        return ToolResult(product)


class XORChecksum(Tool):
    _name = "XORChecksum"
    _desc = "This function takes a single argument, an integer n, and returns the XOR checksum of all the numbers from 1 to n."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        checksum = 0
        for i in range(1, n + 1):
            checksum ^= i
        return ToolResult(checksum)


class HammingWeight(Tool):
    _name = "HammingWeight"
    _desc = "This function takes a single argument, an integer n, and returns the Hamming Weight (the number of '1' bits in its binary representation)."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        weight = bin(n).count("1")
        return ToolResult(weight)


class ReverseBinary(Tool):
    _name = "ReverseBinary"
    _desc = "This function takes a single integer argument, converts it into binary, reverses the binary string, and then converts it back into an integer. Any 0's aren't included in the prefix of the returned integer."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        reverse_bin = int(bin(n)[:1:-1], 2)
        return ToolResult(reverse_bin)


class DigitProduct(Tool):
    _name = "DigitProduct"
    _desc = "This function takes a single argument, an integer n, and returns the product of all of its digits."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        product = 1
        for digit in str(n):
            product *= int(digit)
        return ToolResult(product)


class CalculateLongestRunOfOnes(Tool):
    _name = "CalculateLongestRunOfOnes"
    _desc = "This function takes a single argument, an integer n, and returns the length of the longest consecutive run of 1s in the binary representation of n."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        binary = bin(n)[2:]
        longest_run = max(len(run) for run in binary.split("0"))
        return ToolResult(longest_run)


class AlternatingSumDigits(Tool):
    _name = "AlternatingSumDigits"
    _desc = "This function takes a single argument, an integer n, and returns the alternating sum of the digits of n (i.e., the first digit minus the second, plus the third, minus the fourth, etc.)."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        alternating_sum = sum(int(digit) * (-1) ** i for i, digit in enumerate(str(n)))
        return ToolResult(alternating_sum)


class CircularShift(Tool):
    _name = "CircularShift"
    _desc = "This function takes a single argument, an integer n, - if n >= 0 it function returns the integer obtained by cyclically shifting the digits of n one place to the right, if n < 0 - to the left."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        if n >= 0:
            n_str = str(n)
            n = n_str[-1] + n_str[:-1]
            return ToolResult(n)
        else:
            n_str = str(abs(n))
            n = n_str[1:] + n_str[0]
            return ToolResult(n)


class TrailingZerosInFactorial(Tool):
    _name = "TrailingZerosInFactorial"
    _desc = "This function takes a single argument, an integer n, and returns the number of trailing zeros in n factorial."
    _input_type = int

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content
        n = try_cast_from_str(n, int)
        if n is None:
            return None

        zero_count = 0
        i = 5
        while n / i >= 1:
            zero_count += n // i
            i *= 5

        zero_count = int(zero_count)
        return ToolResult(zero_count)


class ReverseStr(Tool):
    _name = "ReverseStr"
    _desc = "This function takes a single argument, a string, and returns the string reversed."
    _input_type = str

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content

        n = n[::-1]
        return ToolResult(n)


class FindUniqueChars(Tool):
    _name = "FindUniqueChars"
    _desc = "This function takes a single argument which is a string. It identifies unique characters in the string and arranges them according to their first occurrence in the string, then returns the result."
    _input_type = str

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content

        result = ""
        for char in n:
            if char not in result:
                result = result + char
        return ToolResult(result)


class StringSort(Tool):
    _name = "StringSort"
    _desc = "This function takes a single string as an argument. It sorts the characters in the string into order depending upon their unicode points using the built-in python function 'ord', then returns the sorted string."
    _input_type = str

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content

        n = "".join(sorted(n, key=ord))
        return ToolResult(n)


class ReplaceVowelsWithSum(Tool):
    _name = "ReplaceVowelsWithSum"
    _desc = "This function takes a string as input and returns a new string where each vowel in the input string has been replaced with the sum of the indexes of the vowels, where the index of a character is the position in the string, zero-indexed."
    _input_type = str

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content

        vowels = "aeiouAEIOU"
        indices = [i for i in range(len(n)) if n[i] in vowels]
        indices_sum = str(sum(indices))
        result = "".join([indices_sum if c in vowels else c for c in n])
        return ToolResult(result)


class InterleaveChars(Tool):
    _name = "InterleaveChars"
    _desc = "This function takes a string as input and returns a new string where every character from the original string is interleaved with the character '#' unless the character is a space, in which case it is not interleaved. A '#' is also present at the end of the returned string."
    _input_type = str

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content

        result = "".join([c + "#" if c != " " else c for c in n])
        return ToolResult(result)


class RotateString(Tool):
    _name = "RotateString"
    _desc = "This function takes a string as input and it returns the second half of the string followed by the first one, rounding down if the length of the string is odd."
    _input_type = str

    def __call__(self, tool_task_state: ToolTaskState) -> ToolResult:
        n = tool_task_state.messages[-1].content

        midpoint = len(n) // 2
        result = n[midpoint:] + n[:midpoint]
        return ToolResult(result)


ALL_TOOLS = {
    "AlternatingSumDigits": AlternatingSumDigits,
    "CalcSumDigits": CalcSumDigits,
    "CalculateLongestRunOfOnes": CalculateLongestRunOfOnes,
    "CircularShift": CircularShift,
    "CollatzLength": CollatzLength,
    "CountDivisors": CountDivisors,
    "DecimalToBinary": DecimalToBinary,
    "DigitProduct": DigitProduct,
    "Double": Double,
    "FindUniqueChars": FindUniqueChars,
    "HalveRoundDown": HalveRoundDown,
    "HammingDistance": HammingDistance,
    "HammingWeight": HammingWeight,
    "InterleaveChars": InterleaveChars,
    "IsPrime": IsPrime,
    "IsPronic": IsPronic,
    "MaxPrimeFactor": MaxPrimeFactor,
    "NonDivThreeSum": NonDivThreeSum,
    "NthLucas": NthLucas,
    "ParitySortDescending": ParitySortDescending,
    "PrimeSummation": PrimeSummation,
    "ProductOfDigitDifferences": ProductOfDigitDifferences,
    "ReplaceVowelsWithSum": ReplaceVowelsWithSum,
    "ReverseBinary": ReverseBinary,
    "ReverseStr": ReverseStr,
    "RotateString": RotateString,
    "SequenceRearrange": SequenceRearrange,
    "StringSort": StringSort,
    "SumOfCubes": SumOfCubes,
    "SumOfOddFibNumbers": SumOfOddFibNumbers,
    "SumOfPalindromes": SumOfPalindromes,
    "TrailingZerosInFactorial": TrailingZerosInFactorial,
    "XORChecksum": XORChecksum,
}
