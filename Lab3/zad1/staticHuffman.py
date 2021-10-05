from collections import deque
from bitarray import bitarray
from bitarray.util import int2ba
from timeit import default_timer
import os
from random import randint


class Node:
    def __init__(self, char, weight: int, left: 'Node' = None, right: 'Node' = None):
        self.char = char
        self.weight = weight
        self.left = left
        self.right = right

    def create_dict(self, dictionary: dict, bitarr: bitarray = bitarray()):
        if self.char:
            dictionary[self.char] = bitarr
        if self.left is not None:
            bitarr_copy = bitarr.copy()
            bitarr_copy.append(0)
            self.left.create_dict(dictionary, bitarr_copy)
        if self.right is not None:
            bitarr_copy = bitarr.copy()
            bitarr_copy.append(1)
            self.right.create_dict(dictionary, bitarr_copy)
        return dictionary


class StaticHuffman:
    def __init__(self, text: str):
        self.text = text
        self.chars = self.count_letters()
        self.root = self.huffman()

    def count_letters(self) -> dict:
        chars_array = {}
        for letter in self.text:
            if letter not in chars_array.keys():
                chars_array[letter] = 1
            else:
                chars_array[letter] += 1
        return chars_array

    def huffman(self):
        nodes_list = []
        leaves = deque()
        internal_nodes = deque()
        letter_counts = self.chars
        for a, w in letter_counts.items():
            nodes_list.append(Node(a, w))
        nodes_list.sort(key=lambda x: x.weight)
        for elem in nodes_list:
            leaves.append(elem)

        while len(leaves) + len(internal_nodes) > 1:
            elems = []
            for i in range(2):
                if len(leaves) == 0:
                    elems.append(internal_nodes.popleft())
                    continue
                if len(internal_nodes) == 0:
                    elems.append(leaves.popleft())
                    continue
                if leaves[0].weight < internal_nodes[0].weight:
                    elems.append(leaves.popleft())
                else:
                    elems.append(internal_nodes.popleft())

            element1, element2 = elems[0], elems[1]
            internal_nodes.append(Node(None, element1.weight + element2.weight, element1, element2))
        return internal_nodes[0]

    def Huffman_dict(self):
        b_array = bitarray()
        d = dict()
        return self.root.create_dict(d, b_array)

    def code(self, dictionary: dict):  # number_of_chars--[letter, lengthofletter(bin), code_of_letter) ]
        # for every char in dict]--[letter_code for code in text]
        bit_code = bitarray()
        bit_code.extend(int2ba(len(dictionary), 8, endian='big'))
        for letter, letter_code in dictionary.items():
            command = bitarray()
            command.frombytes(bytes(letter, 'ascii'))
            command.extend(int2ba(len(letter_code), 8, endian='big'))
            command.extend(letter_code)
            bit_code.extend(command)

        text_code = bitarray()
        text_code.encode(dictionary, self.text)
        bit_code.extend(text_code)

        return bit_code

    def decoder(self, bits: bitarray):
        dictionary = dict()
        dict_size = int.from_bytes(bits[:8], byteorder='big', signed=True)
        bits = bits[8:]
        i: int = 0
        while i < dict_size:
            letter = bits[:8].tobytes().decode()
            bits = bits[8:]
            length = int.from_bytes(bits[:8], byteorder='big', signed=True)
            bits = bits[8:]
            letter_code = bits[:length]
            bits = bits[length:]
            dictionary[letter] = letter_code
            i += 1

        return bits.decode(dictionary)


def static_Huffman_test(filename, encode_file, results):
    with open(filename, "r") as file:
        T = file.read()
    SH = StaticHuffman(T)
    huffman_dict = SH.Huffman_dict()
    code_start = default_timer()
    encode_text = SH.code(huffman_dict)
    nieszczesne_modulo = 8 - len(encode_text) % 8
    with open(encode_file, "wb") as file:
        encode_text.tofile(file)
    code_end = default_timer()

    bit_arr = bitarray()
    with open(encode_file, "rb") as file:
        bit_arr.fromfile(file)
    bit_arr = bit_arr[:-nieszczesne_modulo]

    decode_start = default_timer()
    decode_text = SH.decoder(bit_arr)
    decode_end = default_timer()
    decode_text = ''.join(decode_text)
    # print(decode_text)

    with open(results, "w") as file:
        file.write(decode_text)

    print("&", 1 - os.stat(encode_file).st_size / os.stat(filename).st_size, "&", code_end - code_start, "&",
          decode_end - decode_start, "\\ \hline")

def prep_random(letters):
    return ''.join([chr(randint(32, 126)) for _ in range(letters)])

'''
static_Huffman_test("big_text.txt", "compress.txt", "res.txt")
static_Huffman_test("mediumbig_text.txt", "compress.txt", "res.txt")
static_Huffman_test("medium_text.txt", "compress.txt", "res.txt")
static_Huffman_test("small_text.txt", "compress.txt", "res.txt")
print("    ")
static_Huffman_test("big_code.txt", "compress.txt", "res.txt")
static_Huffman_test("mediumbig_code.txt", "compress.txt", "res.txt")
static_Huffman_test("medium_code.txt", "compress.txt", "res.txt")
static_Huffman_test("small_code.txt", "compress.txt", "res.txt")
print("    ")
'''

with open("small_rand.txt", "w") as file:
    file.write(prep_random(1000))

with open("medium_rand.txt", "w") as file:
    file.write(prep_random(20000))

with open("mediumbig_rand.txt", "w") as file:
    file.write(prep_random(100000))

with open("big_rand.txt", "w") as file:
    file.write(prep_random(1000000))

static_Huffman_test("small_rand.txt", "compress.txt", "res.txt")
static_Huffman_test("medium_rand.txt", "compress.txt", "res.txt")
static_Huffman_test("mediumbig_rand.txt", "compress.txt", "res.txt")
static_Huffman_test("big_rand.txt", "compress.txt", "res.txt")
