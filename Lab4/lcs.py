from spacy.tokenizer import Tokenizer
from spacy.lang.pl import Polish
from random import randint
from math import floor


def tokenize(filename):
    res = []
    with open(filename, 'r', encoding="utf8") as file:
        T = file.readlines()
    tokenizer = Tokenizer(Polish().vocab)
    for t in T:
        chars = tokenizer(t)
        for c in chars:
            res.append(str(c))

    return res


def delete_tokens(text):
    new_text: list = text.copy()
    length: int = len(text)
    tokens_to_delete: int = floor(length * 0.03)
    for i in range(tokens_to_delete):
        p = randint(0, length - 1)
        new_text.pop(p)
        length -= 1
    return new_text


def LCS(a: list, b: list):
    size_a: int = len(a)
    size_b: int = len(b)
    C = [[0 for _ in range(size_b + 1)] for _ in range(size_a + 1)]
    commons = []
    for i in range(1, size_a + 1):
        for j in range(1, size_b + 1):
            if a[i - 1] == b[j - 1]:
                C[i][j] = C[i - 1][j - 1] + 1
            else:
                C[i][j] = max(C[i - 1][j], C[i][j - 1])
    i: int = len(a)
    j: int = len(b)
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            commons.append(a[i - 1])
            i -= 1
            j -= 1
        elif C[i - 1][j] > C[i][j - 1]:
            i -= 1
        else:
            j -= 1
    commons.reverse()
    return C, commons


def diff_tool(text, common_arr):
    text_index: int = 0
    arr_index: int = 0
    line: int = 1
    diffs = []
    while text_index < len(text) and arr_index < len(common_arr):
        if text[text_index] == common_arr[arr_index]:
            if text[text_index] == "\n":
                line += 1
            text_index += 1
            arr_index += 1
            continue
        if text[text_index] == "\n":
            text_index += 1
            line += 1
            continue
        elif text[text_index] != common_arr[arr_index]:
            diffs.append((text[text_index], line))
            text_index += 1
    return diffs


def LCS_from_tokens_and_diffs(chars):
    text_a: list = delete_tokens(chars)
    text_b: list = delete_tokens(chars)
    lcs, commons = LCS(text_a, text_b)
    # print(commons)
    print("The length of LCS of two tokens sequence is:", lcs[len(text_a)][len(text_b)])
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("The diff operation on two tokens:")
    diff_in_a = diff_tool(text_a, commons)
    diff_in_b = diff_tool(text_b, commons)
    for elem in diff_in_a:
        print(elem[0], "<", elem[1])
    for elem in diff_in_b:
        print(elem[0], ">", elem[1])


tokens = tokenize("romeo-i-julia-700.txt")

LCS_from_tokens_and_diffs(tokens)
