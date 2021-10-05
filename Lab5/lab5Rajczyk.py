import numpy as np
from collections import defaultdict
from typing import Callable
from sklearn.cluster import DBSCAN
from copy import deepcopy
from collections import Counter
from timeit import default_timer
from math import fsum

N = 4


# ngrams

def make_ngrams(text):
    n_grams = defaultdict(lambda: 0)
    for i in range(len(text) - N + 2):
        n_grams[text[i:i + N]] += 1
    return n_grams


# METRICES

def Levenstein(a, b):
    size_a: int = len(a)
    size_b: int = len(b)
    maximum = max(size_b, size_a)
    L = [[0 for _ in range(size_b + 1)] for _ in range(size_a + 1)]
    for i in range(size_b + 1):
        L[0][i] = i

    for i in range(size_a + 1):
        L[i][0] = i

    for i in range(1, size_a + 1):
        for j in range(1, size_b + 1):
            cost: int = 0
            if a[i - 1] != b[j - 1]:
                cost = 1
            L[i][j] = min(L[i - 1][j] + 1, L[i][j - 1] + 1, L[i - 1][j - 1] + cost)
    return L[-1][-1] / maximum


def DICE_distance(line1, line2):
    ngrams1 = make_ngrams(line1)
    ngrams2 = make_ngrams(line2)
    intersection = set(ngrams1.keys()) & set(ngrams2.keys())
    return 1 - 2 * len(intersection) / (len(ngrams1.keys()) + len(ngrams2.keys()))


def cosine_distance(line1, line2):
    ngrams1 = list(make_ngrams(line1).values())
    ngrams2 = list(make_ngrams(line2).values())
    a = np.zeros(max(len(ngrams1), len(ngrams2)))
    b = np.zeros(max(len(ngrams1), len(ngrams2)))
    for i in range(len(ngrams1)):
        a[i] = ngrams1[i]
    for i in range(len(ngrams2)):
        b[i] = ngrams2[i]
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    return cos_sim


def LCS_distance(line1, line2):
    len_a = len(line1)
    len_b = len(line2)
    maximum: float = 0.0
    C = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            if line1[i - 1] == line2[j - 1]:
                C[i][j] = C[i - 1][j - 1] + 1
                maximum = max(maximum, C[i][j])
    return 1.0 - maximum / max(len_a, len_b)


# DIST MATRIX

def create_dist_matrix(lines, metric_function: Callable):
    n = len(lines)
    matrix = np.zeros((n, n))
    for i in range(len(lines)):
        line1 = lines[i]
        for j in range(i + 1, len(lines)):
            line2 = lines[j]
            p = metric_function(line1, line2)
            matrix[i][j] = p
            matrix[j][i] = p
    return matrix


# STOPLISTA

def create_stoplist(lines, words_to_remove):
    signs_to_remove = '\'"/:.;,()'
    lines_dpcp = deepcopy(lines)
    for i in range(len(lines_dpcp)):
        for sign in signs_to_remove:
            lines_dpcp[i] = lines_dpcp[i].replace(sign, ' ')
    words_count = Counter()
    split_lines = []
    for line in lines_dpcp:
        split_lines.append(line.split(' '))
    for line in split_lines:
        for word in line:
            if word != '':
                words_count[word] += 1
    words_count = words_count.most_common(words_to_remove)
    for i in range(len(lines_dpcp)):
        for word in words_count:
            lines_dpcp[i] = lines_dpcp[i].replace(word[0], ' ')
    return lines_dpcp


# KLASTERYZACJA

def clustering(lines_set, metric_fun: Callable, epsilon):
    X: np.array = create_dist_matrix(lines_set, metric_fun)
    clusters = DBSCAN(eps=epsilon, min_samples=2, metric="precomputed").fit(X)
    return clusters, X


def Dunn_index(clusters_set: DBSCAN, dist_matrix):
    labels = clusters_set.labels_
    n = max(labels) + 1
    T = [[] for _ in range(n)]
    min_d = 10 ** 5
    max_size = 0
    centroids = []
    for i in range(len(labels)):
        if labels[i] != -1:
            T[labels[i]].append(i)
    for elem in T:
        centroids.append(find_centroid(elem, dist_matrix)[0])
    for i in range(n):
        max_size = max(max_size, len(T[i]))
        for j in range(i + 1, n):
            p = dist_matrix[centroids[i]][centroids[j]]
            min_d = min(p, min_d)
    return min_d / max_size


def find_centroid(cluster, dist_matrix):
    min_index = 0
    min_val = 10 ** 5
    mean = 0
    for elem in cluster:
        elem_sum = 0
        for neighbour in cluster:
            if neighbour != elem:
                elem_sum += dist_matrix[elem][neighbour]
        mean += elem_sum
        if elem_sum < min_val:
            min_val = elem_sum
            min_index = elem
    mean = mean / (2 * len(cluster))
    return min_index, mean


def DB_index(clusters_set: DBSCAN, dist_matrix):
    labels = clusters_set.labels_
    n = max(labels) + 1
    T = [[] for _ in range(n)]
    centroids = [0 for _ in range(n)]
    means = [0 for _ in range(n)]
    max_val = 0
    for i in range(len(labels)):
        if labels[i] != -1:
            T[labels[i]].append(i)
    for i in range(n):
        elems = T[i]
        c, m = find_centroid(elems, dist_matrix)
        means[i] = m
        centroids[i] = c
    for i in range(n):
        for j in range(i + 1, n):
            val = (means[i] + means[j]) / dist_matrix[i][j]
            max_val = max(max_val, val)
    return max_val


def run_test(lines_set, algorithm: Callable, lines_num, slist_size, epsilon):
    data = lines_set[:lines_num]
    if slist_size > 0:
        data = create_stoplist(data, slist_size)
    start = default_timer()
    clusters, dist_matrix = clustering(data, algorithm, epsilon)
    end = default_timer()
    d_ind = Dunn_index(clusters, dist_matrix)
    db_ind = DB_index(clusters, dist_matrix)
    print("Czas realizacji:", end - start, "Indeks Daviesa-Bouldina", db_ind, "Indeks Dunna:", d_ind)


with open("lines.txt", "r") as file:
    lines = file.readlines()

# TESTS
# DICE
run_test(lines, DICE_distance, 130, 10, 0.3)
run_test(lines, DICE_distance, 130, 0, 0.3)
# euclidean
run_test(lines, cosine_distance, 130, 10, 0.3)
run_test(lines, cosine_distance, 130, 0, 0.3)
# LCS
run_test(lines, LCS_distance, 130, 10, 0.3)
run_test(lines, LCS_distance, 130, 0, 0.3)
# Levenshtein
run_test(lines, Levenstein, 130, 10, 0.3)
run_test(lines, Levenstein, 130, 0, 0.3)
