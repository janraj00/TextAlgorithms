def Levenstein(a: str, b: str):
    size_a: int = len(a)
    size_b: int = len(b)
    # prep
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
    return L


def find_differences(a: str, b: str):
    Lev = Levenstein(a, b)
    i: int = len(a)
    j: int = len(b)
    changes = []
    while i >= 0 and j >= 0:
        if i == 0 and j == 0:
            break
        if i >= 1 and j >= 1:
            if Lev[i - 1][j - 1] <= Lev[i - 1][j] and Lev[i - 1][j - 1] <= Lev[i][j - 1]:
                if a[i - 1] == b[j - 1]:
                    i, j = i - 1, j - 1
                    continue
                changes.append((i - 1, j - 1, 1))
                i, j = i - 1, j - 1
            elif Lev[i - 1][j - 1] > Lev[i][j - 1]:
                changes.append((i, j - 1, 2))
                j -= 1
            elif Lev[i - 1][j - 1] > Lev[i - 1][j]:
                changes.append((i - 1, j, 3))
                i -= 1
        elif i >= 1:
            changes.append((i - 1, j, 3))
            i -= 1
        elif j >= 1:
            changes.append((i, j - 1, 2))
            j -= 1

    changes.reverse()
    return changes


def visualize_differences(a: str, b: str):
    differences = find_differences(a, b)
    i: int = 0
    for operation in differences:
        shift: int = operation[0]
        char: str = b[operation[1]]
        if operation[2] == 1:
            print(a[:i + shift] + '*' + char + '*' + a[i + shift + 1:], "(zamiana litery", a[i + shift], "na", char,
                  ")")
            a = a[:i + shift] + char + a[i + shift + 1:]
        elif operation[2] == 2:
            print(a[:i + shift] + '+' + char + '+' + a[i + shift:], "(dodanie litery", char, ")")
            a = a[:i + shift] + char + a[i + shift:]
            i += 1
        else:
            print(a[:i + shift] + '_' + a[i + shift+1:], "(usunięcie litery", a[i+shift], ")")
            a = a[:i + shift] + a[i + shift+1:]
            i -= 1
    print(a)

pairs = [["los", "kloc"], ["Łódź", "Lodz"], ["kwintesencja", "quintessence"],
         ["ATGAATCTTACCGCCTCG", "ATGAGGCTCTGGCCCCTG"]]

for pair in pairs:
    print(pair[0], " ", pair[1])
    visualize_differences(pair[0], pair[1])
    print("@@@@@@@@")