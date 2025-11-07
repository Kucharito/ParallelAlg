# srflp.py
from typing import List, Tuple

def load_data(path: str = "Y-10_t.txt") -> Tuple[List[int], List[List[float]]]:
    # načítaj a odfiltruj prázdne riadky
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]

    # n a widths
    if len(lines) < 2:
        raise ValueError("Input too short: need at least 2 lines (n, widths).")
    n = int(lines[0])
    widths = [int(x) for x in lines[1].split()]
    if len(widths) != n:
        raise ValueError(f"Widths length {len(widths)} != n={n}. Line: {lines[1]!r}")

    # pozbieraj všetky čísla matice
    tokens: List[float] = []
    for ln in lines[2:]:
        for p in ln.split():
            try:
                tokens.append(float(p))
            except Exception as e:
                raise ValueError(f"Non-numeric token {p!r} on line {ln!r}") from e

    need_full = n * n                    # plná matica
    need_upper = n * (n + 1) // 2        # horný trojuholník vrátane diagonály
    need_n_minus_1_rows = (n - 1) * n    # tvoj prípad: 9×10 = 90 pri n=10

    if len(tokens) == need_full:
        # plná n×n
        flows = [tokens[i*n:(i+1)*n] for i in range(n)]

    elif len(tokens) == need_upper:
        # horný trojuholník → rozvinúť do symetrickej
        flows = [[0.0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i, n):
                val = tokens[idx]; idx += 1
                flows[i][j] = val
                flows[j][i] = val

    elif len(tokens) == need_n_minus_1_rows:
        # chýba posledný riadok; máme (n-1) riadkov po n čísel
        rows = [tokens[i*n:(i+1)*n] for i in range(n-1)]
        # zostroj posledný riadok zo symetrie: row_n-1[j] = rows[j][n-1], diag=0
        last_row = [rows[i][n-1] for i in range(n-1)] + [0.0]
        flows = rows + [last_row]
        # zosymetrizuj pre istotu
        for i in range(n):
            for j in range(i+1, n):
                if flows[j][i] == 0.0 and flows[i][j] != 0.0:
                    flows[j][i] = flows[i][j]
                elif flows[i][j] == 0.0 and flows[j][i] != 0.0:
                    flows[i][j] = flows[j][i]

    else:
        raise ValueError(
            "Input matrix size mismatch.\n"
            f"n={n}\n"
            f"Collected numeric tokens after widths: {len(tokens)}\n"
            f"Expected full n*n={need_full}, or upper-tri n*(n+1)/2={need_upper}, "
            f"or (n-1)*n={need_n_minus_1_rows} (missing last row).\n"
            "Tip: your file likely misses the last row; this loader now handles that."
        )

    # finálna symetria (nič nepokazí)
    for i in range(n):
        for j in range(i+1, n):
            if flows[j][i] == 0.0 and flows[i][j] != 0.0:
                flows[j][i] = flows[i][j]
            elif flows[i][j] == 0.0 and flows[j][i] != 0.0:
                flows[i][j] = flows[j][i]

    return widths, flows


def cost(permutation, widths, data):
    total_cost = 0.0
    n = len(permutation)
    centers = []
    position = 0
    for idx in permutation:
        centers.append(position + widths[idx] / 2)
        position += widths[idx]
    for i in range(n):
        for j in range(i + 1, n):
            dist = abs(centers[j] - centers[i])
            total_cost += data[permutation[i]][permutation[j]] * dist
    return total_cost


def partial_cost(partial_perm, widths, data):
    pcost = 0.0
    n = len(partial_perm)
    centers = []
    position = 0
    for idx in partial_perm:
        centers.append(position + widths[idx] / 2)
        position += widths[idx]
    for i in range(n):
        ci = centers[i]
        pi = partial_perm[i]
        for j in range(i + 1, n):
            dist = abs(centers[j] - ci)
            pcost += data[pi][partial_perm[j]] * dist
    return pcost


def lower_bound(partial_perm, widths, data, best_cost):
    n = len(data)
    placed = set(partial_perm)
    remaining = [i for i in range(n) if i not in placed]

    lb = partial_cost(partial_perm, widths, data)

    # guard: ak ostáva <2 prvkov, nič nepridávaj (max() by padol / LB by bol príliš agresívny)
    if len(remaining) >= 2:
        # pri tvojej inštancii widths==1 → min_width=1; necháme univerzálne
        min_width = min(widths[i] for i in remaining)
        max_data = max(data[i][j] for i in remaining for j in remaining if i != j)
        optimistic_remaining = max_data * min_width * len(remaining)
        lb += optimistic_remaining

    return lb


def branch_and_bound(partial_perm, widths, data, best_cost, best_perm):
    n = len(data)
    if len(partial_perm) == n:
        c = cost(partial_perm, widths, data)
        if c < best_cost[0]:
            best_cost[0] = c
            best_perm[0] = partial_perm.copy()
            print(f"New best cost: {best_cost[0]:.2f} -> perm {best_perm[0]}")
        return

    lb = lower_bound(partial_perm, widths, data, best_cost[0])
    if lb >= best_cost[0]:
        return

    # jednoduchá expanzia (môžeš pridať heuristické poradie)
    remaining = [i for i in range(n) if i not in partial_perm]
    for nxt in remaining:
        partial_perm.append(nxt)
        branch_and_bound(partial_perm, widths, data, best_cost, best_perm)
        partial_perm.pop()


if __name__ == "__main__":
    widths, data = load_data("Y-10_t.txt")
    best_cost = [float("inf")]
    best_perm = [None]
    branch_and_bound([], widths, data, best_cost, best_perm)
    print("\n✅ Best permutation:", best_perm[0])
    print("✅ Best cost:", best_cost[0])
