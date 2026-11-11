"""
Paired significance test (exact McNemar) between two AgentMove result folders.

Both folders must contain prediction files for the same users, e.g.
  results/baseline_200/Shanghai/agentmove/gpt4omini-or/agent_move_v6
  results/baseline_200/Shanghai/agentmove/gpt4omini-or/llmzs

Usage:
  python mcnemar.py <folder_A> <folder_B> [k]
  k = top-k cutoff (default 1 -> acc@1; use 5 for acc@5)
"""
import glob
import json
import os
import sys
from math import comb


def load_hits(folder, k):
    """Map (user, trajectory) -> True/False: was the true place in the top-k predictions?"""
    hits = {}
    for path in glob.glob(os.path.join(folder, "*.json")):
        # file name looks like: <model>_<method>_<user>_<traj>_True.json
        parts = os.path.basename(path).split("_")
        key = (parts[-3], parts[-2])
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        preds = data.get("prediction") or []
        if not isinstance(preds, list):
            preds = [preds]
        top_k = [str(p) for p in preds[:k]]
        hits[key] = str(data.get("true")) in top_k
    return hits


def exact_mcnemar(b, c):
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    folder_a, folder_b = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    a, b_hits = load_hits(folder_a, k), load_hits(folder_b, k)
    shared = sorted(set(a) & set(b_hits))
    if not shared:
        print("No matching users between the two folders.")
        sys.exit(1)

    both = sum(a[u] and b_hits[u] for u in shared)
    only_a = sum(a[u] and not b_hits[u] for u in shared)
    only_b = sum(b_hits[u] and not a[u] for u in shared)
    neither = len(shared) - both - only_a - only_b

    name_a = os.path.basename(os.path.normpath(folder_a))
    name_b = os.path.basename(os.path.normpath(folder_b))
    print(f"Matched users: {len(shared)}   (top-{k})")
    print(f"acc@{k}  A ({name_a}): {sum(a[u] for u in shared) / len(shared):.3f}")
    print(f"acc@{k}  B ({name_b}): {sum(b_hits[u] for u in shared) / len(shared):.3f}")
    print(f"Both right: {both} | only A right: {only_a} | only B right: {only_b} | both wrong: {neither}")
    p = exact_mcnemar(only_a, only_b)
    print(f"Exact McNemar p-value: {p:.4f}  ->  {'significant' if p < 0.05 else 'NOT significant'} at 0.05")


if __name__ == "__main__":
    main()