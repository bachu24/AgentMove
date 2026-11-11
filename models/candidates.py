"""Leak-free candidate generation for the agent_move_explore variant (thesis adaptation 3c).

The pool holds every venue that appears in ANY user's historical or current-day stays
(align file), never the test answers, so no target information leaks into the prompt.
"""
import ast
import json
import math
import os

_POOL = None


def _haversine_km(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


def load_pool(city_name, processed_dir="data/processed"):
    """venue_id(int) -> {'pos': (lon, lat), 'cat': str, 'addr': 'subdistrict, street'}"""
    global _POOL
    if _POOL is not None:
        return _POOL
    align = json.load(open(os.path.join(processed_dir, f"align_locations_{city_name}_trajectory_split_int.json")))
    test = json.load(open(os.path.join(processed_dir, f"test_dictionary_{city_name}_trajectory_split_int.json")))
    pool = {}
    for trajs in align.values():
        trajs = ast.literal_eval(trajs) if isinstance(trajs, str) else trajs
        for parts in trajs.values():
            for rows in parts.values():
                for r in rows:
                    pool[int(r[7])] = {"pos": (float(r[4]), float(r[5])), "cat": r[6], "addr": ""}
    for trajs in test.values():
        for t in trajs.values():
            for s in t.get("historical_stays_long", []) + t.get("context_stays", []):
                vid = int(s[3])
                if vid in pool and not pool[vid]["addr"]:
                    pool[vid]["addr"] = ", ".join(x for x in (s[5], s[7]) if x)
    _POOL = pool
    return pool


def nearest_unvisited(traj_seqs, city_name, k=20):
    """The k nearest venues the user has not visited, measured from the last current-day stay."""
    pool = load_pool(city_name)
    seen = {int(x[3]) for x in traj_seqs["historical_stays"] + traj_seqs["context_stays"]}
    lon, lat = traj_seqs["context_pos"][-1]
    scored = sorted((_haversine_km(lon, lat, *p["pos"]), vid) for vid, p in pool.items() if vid not in seen)
    return [(vid, pool[vid]["cat"], pool[vid]["addr"], round(d, 2)) for d, vid in scored[:k]]
