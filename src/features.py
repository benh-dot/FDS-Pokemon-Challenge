"""Feature engineering for the Pokemon Gen2 OU battle prediction challenge.
Every feature is computed only from the provided data (team, p2 lead, first 30 turns, type chart)."""
import json
import numpy as np
import pandas as pd

STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
STATUSES = ["slp", "frz", "par", "brn", "psn", "tox"]
TYPES = ["normal","fire","water","electric","grass","ice","fighting","poison","ground","flying",
         "psychic","bug","rock","ghost","dragon","dark","steel"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def load_type_chart(path):
    df = pd.read_csv(path)
    return {(r.attacking_type, r.defending_type): r.multiplier for r in df.itertuples()}


def mult(chart, atk_t, def_types):
    m = 1.0
    for d in def_types:
        if d != "notype":
            m *= chart.get((atk_t, d), 1.0)
    return m


def side_timeline_feats(tl, side, team_names, n_team=6):
    """Features for one side ('p1'/'p2') from the 30-turn timeline."""
    f = {}
    st = [t[f"{side}_pokemon_state"] for t in tl]
    mv = [t[f"{side}_move_details"] for t in tl]
    hp = np.array([s["hp_pct"] for s in st], float)
    f["hp_last"] = hp[-1]
    f["hp_mean"] = hp.mean()
    f["hp_min"] = hp.min()
    f["hp_t10"] = hp[min(9, len(hp)-1)]
    f["hp_t20"] = hp[min(19, len(hp)-1)]
    # last known HP of each seen pokemon; unseen assumed full health
    last = {}
    for s in st:
        last[s["name"]] = s["hp_pct"]
    f["n_seen"] = len(last)
    f["n_fainted"] = sum(1 for v in last.values() if v <= 0)
    f["team_hp_sum"] = sum(last.values()) + max(0, n_team - len(last))
    f["switches"] = sum(1 for a, b in zip(st[:-1], st[1:]) if a["name"] != b["name"])
    for s_ in STATUSES:
        f[f"status_{s_}_turns"] = sum(1 for s in st if s["status"] == s_)
    f["status_any_turns"] = sum(1 for s in st if s["status"] not in ("nostatus",))
    f["status_last"] = int(st[-1]["status"] != "nostatus")
    f["n_effects_turns"] = sum(1 for s in st if s["effects"] and s["effects"] != ["noeffect"])
    for b in ["atk", "def", "spa", "spd", "spe"]:
        vals = [s["boosts"].get(b, 0) for s in st]
        f[f"boost_{b}_mean"] = np.mean(vals)
    f["boost_sum_last"] = sum(st[-1]["boosts"].values())
    used = [m for m in mv if m]
    f["n_moves"] = len(used)
    f["no_move_turns"] = len(mv) - len(used)
    f["status_move_frac"] = np.mean([m["category"] == "status" for m in used]) if used else 0
    f["bp_mean"] = np.mean([m["base_power"] for m in used]) if used else 0
    f["priority_moves"] = sum(1 for m in used if m["priority"] > 0)
    f["acc_mean"] = np.mean([m["accuracy"] for m in used]) if used else 0
    f["n_unique_moves"] = len({m["name"] for m in used})
    return f


def featurize(b, chart):
    f = {}
    team = b["p1_team_details"]
    lead = b["p2_lead_details"]
    for s in STATS:
        v = np.array([p[f"base_{s}"] for p in team], float)
        f[f"p1_{s}_mean"], f[f"p1_{s}_max"], f[f"p1_{s}_min"] = v.mean(), v.max(), v.min()
        f[f"p2lead_{s}"] = lead[f"base_{s}"]
        f[f"lead_diff_{s}"] = v.mean() - lead[f"base_{s}"]
    f["p1_bst_mean"] = np.mean([sum(p[f"base_{s}"] for s in STATS) for p in team])
    f["p2lead_bst"] = sum(lead[f"base_{s}"] for s in STATS)
    f["p1_n_faster_than_lead"] = sum(p["base_spe"] > lead["base_spe"] for p in team)
    # type counts on P1 team
    for t in TYPES:
        f[f"p1_n_{t}"] = sum(t in p["types"] for p in team)
    for t in TYPES:
        f[f"p2lead_is_{t}"] = int(t in lead["types"])
    # type matchup: P1 pokemon types vs P2 lead and reverse
    best_vs, best_by = [], []
    for p in team:
        atk = max(mult(chart, t, lead["types"]) for t in p["types"] if t != "notype")
        dfn = max(mult(chart, t, p["types"]) for t in lead["types"] if t != "notype")
        best_vs.append(atk); best_by.append(dfn)
    f["tm_p1_off_max"], f["tm_p1_off_mean"] = max(best_vs), np.mean(best_vs)
    f["tm_p1_n_se"] = sum(x >= 2 for x in best_vs)
    f["tm_p2_off_max"], f["tm_p2_off_mean"] = max(best_by), np.mean(best_by)
    f["tm_p1_n_threatened"] = sum(x >= 2 for x in best_by)
    f["tm_net"] = f["tm_p1_off_mean"] - f["tm_p2_off_mean"]
    # items
    items = [p.get("item", "") for p in team]
    f["p1_n_leftovers"] = sum(i == "leftovers" for i in items)
    f["p1_n_noitem"] = sum(i in ("", "noitem", None) for i in items)
    f["p1_n_unique_items"] = len(set(items))
    # timeline
    tl = b["battle_timeline"]
    a = side_timeline_feats(tl, "p1", None)
    c = side_timeline_feats(tl, "p2", None)
    for k, v in a.items(): f[f"p1_{k}"] = v
    for k, v in c.items(): f[f"p2_{k}"] = v
    for k in ["hp_last", "hp_mean", "hp_t10", "hp_t20", "team_hp_sum", "n_fainted", "n_seen",
              "switches", "status_any_turns", "bp_mean", "status_move_frac"]:
        f[f"diff_{k}"] = a[k] - c[k]
    f["hp_diff_slope"] = np.polyfit(np.arange(len(tl)),
        [t["p1_pokemon_state"]["hp_pct"] - t["p2_pokemon_state"]["hp_pct"] for t in tl], 1)[0]
    return f


def build(path, chart_path, with_target=True):
    chart = load_type_chart(chart_path)
    data = load_jsonl(path)
    X = pd.DataFrame([featurize(b, chart) for b in data])
    ids = np.array([b["battle_id"] for b in data])
    y = np.array([b["player_won"] for b in data]) if with_target else None
    return X, y, ids
