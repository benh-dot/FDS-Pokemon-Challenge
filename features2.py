"""Extra feature groups (v2) on top of features.py. Each group is a separate function so it can be ablated."""
import numpy as np, pandas as pd
from features import STATS, STATUSES, best_stab_matchup

RECOVER = {"rest","recover","softboiled","moonlight","morningsun","synthesis"}
SLEEPERS = {"sleeppowder","spore","hypnosis","lovelykiss","sing","sleeptalk"}
PHAZE = {"roar","whirlwind"}
HAZARD = {"spikes"}

def _series(b, team):
    tl = b["battle_timeline"]
    p1 = {p["name"]: 1.0 for p in team}; p2 = {}
    d, h1, h2 = [], [], []
    for t in tl:
        s1, s2 = t["p1_pokemon_state"], t["p2_pokemon_state"]
        p1[s1["name"]] = s1["hp_pct"]; p2[s2["name"]] = s2["hp_pct"]
        a = sum(p1.values()); c = sum(p2.values()) + max(0, 6 - len(p2))
        h1.append(a); h2.append(c); d.append(a - c)
    return np.array(d), np.array(h1), np.array(h2), p1, p2

def extra_features(b, chart, lut):
    f = {}
    team = b["p1_team_details"]; tl = b["battle_timeline"]
    d, h1, h2, p1, p2 = _series(b, team)
    # --- A: momentum ---
    x = np.arange(10)
    f["A_slope10"] = np.polyfit(x, d[-10:], 1)[0]
    f["A_slope20"] = np.polyfit(np.arange(20), d[-20:], 1)[0]
    f["A_mean_last5"] = d[-5:].mean(); f["A_mean_last10"] = d[-10:].mean()
    f["A_std10"] = d[-10:].std()
    f["A_max"] = d.max(); f["A_min"] = d.min()
    f["A_h1_drop10"] = h1[-11] - h1[-1]; f["A_h2_drop10"] = h2[-11] - h2[-1]
    f["A_drop_ratio10"] = (h2[-11] - h2[-1]) - (h1[-11] - h1[-1])
    f["A_ratio"] = (h1[-1] + .5) / (h2[-1] + .5)
    f["A_first_faint_p1"] = next((i for i, v in enumerate(h1) if i and h1[i] < np.floor(h1[0] - 0.5 + 1e-9) + 6 and False), 30)
    ff1 = ff2 = 30
    for i, t in enumerate(tl):
        if ff1 == 30 and t["p1_pokemon_state"]["status"] == "fnt": ff1 = i
        if ff2 == 30 and t["p2_pokemon_state"]["status"] == "fnt": ff2 = i
    f["A_first_faint_p1"], f["A_first_faint_p2"] = ff1, ff2
    f["A_first_blood"] = ff1 - ff2
    # --- B: end state ---
    last = {}
    for t in tl:
        last[("1", t["p1_pokemon_state"]["name"])] = t["p1_pokemon_state"]["status"]
        last[("2", t["p2_pokemon_state"]["name"])] = t["p2_pokemon_state"]["status"]
    for side in "12":
        sts = [v for (s, _), v in last.items() if s == side]
        f[f"B_p{side}_n_statused"] = sum(v not in ("nostatus", "fnt") for v in sts)
        f[f"B_p{side}_n_slp"] = sum(v == "slp" for v in sts)
        f[f"B_p{side}_n_par"] = sum(v == "par" for v in sts)
    f["B_statused_diff"] = f["B_p2_n_statused"] - f["B_p1_n_statused"]
    alive1 = sum(v > 0 for v in p1.values()); alive2 = sum(v > 0 for v in p2.values()) + max(0, 6 - len(p2))
    f["B_alive1"], f["B_alive2"], f["B_alive_diff"] = alive1, alive2, alive1 - alive2
    n1, n2 = tl[-1]["p1_pokemon_state"]["name"], tl[-1]["p2_pokemon_state"]["name"]
    tmap = {p["name"]: p for p in team}
    a1 = tmap[n1]; a1t = [t for t in a1["types"] if t != "notype"]
    a2 = lut.get(n2)
    if a2:
        a2t = a2["types"]
        f["B_act_spe_diff"] = a1["base_spe"] - a2["spe"]
        f["B_act_bst_diff"] = sum(a1[f"base_{s}"] for s in STATS) - sum(a2[s] for s in STATS)
        f["B_act_mult_1v2"] = best_stab_matchup(chart, a1t, a2t)
        f["B_act_mult_2v1"] = best_stab_matchup(chart, a2t, a1t)
        f["B_act_mult_diff"] = f["B_act_mult_1v2"] - f["B_act_mult_2v1"]
        f["B_act_def_diff"] = a1["base_def"] + a1["base_spd"] - a2["def"] - a2["spd"]
        f["B_act_off_diff"] = a1["base_atk"] + a1["base_spa"] - a2["atk"] - a2["spa"]
    # --- C: hp-weighted team value ---
    w1 = sum(p1[p["name"]] * sum(p[f"base_{s}"] for s in STATS) for p in team)
    w2 = sum(v * sum(lut[n][s] for s in STATS) for n, v in p2.items() if n in lut)
    unseen = max(0, 6 - len(p2)); avg = np.mean([sum(v[s] for s in STATS) for v in lut.values()])
    w2 += unseen * avg
    f["C_bst_hp_p1"], f["C_bst_hp_p2"], f["C_bst_hp_diff"] = w1, w2, w1 - w2
    f["C_abs_hp_diff"] = sum(p1[p["name"]] * p["base_hp"] for p in team) - (
        sum(v * lut[n]["hp"] for n, v in p2.items() if n in lut) + unseen * np.mean([v["hp"] for v in lut.values()]))
    # --- D: moves / items ---
    for side, key in (("1", "p1_move_details"), ("2", "p2_move_details")):
        ms = [t[key] for t in tl if t[key]]
        names = [m["name"] for m in ms]
        f[f"D_p{side}_mean_bp"] = np.mean([m["base_power"] for m in ms]) if ms else 0
        f[f"D_p{side}_recover"] = sum(n in RECOVER for n in names)
        f[f"D_p{side}_sleepmv"] = sum(n in SLEEPERS for n in names)
        f[f"D_p{side}_phaze"] = sum(n in PHAZE for n in names)
        f[f"D_p{side}_spikes"] = sum(n in HAZARD for n in names)
        f[f"D_p{side}_n_unique"] = len(set(names))
        f[f"D_p{side}_prio"] = sum(m["priority"] > 0 for m in ms)
        f[f"D_p{side}_last10_status"] = sum(1 for t in tl[-10:] if t[key] and t[key]["category"] == "status")
    for it in ("leftovers", "berserkgene", "mintberry", "miracleberry", "goldberry", "bitterberry"):
        f[f"D_item_{it}"] = sum(p.get("item") == it for p in team)
    # --- E: boosts at end ---
    s1, s2 = tl[-1]["p1_pokemon_state"]["boosts"], tl[-1]["p2_pokemon_state"]["boosts"]
    for k in ("atk", "def", "spa", "spd", "spe"):
        f[f"E_b1_{k}"], f[f"E_b2_{k}"] = s1[k], s2[k]
    f["E_boost_end_diff"] = sum(s1.values()) - sum(s2.values())
    return f

def make_extra(battles, chart, lut):
    return pd.DataFrame([extra_features(b, chart, lut) for b in battles])
