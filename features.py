"""Feature engineering for the FDS Pokemon Gen 2 OU battle prediction challenge.

Only competition data is used (train/test jsonl + supplied type chart).
Everything is computed per battle, so no information flows between train and test rows
(except the name -> base-stat lookup, built from the base stats listed in the data files,
which contains no labels).
"""
import json
import numpy as np
import pandas as pd

STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
STATUSES = ["slp", "par", "brn", "psn", "tox", "frz"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_type_chart(path):
    tc = pd.read_csv(path)
    return {(a, d): m for a, d, m in tc.itertuples(index=False)}


def build_stat_lookup(*battle_lists):
    """name -> base stats, collected from p1 teams and p2 leads (labels not used)."""
    lut = {}
    for battles in battle_lists:
        for b in battles:
            for p in b["p1_team_details"] + [b["p2_lead_details"]]:
                lut[p["name"]] = {
                    "types": [t for t in p["types"] if t != "notype"],
                    **{s: p[f"base_{s}"] for s in STATS},
                }
    return lut


def type_mult(chart, atk_type, def_types):
    m = 1.0
    for d in def_types:
        m *= chart.get((atk_type, d), 1.0)
    return m


def best_stab_matchup(chart, att_types, def_types):
    """Best type multiplier any of the attacker's own types has vs the defender."""
    if not att_types:
        return 1.0
    return max(type_mult(chart, t, def_types) for t in att_types)


def battle_features(b, chart, lut):
    f = {}
    team = b["p1_team_details"]
    lead = b["p2_lead_details"]

    # ---- static team features -------------------------------------------------
    for s in STATS:
        vals = [p[f"base_{s}"] for p in team]
        f[f"p1_mean_{s}"] = np.mean(vals)
        f[f"p1_max_{s}"] = np.max(vals)
        f[f"lead_{s}"] = lead[f"base_{s}"]
        f[f"diff_mean_{s}"] = np.mean(vals) - lead[f"base_{s}"]
    f["p1_bst"] = sum(f[f"p1_mean_{s}"] for s in STATS)
    f["lead_bst"] = sum(lead[f"base_{s}"] for s in STATS)

    lead_types = [t for t in lead["types"] if t != "notype"]
    mm = [best_stab_matchup(chart, [t for t in p["types"] if t != "notype"], lead_types) for p in team]
    f["p1_vs_lead_best_mult_mean"] = np.mean(mm)
    f["p1_vs_lead_n_superEff"] = sum(m > 1 for m in mm)
    rev = [best_stab_matchup(chart, lead_types, [t for t in p["types"] if t != "notype"]) for p in team]
    f["lead_vs_p1_best_mult_mean"] = np.mean(rev)
    f["lead_vs_p1_n_superEff"] = sum(m > 1 for m in rev)
    items = [p.get("item") for p in team]
    f["p1_n_leftovers"] = sum(i == "leftovers" for i in items)

    # ---- timeline -------------------------------------------------------------
    tl = b["battle_timeline"]
    p1_hp = {p["name"]: 1.0 for p in team}   # last seen hp per p1 mon (unseen = full)
    p2_hp = {}                                # last seen hp per revealed p2 mon
    p1_dmg = p2_dmg = 0.0                     # hp% damage inflicted on the other side
    prev1 = prev2 = None
    series_diff, series_p1_total, series_p2_total = [], [], []
    st_p1 = {s: 0 for s in STATUSES}
    st_p2 = {s: 0 for s in STATUSES}
    p1_switches = p2_switches = 0
    p1_moves = p2_moves = 0
    cat1 = {"physical": 0, "special": 0, "status": 0}
    cat2 = {"physical": 0, "special": 0, "status": 0}
    boost1 = boost2 = 0
    p1_no_move = p2_no_move = 0
    stat_turns_p1 = stat_turns_p2 = 0
    eff1 = eff2 = 0
    for t in tl:
        s1, s2 = t["p1_pokemon_state"], t["p2_pokemon_state"]
        p1_hp[s1["name"]] = s1["hp_pct"]
        p2_hp[s2["name"]] = s2["hp_pct"]
        if prev1 is not None:
            if s1["name"] != prev1["name"]:
                p1_switches += 1
            elif s1["hp_pct"] < prev1["hp_pct"]:
                p2_dmg += prev1["hp_pct"] - s1["hp_pct"]
            if s2["name"] != prev2["name"]:
                p2_switches += 1
            elif s2["hp_pct"] < prev2["hp_pct"]:
                p1_dmg += prev2["hp_pct"] - s2["hp_pct"]
        prev1, prev2 = s1, s2
        for st in STATUSES:
            st_p1[st] += s1["status"] == st
            st_p2[st] += s2["status"] == st
        stat_turns_p1 += s1["status"] not in ("nostatus", "fnt")
        stat_turns_p2 += s2["status"] not in ("nostatus", "fnt")
        for (m, c) in ((t["p1_move_details"], cat1), (t["p2_move_details"], cat2)):
            if m:
                c[m["category"]] = c.get(m["category"], 0) + 1
        p1_no_move += t["p1_move_details"] is None
        p2_no_move += t["p2_move_details"] is None
        boost1 += sum(s1["boosts"].values())
        boost2 += sum(s2["boosts"].values())
        eff1 += any(e != "noeffect" for e in s1["effects"])
        eff2 += any(e != "noeffect" for e in s2["effects"])
        p1_total = sum(p1_hp.values())
        p2_total = sum(p2_hp.values()) + max(0, 6 - len(p2_hp))  # unseen p2 mons assumed full
        series_p1_total.append(p1_total)
        series_p2_total.append(p2_total)
        series_diff.append(s1["hp_pct"] - s2["hp_pct"])

    p1_faint = sum(v <= 0 for v in p1_hp.values())
    p2_faint = sum(v <= 0 for v in p2_hp.values())
    f["p1_faint"] = p1_faint
    f["p2_faint"] = p2_faint
    f["faint_diff"] = p2_faint - p1_faint
    f["p2_revealed"] = len(p2_hp)
    f["p1_used"] = sum(1 for n in {t["p1_pokemon_state"]["name"] for t in tl})
    f["p1_team_hp"] = series_p1_total[-1]
    f["p2_team_hp"] = series_p2_total[-1]
    f["team_hp_diff"] = series_p1_total[-1] - series_p2_total[-1]
    for k in (5, 10, 15, 20, 25):
        f[f"team_hp_diff_t{k}"] = series_p1_total[k - 1] - series_p2_total[k - 1]
        f[f"active_hp_diff_t{k}"] = series_diff[k - 1]
    f["active_hp_diff_t30"] = series_diff[-1]
    f["team_hp_diff_mean"] = float(np.mean(np.array(series_p1_total) - np.array(series_p2_total)))
    f["team_hp_diff_slope"] = f["team_hp_diff"] - f["team_hp_diff_t20"]
    f["p1_dmg"], f["p2_dmg"] = p1_dmg, p2_dmg
    f["dmg_diff"] = p1_dmg - p2_dmg
    f["p1_switches"], f["p2_switches"] = p1_switches, p2_switches
    for st in STATUSES:
        f[f"p1_{st}_turns"], f[f"p2_{st}_turns"] = st_p1[st], st_p2[st]
        f[f"{st}_diff"] = st_p2[st] - st_p1[st]
    f["status_turns_diff"] = stat_turns_p2 - stat_turns_p1
    for c in ("physical", "special", "status"):
        f[f"p1_{c}_moves"], f[f"p2_{c}_moves"] = cat1[c], cat2[c]
    f["p1_no_move"], f["p2_no_move"] = p1_no_move, p2_no_move
    f["boost_sum_diff"] = boost1 - boost2
    f["effect_turns_diff"] = eff2 - eff1

    # last-state snapshot of the active mons
    last = tl[-1]
    f["last_p1_hp"] = last["p1_pokemon_state"]["hp_pct"]
    f["last_p2_hp"] = last["p2_pokemon_state"]["hp_pct"]
    for side, key in (("p1", "p1_pokemon_state"), ("p2", "p2_pokemon_state")):
        f[f"last_{side}_status"] = (["nostatus"] + STATUSES + ["fnt"]).index(last[key]["status"])

    # revealed p2 roster strength (base stats via name lookup from competition data)
    for s in STATS:
        vals = [lut[n][s] for n in p2_hp if n in lut]
        f[f"p2_seen_mean_{s}"] = np.mean(vals) if vals else np.nan
    f["p2_seen_bst"] = sum(f[f"p2_seen_mean_{s}"] for s in STATS)
    f["bst_seen_diff"] = f["p1_bst"] - f["p2_seen_bst"]
    # remaining (non-fainted) mons' strength
    alive1 = [p for p in team if p1_hp[p["name"]] > 0]
    f["p1_alive_bst"] = np.mean([sum(p[f"base_{s}"] for s in STATS) for p in alive1]) if alive1 else 0
    alive2 = [n for n, v in p2_hp.items() if v > 0 and n in lut]
    f["p2_alive_seen_bst"] = np.mean([sum(lut[n][s] for s in STATS) for n in alive2]) if alive2 else 0
    return f


def make_matrix(battles, chart, lut):
    X = pd.DataFrame([battle_features(b, chart, lut) for b in battles])
    ids = [b["battle_id"] for b in battles]
    return X, ids


def composition_features(battles):
    """Sparse counts: own team members, opponent Pokemon revealed, moves used by each side."""
    rows = []
    for b in battles:
        r = {}
        for p in b["p1_team_details"]:
            r["own_" + p["name"]] = 1
        for n in {t["p2_pokemon_state"]["name"] for t in b["battle_timeline"]}:
            r["opp_" + n] = 1
        for t in b["battle_timeline"]:
            for side, key in (("m1_", "p1_move_details"), ("m2_", "p2_move_details")):
                m = t[key]
                if m:
                    r[side + m["name"]] = r.get(side + m["name"], 0) + 1
        rows.append(r)
    return pd.DataFrame(rows).fillna(0)


def extra_features(b, chart, lut):
    """Second-generation features: HP-weighted strength, last-seen status per Pokemon,
    active-matchup at turn 30, damage-potential of moves used, faint timing, full HP trajectory."""
    f = {}
    team = {p["name"]: p for p in b["p1_team_details"]}
    tl = b["battle_timeline"]
    hp1 = {n: 1.0 for n in team}; hp2 = {}
    stat1 = {n: "nostatus" for n in team}; stat2 = {}
    faint_turn1, faint_turn2 = [], []
    traj = []
    for t in tl:
        s1, s2 = t["p1_pokemon_state"], t["p2_pokemon_state"]
        if hp1[s1["name"]] > 0 and s1["hp_pct"] <= 0:
            faint_turn1.append(t["turn"])
        if s2["name"] in hp2 and hp2[s2["name"]] > 0 and s2["hp_pct"] <= 0 or (s2["name"] not in hp2 and s2["hp_pct"] <= 0):
            faint_turn2.append(t["turn"])
        hp1[s1["name"]] = s1["hp_pct"]; stat1[s1["name"]] = s1["status"]
        hp2[s2["name"]] = s2["hp_pct"]; stat2[s2["name"]] = s2["status"]
        h1 = sum(hp1.values()); h2 = sum(hp2.values()) + max(0, 6 - len(hp2))
        traj.append(h1 - h2)
    # full trajectory of team-hp difference (every 2nd turn) + summary stats
    for k in range(1, 31, 2):
        f[f"traj_{k}"] = traj[k - 1]
    f["traj_min"], f["traj_max"] = min(traj), max(traj)
    f["traj_last5_mean"] = float(np.mean(traj[-5:]))
    f["traj_std"] = float(np.std(traj))
    f["p1_last_faint"] = max(faint_turn1) if faint_turn1 else 0
    f["p2_last_faint"] = max(faint_turn2) if faint_turn2 else 0
    f["faints_last10_diff"] = sum(x > 20 for x in faint_turn2) - sum(x > 20 for x in faint_turn1)

    # HP-weighted base-stat total of surviving mons
    def bst(p): return sum(p[f"base_{s}"] for s in STATS)
    f["p1_hpw_bst"] = sum(hp1[n] * bst(team[n]) for n in team)
    f["p2_hpw_bst"] = sum(v * (sum(lut[n][s] for s in STATS) if n in lut else 450) for n, v in hp2.items()) \
        + max(0, 6 - len(hp2)) * 480
    f["hpw_bst_diff"] = f["p1_hpw_bst"] - f["p2_hpw_bst"]

    # statuses carried by living mons at turn 30 (last seen)
    for side, hp, st in (("p1", hp1, stat1), ("p2", hp2, stat2)):
        alive = [n for n in hp if hp[n] > 0]
        f[f"{side}_alive_n"] = len(alive) + (max(0, 6 - len(hp)) if side == "p2" else 0)
        f[f"{side}_alive_slp"] = sum(st[n] == "slp" for n in alive)
        f[f"{side}_alive_par"] = sum(st[n] == "par" for n in alive)
        f[f"{side}_alive_tox"] = sum(st[n] in ("tox", "psn") for n in alive)
        f[f"{side}_alive_brn_frz"] = sum(st[n] in ("brn", "frz") for n in alive)
        f[f"{side}_alive_healthy_hp"] = sum(hp[n] for n in alive if st[n] == "nostatus")
    f["alive_diff"] = f["p1_alive_n"] - f["p2_alive_n"]
    f["alive_status_diff"] = (f["p2_alive_slp"] + f["p2_alive_par"] + f["p2_alive_tox"] + f["p2_alive_brn_frz"]) - \
                             (f["p1_alive_slp"] + f["p1_alive_par"] + f["p1_alive_tox"] + f["p1_alive_brn_frz"])

    # active matchup at turn 30 (who is on the field, speed, type advantage)
    last = tl[-1]
    a1, a2 = last["p1_pokemon_state"]["name"], last["p2_pokemon_state"]["name"]
    if a1 in team and a2 in lut:
        P1, P2 = team[a1], lut[a2]
        t1 = [t for t in P1["types"] if t != "notype"]
        f["act_spe_diff"] = P1["base_spe"] - P2["spe"]
        f["act_bst_diff"] = bst(P1) - sum(P2[s] for s in STATS)
        f["act_p1_best_mult"] = best_stab_matchup(chart, t1, P2["types"])
        f["act_p2_best_mult"] = best_stab_matchup(chart, P2["types"], t1)
        f["act_mult_diff"] = f["act_p1_best_mult"] - f["act_p2_best_mult"]
    else:
        for k in ("act_spe_diff", "act_bst_diff", "act_p1_best_mult", "act_p2_best_mult", "act_mult_diff"):
            f[k] = np.nan

    # expected damage potential of attacking moves used (base power * accuracy * STAB-free type mult vs opp active)
    pot1 = pot2 = 0.0; n_ko_moves1 = n_ko_moves2 = 0
    for t in tl:
        m1, m2 = t["p1_move_details"], t["p2_move_details"]
        n2, n1 = t["p2_pokemon_state"]["name"], t["p1_pokemon_state"]["name"]
        if m1 and m1["base_power"] > 0 and n2 in lut:
            mult = type_mult(chart, m1["type"], lut[n2]["types"])
            pot1 += m1["base_power"] * m1["accuracy"] * mult
        if m2 and m2["base_power"] > 0 and n1 in team:
            mult = type_mult(chart, m2["type"], [x for x in team[n1]["types"] if x != "notype"])
            pot2 += m2["base_power"] * m2["accuracy"] * mult
    f["pot1"], f["pot2"], f["pot_diff"] = pot1, pot2, pot1 - pot2

    # priority / healing / setup move usage
    heal = {"rest", "recover", "softboiled", "milkdrink", "moonlight", "morningsun", "synthesis", "slackoff"}
    setup = {"curse", "swordsdance", "bellydrum", "amnesia", "agility", "calmmind"}
    for side, key in (("p1", "p1_move_details"), ("p2", "p2_move_details")):
        names = [t[key]["name"] for t in tl if t[key]]
        f[f"{side}_heal_moves"] = sum(n in heal for n in names)
        f[f"{side}_setup_moves"] = sum(n in setup for n in names)
        f[f"{side}_distinct_moves"] = len(set(names))
    f["heal_diff"] = f["p2_heal_moves"] - f["p1_heal_moves"]
    return f


def make_extra_matrix(battles, chart, lut):
    return pd.DataFrame([extra_features(b, chart, lut) for b in battles])
