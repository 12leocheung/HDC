"""
experiment4.py — reading out SETS: all, none, and how many.

Task: each person owns 0-4 pets. Facts bind(R_pet, person) -> pet share ONE
key, so their traces superpose in the same bucket. A count fact
bind(R_count, person) -> COUNT_k is also stored for every person.

Methods:
  threshold   rank pets by similarity, accept above a fixed cutoff theta
              (tuned once at the smallest N, then frozen — the honest
              naive baseline)
  deflate     retrieve top candidate -> VERIFY it is stored -> SUBTRACT its
              trace from the readout signal -> repeat until verification
              fails. Emptiness = first candidate already fails.
  deflate_cal same, but the verification margin is SELF-CALIBRATED per N:
              quiz known members vs known non-members, put the bar at the
              midpoint (the learned-margin idea, pull-only supervision
              from the memory's own contents).
  deflate_cnt calibrated deflation + read the stored count first; retrieve
              exactly that many members, cross-checking with verification.

Metrics vs N: exact-set accuracy (persons with >=1 pet) and correct-empty
rate (persons with 0 pets).
"""

from __future__ import annotations

import numpy as np

from experiment import D, cleanup, normalize_rows
from experiment3 import rand_vecs, LSHMem

rng = np.random.default_rng(31)
MAX_SET = 6


def build_world(n_persons):
    persons = rand_vecs(n_persons)
    pets = rand_vecs(n_persons)          # pet inventory, random IDs
    counts = rand_vecs(5)                # COUNT_0 .. COUNT_4 symbols
    R_pet, R_count = rand_vecs(2)

    sizes = rng.choice([0, 1, 2, 3, 4], size=n_persons,
                       p=[0.20, 0.25, 0.25, 0.20, 0.10])
    owns = [rng.choice(n_persons, size=k, replace=False) for k in sizes]

    keys, vals = [], []
    for p in range(n_persons):
        k_pet = (R_pet * persons[p]).astype(np.int8)
        for pet in owns[p]:
            keys.append(k_pet)
            vals.append(pets[pet])
        keys.append((R_count * persons[p]).astype(np.int8))
        vals.append(counts[sizes[p]])
    keys, vals = np.array(keys), np.array(vals)
    mem = LSHMem(keys, vals)
    M_flat = np.sum(keys.astype(np.int64) * vals.astype(np.int64), axis=0)
    return dict(persons=persons, pets=pets, pets_f=normalize_rows(pets),
                counts=counts, counts_f=normalize_rows(counts),
                R_pet=R_pet, R_count=R_count, sizes=sizes, owns=owns,
                mem=mem, M_flat=M_flat)


def verify_score(mem, leaf, working, key, cand_vec):
    trace = key.astype(np.float64) * cand_vec.astype(np.float64)
    norm = np.linalg.norm(working.astype(np.float64))
    return float(working @ trace) / (norm * np.sqrt(D)) if norm else 0.0


def deflate_readout(w, person_id, margin):
    """Retrieve-verify-subtract loop. Returns the set of pet ids."""
    key = (w["R_pet"] * w["persons"][person_id]).astype(np.int8)
    mem = w["mem"]
    leaf = mem._hash(key)
    working = mem.leaf_M[leaf].copy()
    found = set()
    for _ in range(MAX_SET):
        noisy = working * key.astype(np.int64)
        cand = int(cleanup(noisy, w["pets_f"])[0])
        if cand in found:
            break
        # support per-leaf margin dict or scalar
        if isinstance(margin, dict):
            m = margin.get(leaf, margin.get("_global", 0.1))
        else:
            m = margin
        if verify_score(mem, leaf, working, key, w["pets"][cand]) < m:
            break                       # stopping rule: next candidate not real
        found.add(cand)
        working -= key.astype(np.int64) * w["pets"][cand].astype(np.int64)
    return found


def read_count(w, person_id, margin):
    key = (w["R_count"] * w["persons"][person_id]).astype(np.int8)
    mem = w["mem"]
    leaf = mem._hash(key)
    working = mem.leaf_M[leaf]
    noisy = working * key.astype(np.int64)
    cand = int(cleanup(noisy, w["counts_f"])[0])
    if isinstance(margin, dict):
        m = margin.get(leaf, margin.get("_global", 0.1))
    else:
        m = margin
    if verify_score(mem, leaf, working, key, w["counts"][cand]) < m:
        return None
    return cand


def calibrate_margin(w, n_quiz=300):
    """Self-quiz calibration, done right on the third attempt.
    v1 midpoint-of-means: bar inside the negative tail -> leaks (0.83).
    v2 random-non-member quantile: WRONG DISTRIBUTION -> collapse (0.00).
       Deflation never tests a random candidate; it tests the argmax
       candidate — the best impostor out of the whole inventory, whose
       scores are max-order statistics, far above random draws.
    v3 (this): simulate the operating condition. Subtract the true members,
    run the actual selection on the residual, record what the surviving
    top impostor scores. Bar sits between the impostor tail and the
    member floor."""
    mem = w["mem"]
    # Per-leaf calibration: collect member / impostor top scores per leaf
    leaf_member = {}
    leaf_neg = {}
    persons_all = np.arange(len(w["sizes"]))
    sampled = rng.choice(persons_all, size=min(n_quiz, len(persons_all)),
                         replace=False)
    for p in sampled:
        key = (w["R_pet"] * w["persons"][int(p)]).astype(np.int8)
        leaf = mem._hash(key)
        working = mem.leaf_M[leaf].copy()
        # record true-member scores on the intact working signal
        for pet in w["owns"][int(p)]:
            leaf_member.setdefault(leaf, []).append(
                verify_score(mem, leaf, working, key, w["pets"][pet]))
        # remove ground truth, then see what the selector digs up
        for pet in w["owns"][int(p)]:
            working -= key.astype(np.int64) * w["pets"][pet].astype(np.int64)
        noisy = working * key.astype(np.int64)
        cand = int(cleanup(noisy, w["pets_f"])[0])
        leaf_neg.setdefault(leaf, []).append(
            verify_score(mem, leaf, working, key, w["pets"][cand]))

    per_leaf_margin = {}
    margins = []
    for leaf, neg_scores in leaf_neg.items():
        member_scores = leaf_member.get(leaf, [])
        lo = float(np.quantile(neg_scores, 0.99)) if neg_scores else 0.0
        hi = float(np.quantile(member_scores, 0.10)) if member_scores else lo * 2
        m = (lo + hi) / 2
        per_leaf_margin[leaf] = m
        margins.append(m)

    # global fallback margin for unseen leaves
    global_margin = float(np.median(margins)) if margins else 0.1
    per_leaf_margin["_global"] = global_margin
    return per_leaf_margin


def flat_threshold_readout(w, person_id, theta):
    """The pre-architecture baseline: one flat superposition, fixed cutoff."""
    key = (w["R_pet"] * w["persons"][person_id]).astype(np.int64)
    nf = (w["M_flat"] * key).astype(np.float32)
    nrm = np.linalg.norm(nf)
    sims = (w["pets_f"] @ nf / nrm) if nrm else np.zeros(len(w["pets"]))
    return set(np.nonzero(sims > theta)[0].tolist())


def threshold_readout(w, person_id, theta):
    key = (w["R_pet"] * w["persons"][person_id]).astype(np.int8)
    mem = w["mem"]
    leaf = mem._hash(key)
    noisy = mem.leaf_M[leaf] * key.astype(np.int64)
    nf = noisy.astype(np.float32)
    nrm = np.linalg.norm(nf)
    sims = (w["pets_f"] @ nf / nrm) if nrm else np.zeros(len(w["pets"]))
    return set(np.nonzero(sims > theta)[0].tolist())


def evaluate(w, method, param):
    with_pets = [p for p in range(len(w["sizes"])) if w["sizes"][p] > 0]
    without = [p for p in range(len(w["sizes"])) if w["sizes"][p] == 0]
    eval_with = rng.choice(with_pets, size=min(150, len(with_pets)), replace=False)
    eval_without = rng.choice(without, size=min(75, len(without)), replace=False)

    def get_set(p):
        if method == "flat_threshold":
            return flat_threshold_readout(w, p, param)
        if method == "threshold":
            return threshold_readout(w, p, param)
        if method in ("deflate", "deflate_cal"):
            return deflate_readout(w, p, param)
        if method == "deflate_cnt":
            margin = param
            cnt = read_count(w, p, margin)
            if cnt == 0:
                return set()
            found = deflate_readout(w, p, margin)
            if cnt is not None and len(found) > cnt:
                found = set(list(found)[:cnt])
            return found

    exact = float(np.mean([get_set(p) == set(w["owns"][p].tolist())
                           for p in eval_with]))
    empty = float(np.mean([get_set(p) == set() for p in eval_without]))
    return exact, empty


def main():
    n_persons_sweep = [300, 1000, 3000, 6000, 20000]
    methods = ["flat_threshold", "threshold", "deflate", "deflate_cal", "deflate_cnt"]
    res = {m: {"exact": [], "empty": [], "N": []} for m in methods}

    # tune theta once at the smallest N, then freeze
    w0 = build_world(n_persons_sweep[0])
    best_theta, best = 0.3, -1
    best_ftheta, fbest = 0.3, -1
    for theta in np.arange(0.05, 0.61, 0.05):
        e, z = evaluate(w0, "threshold", theta)
        if e + z > best:
            best, best_theta = e + z, float(theta)
        e, z = evaluate(w0, "flat_threshold", theta)
        if e + z > fbest:
            fbest, best_ftheta = e + z, float(theta)
    print(f"theta tuned once: bucketed={best_theta:.2f} flat={best_ftheta:.2f}")

    FIXED_MARGIN = 0.10
    for n_persons in n_persons_sweep:
        w = build_world(n_persons)
        N = sum(w["sizes"]) + n_persons
        cal = calibrate_margin(w)
        for m, param in [("flat_threshold", best_ftheta),
                         ("threshold", best_theta), ("deflate", FIXED_MARGIN),
                         ("deflate_cal", cal), ("deflate_cnt", cal)]:
            exact, empty = evaluate(w, m, param)
            res[m]["exact"].append(exact)
            res[m]["empty"].append(empty)
            res[m]["N"].append(N)
        cal_val = cal.get("_global", cal) if isinstance(cal, dict) else cal
        print(f"N={N:6d} (cal margin {cal_val:.3f})  " + "  ".join(
            f"{m}: set={res[m]['exact'][-1]:.2f}/none={res[m]['empty'][-1]:.2f}"
            for m in methods))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"flat_threshold": "flat memory + threshold (pre-architecture)",
              "threshold": "fixed threshold",
              "deflate": "deflation + verify (fixed margin)",
              "deflate_cal": "deflation + self-calibrated margin",
              "deflate_cnt": "deflation + stored count"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for i, metric in enumerate(["exact", "empty"]):
        ax = axes[i]
        for m in methods:
            ax.plot(res[m]["N"], res[m][metric], marker="o", label=labels[m])
        ax.set_xscale("log")
        ax.set_xlabel("N — total stored facts")
        ax.set_ylabel("exact-set accuracy" if metric == "exact"
                      else "correct \u2018none\u2019 rate")
        ax.set_title("\u201cName ALL of Leo\u2019s pets\u201d" if metric == "exact"
                     else "\u201cDoes Leo have any pets?\u201d \u2192 no")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig("set_readout.png", dpi=150)
    print("\nSaved set_readout.png")


if __name__ == "__main__":
    main()