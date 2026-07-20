"""
experiment3.py — closing the three remaining problems.

PANEL 1 (Problem 1: similar entities confuse everything).
  World rebuilt with CORRELATED entities: clusters of 50 siblings sharing
  ~50% of their bits (Rex and Max are both dogs). Two architectures:
    naive : semantic vectors used as memory keys/values directly
    id    : random-ID indirection — semantics live only in an entrance
            table; the memory stores maximally-different random IDs.
  Prediction: correlation poisons the naive memory (LSH buckets fill with
  siblings whose crosstalk no longer averages out) while the ID memory
  keeps the interior of the system in the easy random-vector regime.

PANEL 2 (Problem 2: do errors compound over long chains?).
  Chains of length 2..8 over the ID architecture, N ~ 10,000 facts.
  Prediction: quantized restarts make each hop independent, so accuracy
  stays flat with depth; flat memory dies immediately at this N.

PANEL 3 (Problem 4: the entrance scan is linear).
  Entry snap latency vs inventory size: exact linear scan vs LSH
  approximate nearest neighbour (hash query, probe 8 buckets, scan only
  those). Recall measured against the exact scan at 20% query noise.
"""

from __future__ import annotations

import time
from itertools import combinations

import numpy as np

from experiment import D, LEAF_SIZE, VERIFY_MARGIN, cleanup, normalize_rows
from experiment2 import flip_bits

SEED = 23
rng = np.random.default_rng(SEED)


def rand_vecs(n):
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(n, D))


def rand_orth_vecs(n):
    """Generate n approximately orthogonal +/-1 vectors via Gram-Schmidt.
    Returns int8 matrix shape (n, D)."""
    # start from gaussian floats for stable orthogonalization
    V = rng.normal(size=(n, D)).astype(np.float32)
    for i in range(n):
        v = V[i]
        for j in range(i):
            u = V[j]
            proj = (v @ u) / (u @ u) if (u @ u) != 0 else 0.0
            v = v - proj * u
        norm = np.linalg.norm(v)
        if norm == 0:
            # fallback to fresh random vector
            v = rng.normal(size=(D,)).astype(np.float32)
            norm = np.linalg.norm(v)
        V[i] = v / norm
    return np.sign(V).astype(np.int8)


def correlated_vecs(n, cluster_size=50, mutate=0.15):
    """Entities in clusters: each is its cluster prototype with `mutate`
    of the bits flipped -> within-cluster cosine ~ (1-2m)^2 ~ 0.49."""
    n_clusters = int(np.ceil(n / cluster_size))
    protos = rand_vecs(n_clusters)
    out = np.empty((n, D), dtype=np.int8)
    for i in range(n):
        flips = np.where(rng.random(D) < mutate, -1, 1).astype(np.int8)
        out[i] = protos[i // cluster_size] * flips
    return out


class LSHMem:
    def __init__(self, keys, vals):
        n_facts = keys.shape[0]
        self.bits = max(1, int(np.ceil(np.log2(max(2, n_facts / LEAF_SIZE)))))
        self.H = rand_vecs(self.bits).astype(np.float32)
        self.n_leaves = 2 ** self.bits
        self.leaf_M = np.zeros((self.n_leaves, D), dtype=np.int64)
        self.leaf_count = np.zeros(self.n_leaves, dtype=np.int64)
        for f in range(n_facts):
            leaf = self._hash(keys[f])
            self.leaf_M[leaf] += keys[f].astype(np.int64) * vals[f].astype(np.int64)
            self.leaf_count[leaf] += 1
        self.leaf_M_norm = np.linalg.norm(self.leaf_M.astype(np.float64), axis=1)

    def _hash(self, key):
        bits = (self.H @ key.astype(np.float32)) > 0
        return int(np.dot(bits, 1 << np.arange(self.bits)))

    def hop(self, key, items_f, items_raw):
        leaf = self._hash(key)
        noisy = self.leaf_M[leaf] * key.astype(np.int64)
        cands = cleanup(noisy, items_f, top=2)
        for cand in cands:
            trace = key.astype(np.float32) * items_raw[cand].astype(np.float32)
            denom = self.leaf_M_norm[leaf] * np.sqrt(D)
            score = float(self.leaf_M[leaf] @ trace) / denom if denom else 0.0
            if score > VERIFY_MARGIN / np.sqrt(max(1, self.leaf_count[leaf])):
                return cand
        return cands[0]


# ======================================================================
# PANEL 1 — correlated entities: naive semantics vs random-ID indirection
# ======================================================================
def panel1():
    """Sweep how SIMILAR entities are (cluster tightness) at fixed 20% query
    noise. mutate=0.25 -> within-cluster cos~0.25; 0.15 -> ~0.49;
    0.05 -> ~0.81 (nearly identical siblings)."""
    n_pairs, n_sounds = 5000, 50
    mutates = [0.25, 0.15, 0.10, 0.05]
    tightness = [round((1 - 2 * m) ** 2, 2) for m in mutates]
    noise_p = 0.2
    acc = {"naive": [], "id": []}

    for m in mutates:
        R1, R2 = rand_vecs(2)
        person_sem = correlated_vecs(n_pairs, mutate=m)
        animal_sem = correlated_vecs(n_pairs, mutate=m)
        # use orthogonalized random IDs to keep IDs maximally different
        person_id, animal_id = rand_orth_vecs(n_pairs), rand_orth_vecs(n_pairs)
        sounds = rand_vecs(n_sounds)
        animal_sound = rng.integers(0, n_sounds, size=n_pairs)

        person_sem_f = normalize_rows(person_sem)
        animal_sem_f = normalize_rows(animal_sem)
        animal_id_f = normalize_rows(animal_id)
        sounds_f = normalize_rows(sounds)

        naive = LSHMem(
            np.vstack([(R1 * person_sem), (R2 * animal_sem)]).astype(np.int8),
            np.vstack([animal_sem, sounds[animal_sound]]))
        idmem = LSHMem(
            np.vstack([(R1 * person_id), (R2 * animal_id)]).astype(np.int8),
            np.vstack([animal_id, sounds[animal_sound]]))

        queries = rng.integers(0, n_pairs, size=200)
        c_naive = c_id = 0
        for q in queries:
            noisy = flip_bits(person_sem[q], noise_p, rng)
            snapped = cleanup(noisy, person_sem_f)[0]

            a = naive.hop((R1 * person_sem[snapped]).astype(np.int8),
                          animal_sem_f, animal_sem)
            s = naive.hop((R2 * animal_sem[a]).astype(np.int8), sounds_f, sounds)
            c_naive += (s == animal_sound[q])

            a = idmem.hop((R1 * person_id[snapped]).astype(np.int8),
                          animal_id_f, animal_id)
            s = idmem.hop((R2 * animal_id[a]).astype(np.int8), sounds_f, sounds)
            c_id += (s == animal_sound[q])
        acc["naive"].append(c_naive / len(queries))
        acc["id"].append(c_id / len(queries))
        print(f"P1 sibling-cos={tightness[mutates.index(m)]:.2f}  "
              f"naive={acc['naive'][-1]:.2f}  id={acc['id'][-1]:.2f}")
    return tightness, acc


# ======================================================================
# PANEL 2 — chain depth 2..8 with random IDs + quantized restarts
# ======================================================================
def panel2():
    depths = [2, 3, 4, 5, 6, 8]
    acc = {"flat_quant": [], "content_quant": []}
    R = rand_vecs(1)

    for L in depths:
        n_chains = max(20, 10_000 // L)
        ents = rand_vecs(n_chains * (L + 1)).reshape(n_chains, L + 1, D)
        keys = np.vstack([(R * ents[c, i]).astype(np.int8)
                          for c in range(n_chains) for i in range(L)])
        vals = np.vstack([ents[c, i + 1]
                          for c in range(n_chains) for i in range(L)])
        all_ents = ents.reshape(-1, D)
        all_ents_f = normalize_rows(all_ents)

        mem = LSHMem(keys, vals)
        M_flat = np.sum(keys.astype(np.int64) * vals.astype(np.int64), axis=0)

        queries = rng.integers(0, n_chains, size=120)
        c_flat = c_lsh = 0
        for q in queries:
            target = q * (L + 1) + L
            cur = ents[q, 0]
            for _ in range(L):
                noisy = M_flat * (R[0] * cur).astype(np.int64)
                cur = all_ents[cleanup(noisy, all_ents_f)[0]]
            c_flat += np.array_equal(cur, all_ents[target])
            cur = ents[q, 0]
            for _ in range(L):
                cur = all_ents[mem.hop((R[0] * cur).astype(np.int8),
                                       all_ents_f, all_ents)]
            c_lsh += np.array_equal(cur, all_ents[target])
        acc["flat_quant"].append(c_flat / len(queries))
        acc["content_quant"].append(c_lsh / len(queries))
        print(f"P2 hops={L}  flat={acc['flat_quant'][-1]:.2f}  "
              f"content={acc['content_quant'][-1]:.2f}")
    return depths, acc


# ======================================================================
# PANEL 3 — entry snap: exact scan vs LSH ANN
# ======================================================================
def flip_bits_d(vec, p, d):
    mask = np.where(rng.random(d) < p, -1, 1).astype(np.int8)
    return (vec * mask).astype(np.int8)


def panel3():
    """Entry snap latency: exact scan vs multi-table LSH with verified
    fallback. T independent tables halve the chance that ALL tables miss;
    a low-confidence result falls back to the exact scan, so the returned
    answer always matches the exact scan (recall 1.0 by construction) —
    the question is purely how often the slow path fires."""
    D3 = 1024
    sizes = [10_000, 40_000, 160_000]
    n_queries = 100
    T_TABLES, K_BITS, CONF_WINDOW, PROBES_TOTAL = 4, 9, 6, 32
    FALLBACK_SIM_MARGIN = 0.75   # accept if best_sim > margin * (1-2p)
    res = {}

    for noise_p in (0.1, 0.2):
        res[noise_p] = {"scan_ms": [], "lsh_ms": [], "fallback": []}
        accept_thresh = FALLBACK_SIM_MARGIN * (1 - 2 * noise_p)
        for size in sizes:
            items = rng.choice(np.array([-1, 1], dtype=np.int8), size=(size, D3))
            items_f = items.astype(np.float32) / np.sqrt(D3)

            tables = []
            weights = 1 << np.arange(K_BITS)
            for _ in range(T_TABLES):
                H = rng.choice(np.array([-1.0, 1.0], dtype=np.float32),
                               size=(K_BITS, D3))
                hashes = ((items.astype(np.float32) @ H.T) > 0) @ weights
                tables.append((H, {h: np.nonzero(hashes == h)[0]
                                   for h in np.unique(hashes)}))

            q_ids = rng.integers(0, size, size=n_queries)
            q_vecs = [flip_bits_d(items[i], noise_p, D3) for i in q_ids]

            t0 = time.perf_counter()
            for v in q_vecs:
                int(np.argmax(items_f @ (v.astype(np.float32) / np.sqrt(D3))))
            scan_ms = (time.perf_counter() - t0) / n_queries * 1000

            t0 = time.perf_counter()
            n_fallback = 0
            for v in q_vecs:
                vf = v.astype(np.float32) / np.sqrt(D3)
                best_sim = -np.inf
                # distribute a global probe budget across tables adaptively
                zt = None
                probes_used = 0
                for H, buckets in tables:
                    if probes_used >= PROBES_TOTAL:
                        break
                    z = H @ v.astype(np.float32)
                    base = (z > 0).astype(np.int64)
                    conf = np.argsort(np.abs(z))[:CONF_WINDOW]
                    # per-table probe budget proportional to remaining budget
                    remaining = PROBES_TOTAL - probes_used
                    per_table_budget = max(1, remaining // (len(tables)))
                    probes = 0
                    done = False
                    for n_flip in range(0, 3):
                        for combo in combinations(conf, n_flip):
                            if probes >= per_table_budget or probes_used >= PROBES_TOTAL:
                                done = True
                                break
                            b = base.copy()
                            for bit in combo:
                                b[bit] ^= 1
                            ids = buckets.get(int(np.dot(b, weights)))
                            probes += 1
                            probes_used += 1
                            if ids is not None and len(ids):
                                sims = items_f[ids] @ vf
                                s = float(np.max(sims))
                                if s > best_sim:
                                    best_sim = s
                        if done:
                            break
                    if best_sim > accept_thresh:
                        break                    # early accept, skip tables
                if best_sim <= accept_thresh:
                    float(np.max(items_f @ vf))  # exact fallback
                    n_fallback += 1
            lsh_ms = (time.perf_counter() - t0) / n_queries * 1000

            res[noise_p]["scan_ms"].append(scan_ms)
            res[noise_p]["lsh_ms"].append(lsh_ms)
            res[noise_p]["fallback"].append(n_fallback / n_queries)
            print(f"P3 noise={noise_p:.1f} size={size:7d}  scan={scan_ms:6.2f}ms  "
                  f"lsh={lsh_ms:6.2f}ms  fallback={n_fallback}/{n_queries}")
    return sizes, res


# ======================================================================
def main():
    p1x, p1 = panel1()
    p2x, p2 = panel2()
    p3x, p3 = panel3()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.plot(p1x, p1["naive"], marker="o",
            label="naive: semantic vectors in memory")
    ax.plot(p1x, p1["id"], marker="o",
            label="fix: random-ID indirection")
    ax.set_xlabel("within-cluster similarity (cosine)")
    ax.set_ylabel("two-hop accuracy (20% query noise)")
    ax.set_title("1. How similar can entities get?")
    ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(p2x, p2["flat_quant"], marker="o", label="flat + quantize")
    ax.plot(p2x, p2["content_quant"], marker="o",
            label="content-addressed + quantize")
    ax.set_xlabel("chain length (hops)")
    ax.set_ylabel("full-chain accuracy")
    ax.set_title("2. Deep chains, N ≈ 10,000 facts")
    ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend()

    ax = axes[2]
    ax.plot(p3x, p3[0.1]["scan_ms"], marker="o", label="exact linear scan")
    ax.plot(p3x, p3[0.1]["lsh_ms"], marker="o",
            label="multi-table LSH, 10% query noise")
    ax.plot(p3x, p3[0.2]["lsh_ms"], marker="o",
            label="multi-table LSH, 20% query noise")
    ax.set_xscale("log"); ax.set_xlabel("entity inventory size")
    ax.set_ylabel("entry-snap latency (ms/query)")
    ax.set_title("3. Fast entrance (exact-scan fallback on low confidence)")
    ax.grid(alpha=0.3); ax.legend()

    plt.tight_layout()
    plt.savefig("final_fixes.png", dpi=150)
    print("\nSaved final_fixes.png")


if __name__ == "__main__":
    main()
