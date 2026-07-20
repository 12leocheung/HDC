"""
experiment2.py — Wall #1: does content-addressed routing survive NOISY queries?

Setup: N = 10,000 facts (where flat memory scores ~0.01 and content
addressing scored 1.00 with clean keys). Now corrupt the hop-1 query by
flipping a fraction p of the person-vector's bits before composing the key
— a stand-in for fuzzy natural-language encoding. Hop 2 always starts from
a quantized (exact) key if hop 1 succeeded.

Conditions:
  A flat_quant     flat memory + quantize (reference floor)
  B lsh_single     content-addressed, naive single probe — one flipped hash
                   bit = wrong bucket = silent failure
  C lsh_multiprobe probe buckets in order of hash-bit confidence (flip the
                   least-confident bits first), verify in each, accept the
                   first verified answer
  D quant_entry    snap the noisy query to the nearest KNOWN entity before
                   routing ("quantize at entry"), then content-address with
                   the exact key. Noise is scrubbed at the front door.

Metric: full two-hop chain accuracy vs bit-flip fraction p.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from experiment import (
    D, LEAF_SIZE, VERIFY_MARGIN, N_SOUNDS,
    World, FlatMemory, LSHMemory, cleanup, run_flat,
)

N_FACTS = 10_000
N_EVAL = 200
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
MAX_PROBES = 8
SEED = 11


def flip_bits(vec, p, rng):
    mask = np.where(rng.random(D) < p, -1, 1).astype(np.int8)
    return (vec * mask).astype(np.int8)


def probe_order(mem: LSHMemory, key, max_probes: int):
    """Multi-probe LSH: yield bucket ids starting from the base bucket,
    then flipping the least-confident hash bits first."""
    z = mem.H @ key.astype(np.float32)
    base_bits = (z > 0).astype(np.int64)
    conf_order = np.argsort(np.abs(z))          # least confident first
    weights = 1 << np.arange(mem.bits)

    yielded = 0
    for n_flip in range(0, mem.bits + 1):
        for combo in combinations(conf_order[:4], n_flip):
            bits = base_bits.copy()
            for b in combo:
                bits[b] ^= 1
            yield int(np.dot(bits, weights))
            yielded += 1
            if yielded >= max_probes:
                return


def lsh_hop_noisy(mem, key, items_f, items_raw, max_probes: int):
    """One hop with multi-probe + verification."""
    fallback = None
    for leaf in probe_order(mem, key, max_probes):
        if mem.leaf_count[leaf] == 0:
            continue
        noisy = mem.unbind(leaf, key)
        cands = cleanup(noisy, items_f, top=2)
        if fallback is None:
            fallback = cands[0]
        for cand in cands:
            if mem.verify(leaf, key, items_raw[cand]):
                return cand
    return fallback if fallback is not None else 0


def run_condition(world, mem, queries, noise_p, rng, mode: str):
    correct = 0
    for p_idx in queries:
        noisy_person = flip_bits(world.persons[p_idx], noise_p, rng)

        if mode == "quant_entry":
            # Snap to the nearest KNOWN person first — noise never reaches
            # the router. This is the return-to-root idea applied at input.
            snapped = cleanup(noisy_person, world.persons_f)[0]
            k1 = (world.R1 * world.persons[snapped]).astype(np.int8)
            probes = 1
        else:
            k1 = (world.R1 * noisy_person).astype(np.int8)
            probes = MAX_PROBES if mode == "multiprobe" else 1

        a = lsh_hop_noisy(mem, k1, world.animals_f, world.animals, probes)
        # Quantized restart: hop 2's key is exact regardless of hop-1 noise
        k2 = (world.R2 * world.animals[a]).astype(np.int8)
        s = lsh_hop_noisy(mem, k2, world.sounds_f, world.sounds, 1)
        correct += (s == world.animal_sound[p_idx])
    return correct / len(queries)


def run_flat_noisy(world, flat, queries, noise_p, rng):
    correct = 0
    for p_idx in queries:
        noisy_person = flip_bits(world.persons[p_idx], noise_p, rng)
        k1 = (world.R1 * noisy_person).astype(np.int8)
        a = cleanup(flat.unbind(k1), world.animals_f)[0]
        k2 = (world.R2 * world.animals[a]).astype(np.int8)
        s = cleanup(flat.unbind(k2), world.sounds_f)[0]
        correct += (s == world.animal_sound[p_idx])
    return correct / len(queries)


def main():
    rng = np.random.default_rng(SEED)
    world = World(N_FACTS // 2, rng)
    world.persons_f = world.persons.astype(np.float32)
    world.persons_f /= np.linalg.norm(world.persons_f, axis=1, keepdims=True)

    flat = FlatMemory(world)
    lsh = LSHMemory(world, rng)
    queries = rng.integers(0, world.n_pairs, size=N_EVAL)

    results = {c: [] for c in ["flat_quant", "lsh_single", "lsh_multiprobe", "quant_entry"]}
    for p in NOISE_LEVELS:
        results["flat_quant"].append(run_flat_noisy(world, flat, queries, p, rng))
        results["lsh_single"].append(run_condition(world, lsh, queries, p, rng, "single"))
        results["lsh_multiprobe"].append(run_condition(world, lsh, queries, p, rng, "multiprobe"))
        results["quant_entry"].append(run_condition(world, lsh, queries, p, rng, "quant_entry"))
        print(f"noise={p:.2f}  " + "  ".join(f"{c}={results[c][-1]:.2f}" for c in results))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "flat_quant": "flat memory (reference)",
        "lsh_single": "content-addressed, single probe",
        "lsh_multiprobe": f"content-addressed, multi-probe ({MAX_PROBES})",
        "quant_entry": "quantize-at-entry + content-addressed",
    }
    plt.figure(figsize=(8, 5))
    for c, accs in results.items():
        plt.plot([p * 100 for p in NOISE_LEVELS], accs, marker="o", label=labels[c])
    plt.xlabel("query noise — % of bits flipped in the hop-1 entity")
    plt.ylabel("two-hop chain accuracy")
    plt.title(f"Wall #1: noisy queries at N = {N_FACTS:,} facts")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("noise_results.png", dpi=150)
    print("\nSaved noise_results.png")


if __name__ == "__main__":
    main()
