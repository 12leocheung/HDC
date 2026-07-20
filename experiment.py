"""
experiment.py — Can structure buy back what gradients buy?

Task: two-hop inference over an associative memory.
  Facts:   bind(R1, person_i) -> animal_i      (hop 1: "person's pet")
           bind(R2, animal_i) -> sound_k       (hop 2: "animal's sound")
  Query:   given person_i, recover sound_k. No single fact contains the
           answer; the output of hop 1 must become the key of hop 2.

Conditions (cumulative architecture from this conversation):
  A flat_raw       one superposed memory; hop-1 output used RAW as hop-2 key
  B flat_quant     one superposed memory; hop-1 output snapped to the
                   nearest known item before hop 2 ("return to root")
  C tree           facts sharded into a 2-level bucket tree; greedy routing
  D tree_verify    + bind-back verification with backtracking beam search
  E tree_practice  + LVQ self-quizzing: route every stored key, punish
                   wrong routes, reward right ones — no gradients

Metric: full-chain accuracy vs N (total stored facts), D fixed at 2048.
"""

from __future__ import annotations

import numpy as np

D = 2048
N_SOUNDS = 50
LEAF_SIZE = 32          # target facts per leaf bucket
N_EVAL = 200            # queries sampled per condition
PRACTICE_EPOCHS = 10
VERIFY_MARGIN = 0.5     # accept if bind-back score > margin * expected signal
SEED = 7


def rand_vec(rng, n=1):
    v = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n, D))
    return v[0] if n == 1 else v


def cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b) / (na * nb) if na and nb else 0.0


def normalize_rows(M):
    M = M.astype(np.float32)
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return np.divide(M, n, out=np.zeros_like(M), where=n > 0)


def sims_to(normed_M, v):
    v = v.astype(np.float32)
    nv = np.linalg.norm(v)
    return (normed_M @ v) / nv if nv else np.zeros(normed_M.shape[0], dtype=np.float32)


class World:
    """Random entities + the two-hop fact set."""

    def __init__(self, n_pairs: int, rng):
        self.rng = rng
        self.R1, self.R2 = rand_vec(rng), rand_vec(rng)
        self.persons = rand_vec(rng, n_pairs)
        self.animals = rand_vec(rng, n_pairs)
        self.sounds = rand_vec(rng, N_SOUNDS)
        self.animal_sound = rng.integers(0, N_SOUNDS, size=n_pairs)
        self.animals_f = normalize_rows(self.animals)
        self.sounds_f = normalize_rows(self.sounds)

        # keys/values for every stored fact (2 * n_pairs facts total)
        k1 = (self.R1[None, :] * self.persons).astype(np.int8)
        k2 = (self.R2[None, :] * self.animals).astype(np.int8)
        self.keys = np.vstack([k1, k2])
        self.vals = np.vstack([self.animals, self.sounds[self.animal_sound]])
        self.n_pairs = n_pairs


# ----------------------------------------------------------------------
# Memories
# ----------------------------------------------------------------------
class FlatMemory:
    def __init__(self, world: World):
        self.M = np.sum(world.keys.astype(np.int64) * world.vals.astype(np.int64), axis=0)

    def unbind(self, key):
        return self.M * key.astype(np.int64)


class TreeMemory:
    """Two-level bucket tree. Facts randomly sharded into leaves; each node's
    address is the thresholded bundle of the keys stored beneath it."""

    def __init__(self, world: World, rng):
        n_facts = world.keys.shape[0]
        self.n_leaves = max(1, int(np.ceil(n_facts / LEAF_SIZE)))
        self.branch = max(2, int(np.ceil(np.sqrt(self.n_leaves))))
        self.n_leaves = self.branch ** 2  # full 2-level tree

        self.leaf_of = rng.integers(0, self.n_leaves, size=n_facts)
        self.leaf_M = np.zeros((self.n_leaves, D), dtype=np.int64)
        self.leaf_addr_acc = np.zeros((self.n_leaves, D), dtype=np.float64)
        self.leaf_facts: list[list[int]] = [[] for _ in range(self.n_leaves)]
        for f in range(n_facts):
            leaf = self.leaf_of[f]
            self.leaf_M[leaf] += world.keys[f].astype(np.int64) * world.vals[f].astype(np.int64)
            self.leaf_addr_acc[leaf] += world.keys[f]
            self.leaf_facts[leaf].append(f)

        # level-1 nodes: leaf i belongs to node i // branch
        self.node_addr_acc = np.zeros((self.branch, D), dtype=np.float64)
        for leaf in range(self.n_leaves):
            self.node_addr_acc[leaf // self.branch] += self.leaf_addr_acc[leaf]
        self.leaf_M_norm = np.linalg.norm(self.leaf_M.astype(np.float64), axis=1)
        self._refresh()

    def _refresh(self):
        self.node_addr = normalize_rows(np.sign(self.node_addr_acc))
        self.leaf_addr = normalize_rows(np.sign(self.leaf_addr_acc))

    def route(self, key, beam=1):
        """Return leaf ids, best-first, exploring `beam` branches per level."""
        node_sims = sims_to(self.node_addr, key)
        nodes = np.argsort(node_sims)[::-1][:beam]
        leaves = []
        for nd in nodes:
            child_ids = np.arange(nd * self.branch, (nd + 1) * self.branch)
            child_sims = sims_to(self.leaf_addr[child_ids], key)
            for c in np.argsort(child_sims)[::-1][:beam]:
                leaves.append(int(child_ids[c]))
        return leaves

    def unbind(self, leaf, key):
        return self.leaf_M[leaf] * key.astype(np.int64)

    def verify(self, leaf, key, val) -> bool:
        """Bind-back check: is bind(key, val) actually present in this leaf?"""
        trace = (key.astype(np.float32) * val.astype(np.float32))
        n_facts = max(1, len(self.leaf_facts[leaf]))
        denom = self.leaf_M_norm[leaf] * np.sqrt(D)
        score = float(self.leaf_M[leaf] @ trace) / denom if denom else 0.0
        expected = 1.0 / np.sqrt(n_facts)   # signal size if present
        return score > VERIFY_MARGIN * expected

    def practice(self, world: World, epochs: int, eta: float = 1.0):
        """LVQ self-quiz: every stored key must route to its own leaf.
        Wrong route -> push wrong addresses away, pull correct ones closer.
        Vectorized: one routing pass per epoch is two matmuls."""
        keys_f = world.keys.astype(np.float32)
        true_leaf = self.leaf_of
        for _ in range(epochs):
            got_node = np.argmax(keys_f @ self.node_addr.T, axis=1)
            # greedy leaf within the chosen node
            leaf_sims = keys_f @ self.leaf_addr.T          # (F, n_leaves)
            got_leaf = np.empty(len(keys_f), dtype=np.int64)
            for nd in range(self.branch):
                mask = got_node == nd
                if mask.any():
                    block = leaf_sims[mask][:, nd * self.branch:(nd + 1) * self.branch]
                    got_leaf[mask] = nd * self.branch + np.argmax(block, axis=1)
            wrong = np.nonzero(got_leaf != true_leaf)[0]
            if len(wrong) == 0:
                break
            for f in wrong:
                key = world.keys[f]
                tl = int(true_leaf[f])
                # PULL-ONLY: reinforce the correct path, never punish the
                # wrong one — punishment corrupts the wrong bucket's address
                # for its own members (stability-plasticity failure, observed
                # empirically in the +/- variant of this update).
                self.leaf_addr_acc[tl] += eta * key
                self.node_addr_acc[tl // self.branch] += eta * key
            self._refresh()


class LSHMemory:
    """Content-addressed bucketing: leaf index = sign bits of the key
    projected on fixed random hyperplanes. No stored addresses, so routing
    cannot degrade with N — the query COMPUTES its bucket. Exact keys
    (guaranteed by quantized restarts between hops) route perfectly."""

    def __init__(self, world: World, rng):
        n_facts = world.keys.shape[0]
        self.bits = max(1, int(np.ceil(np.log2(max(2, n_facts / LEAF_SIZE)))))
        self.H = rng.choice(np.array([-1, 1], dtype=np.int8),
                            size=(self.bits, D)).astype(np.float32)
        self.n_leaves = 2 ** self.bits
        self.leaf_M = np.zeros((self.n_leaves, D), dtype=np.int64)
        self.leaf_count = np.zeros(self.n_leaves, dtype=np.int64)
        for f in range(n_facts):
            leaf = self._hash(world.keys[f])
            self.leaf_M[leaf] += world.keys[f].astype(np.int64) * world.vals[f].astype(np.int64)
            self.leaf_count[leaf] += 1
        self.leaf_M_norm = np.linalg.norm(self.leaf_M.astype(np.float64), axis=1)

    def _hash(self, key) -> int:
        bits = (self.H @ key.astype(np.float32)) > 0
        return int(np.dot(bits, 1 << np.arange(self.bits)))

    def unbind(self, leaf, key):
        return self.leaf_M[leaf] * key.astype(np.int64)

    def verify(self, leaf, key, val) -> bool:
        trace = key.astype(np.float32) * val.astype(np.float32)
        denom = self.leaf_M_norm[leaf] * np.sqrt(D)
        score = float(self.leaf_M[leaf] @ trace) / denom if denom else 0.0
        expected = 1.0 / np.sqrt(max(1, self.leaf_count[leaf]))
        return score > VERIFY_MARGIN * expected


def lsh_hop(mem, key, items_f, items_raw):
    leaf = mem._hash(key)
    noisy = mem.unbind(leaf, key)
    for cand in cleanup(noisy, items_f, top=2):
        if mem.verify(leaf, key, items_raw[cand]):
            return cand
    return cleanup(noisy, items_f)[0]


def run_lsh(world, mem, queries):
    correct = 0
    for p in queries:
        k1 = (world.R1 * world.persons[p]).astype(np.int8)
        a = lsh_hop(mem, k1, world.animals_f, world.animals)
        k2 = (world.R2 * world.animals[a]).astype(np.int8)   # quantized restart
        s = lsh_hop(mem, k2, world.sounds_f, world.sounds)
        correct += (s == world.animal_sound[p])
    return correct / len(queries)


# ----------------------------------------------------------------------
# Retrieval procedures per condition
# ----------------------------------------------------------------------
def cleanup(noisy, items_f, top=1):
    sims = sims_to(items_f, noisy)
    return np.argsort(sims)[::-1][:top]


def run_flat(world, mem, quantize: bool, queries):
    correct = 0
    for p in queries:
        k1 = world.R1 * world.persons[p]
        noisy_animal = mem.unbind(k1)
        if quantize:
            a = cleanup(noisy_animal, world.animals_f)[0]
            hop2_carrier = world.animals[a].astype(np.int64)
        else:
            hop2_carrier = noisy_animal  # raw smudge feeds hop 2
        k2 = world.R2.astype(np.int64) * hop2_carrier
        noisy_sound = mem.unbind(np.sign(k2).astype(np.int8))
        s = cleanup(noisy_sound, world.sounds_f)[0]
        correct += (s == world.animal_sound[p])
    return correct / len(queries)


def tree_hop(world, tree, key, items_f, items_raw, use_verify: bool):
    """One hop through the tree. Returns best item id."""
    beam = 2 if use_verify else 1
    leaves = tree.route(key, beam=beam)
    if not use_verify:
        return cleanup(tree.unbind(leaves[0], key), items_f)[0]
    for leaf in leaves:
        noisy = tree.unbind(leaf, key)
        for cand in cleanup(noisy, items_f, top=2):
            if tree.verify(leaf, key, items_raw[cand]):
                return cand
    return cleanup(tree.unbind(leaves[0], key), items_f)[0]  # unverified fallback


def run_tree(world, tree, use_verify: bool, queries):
    correct = 0
    for p in queries:
        k1 = (world.R1 * world.persons[p]).astype(np.int8)
        a = tree_hop(world, tree, k1, world.animals_f, world.animals, use_verify)
        k2 = (world.R2 * world.animals[a]).astype(np.int8)   # quantized restart
        s = tree_hop(world, tree, k2, world.sounds_f, world.sounds, use_verify)
        correct += (s == world.animal_sound[p])
    return correct / len(queries)


# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    Ns = [100, 300, 1000, 3000, 10000]
    results = {c: [] for c in ["flat_raw", "flat_quant", "tree",
                               "tree_verify", "tree_practice", "content_tree"]}

    for N in Ns:
        n_pairs = N // 2
        world = World(n_pairs, rng)
        queries = rng.integers(0, n_pairs, size=min(N_EVAL, n_pairs))

        flat = FlatMemory(world)
        results["flat_raw"].append(run_flat(world, flat, False, queries))
        results["flat_quant"].append(run_flat(world, flat, True, queries))

        tree = TreeMemory(world, rng)
        results["tree"].append(run_tree(world, tree, False, queries))
        results["tree_verify"].append(run_tree(world, tree, True, queries))

        tree.practice(world, PRACTICE_EPOCHS)
        results["tree_practice"].append(run_tree(world, tree, True, queries))

        lsh = LSHMemory(world, rng)
        results["content_tree"].append(run_lsh(world, lsh, queries))

        print(f"N={N:6d}  " + "  ".join(
            f"{c}={results[c][-1]:.2f}" for c in results))

    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "flat_raw": "flat, raw chaining",
        "flat_quant": "flat + quantize between hops",
        "tree": "bucket tree",
        "tree_verify": "tree + verify/backtrack",
        "tree_practice": "tree + verify + practice (LVQ, pull-only)",
        "content_tree": "content-addressed buckets (LSH) + verify",
    }
    plt.figure(figsize=(8, 5))
    for c, accs in results.items():
        plt.plot(Ns, accs, marker="o", label=labels[c])
    plt.xscale("log")
    plt.xlabel("N — total stored facts (D fixed at 2048)")
    plt.ylabel("two-hop chain accuracy")
    plt.title("Can structure buy back what gradients buy?")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("two_hop_results.png", dpi=150)
    print("\nSaved two_hop_results.png")


if __name__ == "__main__":
    main()
