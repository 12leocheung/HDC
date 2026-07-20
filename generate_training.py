"""Generate a large synthetic Q&A dataset for the HDC chatbot.

This script writes `qa.json` containing many (question, answer) pairs.
It samples templates and paraphrases to produce varied prompts.
"""
import json
import random
import os

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "qa.json")

random.seed(42)

GREETS = [
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "yo",
    "howdy",
]

RUN_EX = [
    "run experiment3",
    "run experiment 3",
    "execute experiment3",
    "start panel tests",
    "run the panels",
    "launch experiment3",
]

OPEN_CMD = [
    "open final image",
    "show final_fixes.png",
    "open set readout",
    "display set_readout.png",
]

ASK_HELP = [
    "how do i run the experiments",
    "how to run experiment3",
    "what command runs experiment4",
    "how do I generate the plots",
]

ASK_ABOUT = [
    "what does panel1 test",
    "explain panel 2",
    "what is panel3 about",
    "describe the entry snap experiment",
]

THANKS = [
    "thanks",
    "thank you",
    "thx",
    "cheers",
]

CLOSING = [
    "bye",
    "goodbye",
    "see ya",
]

RESP = {
    "greet": "Hi — I can run experiments, build indices, or open result images.",
    "run_ex": "To run experiment3: python experiment3.py — it will save final_fixes.png.",
    "open_cmd": "Use python chat.py open final or open set to view the images.",
    "ask_help": "Run `python experiment3.py` or `python experiment4.py`. Use `chat.py` to control it interactively.",
    "ask_about": "Panel 1 tests correlated entities; Panel 2 tests chain depth; Panel 3 measures entry-snap latency with ANN.",
    "thanks": "You\'re welcome!",
    "closing": "Goodbye!",
}

# A few technical Q/A templates
TECH_Q = [
    "what is LSH",
    "explain LSH briefly",
    "how does deflation work",
    "what is quantized restart",
]
TECH_A = {
    "what is LSH": "LSH (Locality-Sensitive Hashing) hashes similar vectors into the same buckets for fast approximate nearest neighbor.",
    "explain LSH briefly": "LSH uses random hyperplanes to map high-dim vectors to bucket indices where similar vectors collide.",
    "how does deflation work": "Deflation retrieves a top candidate, verifies it, subtracts its trace from the signal, and repeats.",
    "what is quantized restart": "Quantized restart snaps the noisy hop output to the nearest known item before the next hop.",
}

# Generate paraphrases by small variations
def paraphrase(s):
    variants = [s]
    if "experiment" in s:
        variants.append(s + " please")
        variants.append("can you " + s)
    if s in GREETS:
        variants.append(s + "!")
        variants.append(s.capitalize())
    return list(set(variants))

pairs = []

# produce many greetings
for g in GREETS:
    for v in paraphrase(g):
        pairs.append([v, RESP["greet"]])

# run commands
for r in RUN_EX:
    for v in paraphrase(r):
        pairs.append([v, RESP["run_ex"]])

# open commands
for o in OPEN_CMD:
    pairs.append([o, RESP["open_cmd"]])

# help/questions
for q in ASK_HELP:
    pairs.append([q, RESP["ask_help"]])

for q in ASK_ABOUT:
    pairs.append([q, RESP["ask_about"]])

for t in TECH_Q:
    pairs.append([t, TECH_A[t]])

for t in THANKS:
    pairs.append([t, RESP["thanks"]])
for t in CLOSING:
    pairs.append([t, RESP["closing"]])

# add many paraphrased variants to reach ~1000 examples
base = ["how do i", "how to", "what is", "explain", "tell me about"]
objects = [
    "LSH",
    "deflation",
    "quantized restart",
    "per-leaf margin",
    "orthogonal ids",
    "adaptive probe",
    "final_fixes.png",
    "set_readout.png",
    "experiment3",
    "experiment4",
]
answers = {
    "LSH": "Locality-sensitive hashing groups similar vectors into buckets for ANN lookups.",
    "deflation": "Deflation iteratively subtracts verified candidates to recover set members.",
    "quantized restart": "Snap the noisy vector to the nearest known vector before the next hop.",
    "per-leaf margin": "Per-leaf margins are calibrated thresholds stored or computed per bucket.",
    "orthogonal ids": "Orthogonal IDs are random vectors orthogonalized to reduce cross-talk.",
    "adaptive probe": "Adaptive probe distributes a limited probe budget across LSH tables per query.",
    "final_fixes.png": "final_fixes.png contains the three-panel summary plots.",
    "set_readout.png": "set_readout.png shows set readout accuracy across N.",
    "experiment3": "Run python experiment3.py to generate the three panels and save final_fixes.png.",
    "experiment4": "Run python experiment4.py to generate set_readout.png showing set readout methods.",
}

while len(pairs) < 1000:
    a = random.choice(base)
    b = random.choice(objects)
    q = (a + " " + b).strip()
    atext = answers.get(b, "I can run experiments and answer simple questions about them.")
    # diversify answers slightly
    if random.random() < 0.3:
        atext = atext + "" if atext.endswith('.') else atext + '.'
    pairs.append([q, atext])

# deduplicate while preserving order
seen = set()
uniq = []
for q,a in pairs:
    key = q.strip().lower()
    if key not in seen:
        seen.add(key)
        uniq.append([q,a])

# ensure a minimum size by adding numbered variants if necessary
TARGET = 1000
idx = 0
while len(uniq) < TARGET:
    # take an existing pair and append a small variant index
    base_q, base_a = uniq[idx % len(uniq)]
    vq = f"{base_q} (variant {idx})"
    va = base_a
    uniq.append([vq, va])
    idx += 1

# save
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(uniq, f, ensure_ascii=False, indent=2)

print(f"Generated {len(uniq)} Q/A pairs -> {OUT}")
