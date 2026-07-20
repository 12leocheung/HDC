"""A tiny trainable chatbot using hyperdimensional text encodings.

Commands:
  python chatbot.py train add "QUESTION" "ANSWER"   # append example
  python chatbot.py train build                         # build index from qa.json
  python chatbot.py chat "your prompt"                 # single-shot chat
  python chatbot.py chat                                # interactive REPL
  python chatbot.py show                                # show DB stats

Implementation notes:
- Token vectors are created on first use and persisted in `token_vectors.json`.
- Questions are encoded as summed token hypervectors (float) and normalized.
- Nearest neighbor is computed via cosine similarity over stored question vectors.
- Small, local, no external models required.
"""

import json
import os
import sys
import numpy as np
from typing import List

BASE = os.path.dirname(__file__)
QA_PATH = os.path.join(BASE, "qa.json")
TOK_PATH = os.path.join(BASE, "token_vectors.json")
KEYS_NPY = os.path.join(BASE, "chat_keys.npy")
KEYS_F_NPY = os.path.join(BASE, "chat_keys_f.npy")
RESP_PATH = os.path.join(BASE, "chat_responses.json")

D = 2048
SEED = 12345
MIN_RESPONSE_SIM = 0.55
rng = np.random.default_rng(SEED)


def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: broken JSON file {path} ignored.")
            return default
    return default


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def rand_vec():
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(D,)).tolist()


def tokenize(s: str) -> List[str]:
    return [t for t in s.lower().split() if t]


def ensure_token_vectors(tokens: List[str]):
    toks = load_json(TOK_PATH, {})
    changed = False
    for t in tokens:
        if t not in toks:
            toks[t] = rand_vec()
            changed = True
    if changed:
        save_json(TOK_PATH, toks)
    return toks


def encode_text(s: str) -> np.ndarray:
    tokens = tokenize(s)
    if not tokens:
        return np.zeros(D, dtype=np.float32)
    toks = ensure_token_vectors(tokens)
    vec = np.zeros(D, dtype=np.float32)
    for t in tokens:
        v = np.array(toks[t], dtype=np.float32)
        vec += v
    # normalize to unit-length float32
    n = np.linalg.norm(vec)
    return (vec / n).astype(np.float32) if n > 0 else vec.astype(np.float32)


def build_index():
    qa = load_json(QA_PATH, [])
    responses = [pair[1] for pair in qa]
    keys = []
    for q, _ in qa:
        keys.append(encode_text(q))
    if keys:
        keys = np.vstack(keys).astype(np.float32)
        # save normalized rows
        norms = np.linalg.norm(keys, axis=1, keepdims=True)
        keys_f = keys / np.where(norms > 0, norms, 1.0)
        np.save(KEYS_NPY, keys)
        np.save(KEYS_F_NPY, keys_f)
    else:
        # empty
        np.save(KEYS_NPY, np.zeros((0, D), dtype=np.float32))
        np.save(KEYS_F_NPY, np.zeros((0, D), dtype=np.float32))
    save_json(RESP_PATH, responses)
    print(f"Built index: {len(responses)} examples")


def nearest_response(prompt: str):
    if not os.path.exists(KEYS_F_NPY) or not os.path.exists(RESP_PATH):
        print("Index not found — run: python chatbot.py train build")
        return None
    qv = encode_text(prompt)
    keys_f = np.load(KEYS_F_NPY)
    if keys_f.size == 0:
        return None
    sims = keys_f @ qv
    idx = int(np.argmax(sims))
    score = float(sims[idx])
    if score < MIN_RESPONSE_SIM:
        return None
    resp = load_json(RESP_PATH, [])
    if idx < 0 or idx >= len(resp):
        return None
    return resp[idx], score


def cmd_train_add(args: List[str]):
    if len(args) < 2:
        print("Usage: python chatbot.py train add \"QUESTION\" \"ANSWER\"")
        return
    q, a = args[0], args[1]
    qa = load_json(QA_PATH, [])
    qa.append([q, a])
    save_json(QA_PATH, qa)
    print("Added example. Now run: python chatbot.py train build")


def cmd_train_build(_args):
    build_index()


def cmd_chat(args: List[str]):
    if args:
        prompt = " ".join(args)
        out = nearest_response(prompt)
        if out is None:
            print("I don't know yet — add examples with: python chatbot.py train add \"Q\" \"A\"")
        else:
            resp, score = out
            print(resp)
            print(f"(sim={score:.3f})")
        return
    # interactive REPL
    print("Chat REPL — type your prompt, Ctrl-D to exit")
    try:
        while True:
            p = input("You: ")
            if not p.strip():
                continue
            out = nearest_response(p)
            if out is None:
                print("Bot: I don't know yet. Add an example with: python chatbot.py train add \"Q\" \"A\"")
            else:
                resp, score = out
                print("Bot:", resp)
    except (KeyboardInterrupt, EOFError):
        print()


def cmd_show(_args):
    qa = load_json(QA_PATH, [])
    resp = load_json(RESP_PATH, [])
    toks = load_json(TOK_PATH, {})
    print(f"examples={len(qa)} responses_indexed={len(resp)} tokens={len(toks)}")


def main():
    if len(sys.argv) <= 1:
        cmd_chat([])
        return
    cmd = sys.argv[1]
    if cmd == "train":
        if len(sys.argv) >= 3 and sys.argv[2] == "add":
            # train add "Q" "A"
            cmd_train_add(sys.argv[3:])
            return
        if len(sys.argv) >= 3 and sys.argv[2] == "build":
            cmd_train_build(sys.argv[3:])
            return
    if cmd == "chat":
        cmd_chat(sys.argv[2:])
        return
    if cmd == "show":
        cmd_show(sys.argv[2:])
        return
    print(__doc__)


if __name__ == "__main__":
    main()
