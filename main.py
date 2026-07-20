"""
main.py — train and chat with the HDC language model.

Usage:
    python main.py train --data dataset.txt --out model.npz
    python main.py chat  --model model.npz

Chat commands:
    /temp 0.5     set sampling temperature
    /topk 3       set top-k
    /quit         exit
"""

import argparse
import os
import sys

from hdc_llm import HDCLM


def load_dataset(path: str) -> list[tuple[str, str]]:
    """dataset.txt format: one exchange per line, USER<TAB>BOT."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" not in line:
                print(f"[warn] line {line_no} has no tab separator, skipping")
                continue
            user, bot = line.split("\t", 1)
            pairs.append((user.strip(), bot.strip()))
    return pairs


def cmd_train(args: argparse.Namespace) -> None:
    pairs = load_dataset(args.data)
    print(f"Loaded {len(pairs)} exchanges from {args.data}")

    model = HDCLM(dims=args.dims, seed=args.seed)
    sequences = [model.sequence_from_pair(u, b) for u, b in pairs]
    print("Training HDC memory heads...")
    model.train(sequences)
    model.save(args.out)
    print(f"Saved model to {args.out}")


def cmd_chat(args: argparse.Namespace) -> None:
    if not os.path.exists(args.model):
        sys.exit(f"No model at {args.model} — run: python main.py train")
    model = HDCLM.load(args.model)
    temperature, top_k = args.temperature, args.top_k
    print(f"HDC chat ready (vocab {len(model.inv_vocab)}, dims {model.dims}). "
          f"/temp N, /topk N, /quit\n")

    while True:
        try:
            user = input("you  > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break
        if not user:
            continue
        if user.startswith("/"):
            parts = user.split()
            if parts[0] == "/quit":
                break
            if parts[0] == "/temp" and len(parts) > 1:
                temperature = float(parts[1])
                print(f"temperature = {temperature}")
                continue
            if parts[0] == "/topk" and len(parts) > 1:
                top_k = int(parts[1])
                print(f"top_k = {top_k}")
                continue
            print("commands: /temp N, /topk N, /quit")
            continue

        reply = model.generate(user, temperature=temperature, top_k=top_k)
        print(f"bot  > {reply}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Chattable HDC language model")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", default="dataset.txt")
    t.add_argument("--out", default="model.npz")
    t.add_argument("--dims", type=int, default=20_000)
    t.add_argument("--seed", type=int, default=1337)
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("chat")
    c.add_argument("--model", default="model.npz")
    c.add_argument("--temperature", type=float, default=0.7)
    c.add_argument("--top-k", type=int, default=4)
    c.set_defaults(func=cmd_chat)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()