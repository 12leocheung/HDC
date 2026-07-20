"""Simple CLI chat utility to run experiments and view outputs.

Usage:
  python chat.py help            # show available commands
  python chat.py run3            # run experiment3.py
  python chat.py run4            # run experiment4.py
  python chat.py open final      # open final_fixes.png
  python chat.py open set        # open set_readout.png
  python chat.py gitlog          # show last commit
  python chat.py exit            # interactive REPL only

Without args this starts an interactive REPL.
"""

import os
import sys
import subprocess

BASE = os.path.dirname(__file__)
FINAL_IMG = os.path.join(BASE, "final_fixes.png")
SET_IMG = os.path.join(BASE, "set_readout.png")


def run_cmd(cmd, capture=False):
    try:
        if capture:
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, cwd=BASE)
            return out.decode(errors="ignore")
        else:
            subprocess.run(cmd, shell=True, cwd=BASE)
            return None
    except subprocess.CalledProcessError as e:
        return e.output.decode(errors="ignore") if e.output is not None else str(e)


def open_file(path):
    if not os.path.exists(path):
        print("file not found:", path)
        return
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", path])
    else:
        subprocess.run(["xdg-open", path])


def handle_command(cmd):
    if cmd in ("help", "h", "?"):
        print(__doc__)
        return
    if cmd == "run3":
        print("Running experiment3.py...")
        run_cmd("python experiment3.py")
        return
    if cmd == "run4":
        print("Running experiment4.py...")
        run_cmd("python experiment4.py")
        return
    if cmd == "open final":
        open_file(FINAL_IMG)
        return
    if cmd == "open set":
        open_file(SET_IMG)
        return
    if cmd == "gitlog":
        out = run_cmd("git --no-pager log -1 --stat", capture=True)
        print(out)
        return
    print("Unknown command. Type 'help'.")


def repl():
    print("chat> type 'help' for commands")
    try:
        while True:
            cmd = input("chat> ").strip()
            if cmd in ("exit", "quit"):
                break
            handle_command(cmd)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        handle_command(" ".join(sys.argv[1:]))
    else:
        repl()
