import argparse
import csv
import msvcrt
import os
from collections import Counter

from serial_common import Board, find_port, N_FEAT

LABELS = "0123456789"
HELP = """keys: 0-9 pick label | SPACE start, draw, SPACE stop | a/m auto/manual | u undo last | q quit"""

def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [row for row in csv.reader(f) if row]

def save_rows(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)

def status(rows, label, recording):
    counts = Counter(r[0] for r in rows)
    parts = " ".join("%s:%d" % (l, counts.get(l, 0)) for l in LABELS)
    state = "REC" if recording else "   "
    print("\r[%s] label=%s | %s   " % (state, label or "-", parts), end="", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--out", default="data/gestures.csv")
    args = ap.parse_args()

    rows = load_rows(args.out)
    print("Loaded %d existing samples from %s" % (len(rows), args.out))
    print(HELP)

    board = Board(find_port(args.port))
    board.send("m")

    label = None
    recording = False
    status(rows, label, recording)

    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in LABELS:
                label = ch
            elif ch == " ":
                if not label:
                    print("\nPick a label digit first.")
                else:
                    board.send("e" if recording else "b")
            elif ch in ("a", "m"):
                board.send(ch)
                recording = False
            elif ch == "u":
                if rows:
                    removed = rows.pop()
                    save_rows(args.out, rows)
                    print("\nRemoved last sample (label %s)." % removed[0])
            elif ch == "q":
                break
            status(rows, label, recording)

        msg = board.get(timeout=0.05)
        if msg is None:
            continue
        kind, payload = msg
        if kind == "S":
            recording = True
            print("\n  drawing...")
        elif kind == "F":
            recording = False
            if label:
                rows.append([label] + ["%.4f" % v for v in payload])
                save_rows(args.out, rows)
                print("\n  saved sample #%d for label %s" % (len(rows), label))
            else:
                print("\n  gesture received but no label selected, discarded")
        elif kind == "E":
            recording = False
            print("\n  discarded: %s (move more, or hold the board still less)" % payload)
        elif kind == "P":
            print("\n  board predicts %s (%.0f%%)" % (payload[0], payload[1] * 100))
        elif kind in ("M", "A"):
            print("\n  board mode: %s" % ("manual" if kind == "M" else "auto"))
        elif kind == "X":
            print("\nSerial link lost.")
            break
        status(rows, label, recording)

    board.close()
    print("\nSaved %d samples to %s" % (len(rows), args.out))

if __name__ == "__main__":
    main()
