import argparse
import os
import tkinter as tk

import numpy as np

from serial_common import Board, find_port

def load_pc_model(path="model.npz"):
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}

def pc_predict(m, x):
    x = np.asarray(x, dtype=np.float32)
    h1 = np.maximum(m["W1"] @ x + m["B1"], 0)
    h2 = np.maximum(m["W2"] @ h1 + m["B2"], 0)
    l = m["W3"] @ h2 + m["B3"]
    e = np.exp(l - l.max())
    p = e / e.sum()
    i = int(p.argmax())
    return str(m["names"][i]), float(p[i])

class App:
    def __init__(self, root, board, pc_model):
        self.root = root
        self.board = board
        self.pc_model = pc_model
        self.recording = False
        self.mode = "manual"

        root.title("Air-writing demo")
        self.digit = tk.Label(root, text="?", font=("TkDefaultFont", 96))
        self.digit.pack(padx=40, pady=20)
        self.info = tk.Label(root, text="")
        self.info.pack()
        self.mode_lbl = tk.Label(root, text="mode: manual")
        self.mode_lbl.pack()
        tk.Label(root, text="space: start/stop   a/m: auto/manual   q: quit").pack(pady=(10, 10))

        root.bind("<space>", self.toggle)
        root.bind("a", lambda e: board.send("a"))
        root.bind("m", lambda e: board.send("m"))
        root.bind("q", lambda e: root.destroy())
        root.bind("<Escape>", lambda e: root.destroy())
        board.send("m")
        root.after(30, self.poll)

    def set_mode(self, mode):
        self.mode = mode
        self.mode_lbl.config(text="mode: %s" % mode)

    def toggle(self, _evt=None):
        if self.mode != "manual":
            return
        self.board.send("e" if self.recording else "b")

    def show(self, digit, conf, source):
        self.digit.config(text=digit)
        self.info.config(text="%.0f%% (%s)" % (conf * 100, source))

    def poll(self):
        while True:
            msg = self.board.get(timeout=0)
            if msg is None:
                break
            kind, payload = msg
            if kind == "S":
                self.recording = True
                self.digit.config(text="...")
                self.info.config(text="drawing")
            elif kind == "F":
                self.recording = False
                if self.pc_model is not None:
                    d, c = pc_predict(self.pc_model, payload)
                    print("PC predicts %s (%.0f%%)" % (d, c * 100))
                    self.show(d, c, "laptop")
                else:
                    self.info.config(text="gesture received, no model yet")
            elif kind == "P":
                self.recording = False
                d, c = payload
                print("Board predicts %s (%.0f%%)" % (d, c * 100))
                self.show(d, c, "on-device")
            elif kind == "E":
                self.recording = False
                self.digit.config(text="?")
                self.info.config(text="discarded: %s" % payload)
            elif kind == "M":
                self.set_mode("manual")
            elif kind == "A":
                self.set_mode("auto")
            elif kind == "X":
                self.info.config(text="serial link lost")
        self.root.after(30, self.poll)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    args = ap.parse_args()
    board = Board(find_port(args.port))
    pc_model = load_pc_model()
    print("PC model: %s" % ("loaded model.npz" if pc_model is not None else "none"))
    root = tk.Tk()
    App(root, board, pc_model)
    try:
        root.mainloop()
    finally:
        board.close()

if __name__ == "__main__":
    main()
