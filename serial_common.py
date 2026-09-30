import queue
import sys
import threading
import time

import serial
from serial.tools import list_ports

N_FEAT = 240

def find_port(explicit=None):
    if explicit:
        return explicit
    ports = [p.device for p in list_ports.comports()]
    if len(ports) == 1:
        return ports[0]
    if not ports:
        sys.exit("No serial ports found. Plug in the TinyS3 and try again.")
    sys.exit("Several ports found (%s). Pass --port COMx." % ", ".join(ports))

class Board:
    def __init__(self, port, baud=115200):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.lines = queue.Queue()
        self._stop = False
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()
        time.sleep(0.3)
        self.ser.reset_input_buffer()

    def _reader(self):
        buf = b""
        while not self._stop:
            try:
                chunk = self.ser.read(4096)
            except serial.SerialException:
                self.lines.put(("X", "serial error"))
                return
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("ascii", errors="replace").strip()
                if text:
                    self.lines.put(parse_line(text))

    def send(self, cmd):
        self.ser.write(cmd.encode("ascii"))
        self.ser.flush()

    def get(self, timeout=None):
        try:
            return self.lines.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self._stop = True
        try:
            self.ser.close()
        except Exception:
            pass

def parse_line(text):
    kind = text[0]
    rest = text[2:] if len(text) > 1 and text[1] == "," else ""
    if kind == "F":
        try:
            vals = [float(v) for v in rest.split(",")]
        except ValueError:
            return ("?", text)
        if len(vals) != N_FEAT:
            return ("?", text)
        return ("F", vals)
    if kind == "P":
        parts = rest.split(",")
        try:
            return ("P", (parts[0], float(parts[1])))
        except (IndexError, ValueError):
            return ("?", text)
    if kind in ("R", "M", "A", "S", "E", "W", "I"):
        return (kind, rest)
    return ("?", text)
