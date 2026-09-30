# Air-writing digit recognition

https://github.com/user-attachments/assets/d82c5ac9-9e06-4d22-91cd-d6b6587e0e07

Inspired by the disney wand id's :)

Uses a bare-metal neural network running on an ESP32-S3 to recognize digits drawn in the air while holding an imu.

**Stack:** UM TinyS3 (ESP32-S3) + LSM9DS1 accel/gyro over I2C. Firmware in Arduino C++.
Training in Python/numpy. Model is a 240-64-32-10 MLP exported as C arrays into `model.h`.

## Wiring

LSM9DS1 VIN→3V3, GND→GND, SDA→IO8, SCL→IO9

## Setup

```
pip install pyserial numpy
```

Arduino IDE, board: UM TinyS3.

## Run

1. Flash `tinys3_adxl335.ino`. Serial Monitor shows `R`. Close the monitor.
2. `python collect.py --port COMx` press a digit key, SPACE, draw, SPACE. ~20 per digit.
3. `python train.py` prints accuracy, writes `model.h`.
4. Reflash. `python demo.py --port COMx` SPACE, draw, SPACE.

## Files

- `tinys3_adxl335.ino` sampling, gesture capture, feature extraction, on-device MLP
- `model.h` trained weights as C arrays (`MODEL_READY 1`; set to `0` to build without a model)
- `collect.py` label and save gestures to `data/gestures.csv`
- `train.py` train and export
- `demo.py` live window
- `serial_common.py` serial link and line parser

## Serial protocol

Commands: `b` begin, `e` end, `a` auto mode, `m` manual mode (default), `d` gyro debug, `?` status

Output: `R` ready, `S` recording started, `F,<240 floats>` features, `P,<class>,<conf>` prediction, `E,<reason>` discarded, `I,...` info

## How it works

Sensor at 119 Hz. Gesture is trimmed to the moving part (gyro > 40 deg/s), resampled to 40 points x 6 axes, mean-removed, max-abs scaled per sensor. Manual mode records between `b` and `e` (max 1 s). Auto mode triggers on motion above 80 deg/s and stops after 300 ms still.

## Tuning

Top of the sketch: `START_THRESH`, `STOP_THRESH`, `STOP_MS`, `MIN_SAMPLES`, `MAX_SAMPLES`
