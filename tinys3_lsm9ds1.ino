#include <Wire.h>
#include <math.h>
#include "model.h"

#define SAMPLE_RATE_HZ 119
#define N_PTS          40
#define N_AXES         6
#define N_FEAT         (N_PTS * N_AXES)
#define MAX_SAMPLES    120
#define MIN_SAMPLES    24
#define PREROLL        10
#define MARGIN         10
#define START_THRESH   80.0f
#define STOP_THRESH    40.0f
#define STOP_MS        300
#define STOP_SAMPLES   ((STOP_MS * SAMPLE_RATE_HZ) / 1000)

const int PIN_SDA = 8;
const int PIN_SCL = 9;

uint8_t agAddr = 0;

const uint8_t WHO_AM_I_AG  = 0x0F;
const uint8_t CTRL_REG1_G  = 0x10;
const uint8_t STATUS_REG   = 0x17;
const uint8_t OUT_X_L_G    = 0x18;
const uint8_t CTRL_REG6_XL = 0x20;
const uint8_t CTRL_REG8    = 0x22;
const uint8_t OUT_X_L_XL   = 0x28;

const float ACCEL_G_PER_LSB  = 0.000061;
const float GYRO_DPS_PER_LSB = 0.00875;

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(agAddr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

uint8_t readReg(uint8_t addr, uint8_t reg) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(addr, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0;
}

void readXYZ(uint8_t startReg, int16_t out[3]) {
  Wire.beginTransmission(agAddr);
  Wire.write(startReg);
  Wire.endTransmission(false);
  Wire.requestFrom(agAddr, (uint8_t)6);
  uint8_t b[6] = {0};
  for (int i = 0; i < 6 && Wire.available(); i++) b[i] = Wire.read();
  for (int i = 0; i < 3; i++) out[i] = (int16_t)(b[2 * i] | (b[2 * i + 1] << 8));
}

enum Mode  { MODE_MANUAL, MODE_AUTO };
enum State { ST_IDLE, ST_RECORDING };

Mode  mode  = MODE_MANUAL;
State state = ST_IDLE;
bool  debugW = false;

float buf[MAX_SAMPLES][N_AXES];
int   nBuf = 0;

float pre[PREROLL][N_AXES];
int   preHead = 0, preCount = 0;

int   startCount = 0;
int   stillCount = 0;
uint32_t dbgCounter = 0;

float feat[N_FEAT];

static inline float gyroMag(const float s[N_AXES]) {
  return sqrtf(s[3] * s[3] + s[4] * s[4] + s[5] * s[5]);
}

void startRecording() {
  nBuf = 0;

  for (int i = 0; i < preCount; i++) {
    int idx = (preHead - preCount + i + PREROLL) % PREROLL;
    for (int a = 0; a < N_AXES; a++) buf[nBuf][a] = pre[idx][a];
    nBuf++;
  }
  stillCount = 0;
  state = ST_RECORDING;
  Serial.println("S");
}

bool trimGesture(int &start, int &end) {
  int first = -1, last = -1;
  for (int i = 0; i < nBuf; i++) {
    if (gyroMag(buf[i]) > STOP_THRESH) {
      if (first < 0) first = i;
      last = i;
    }
  }
  if (first < 0) return false;
  start = first - MARGIN; if (start < 0) start = 0;
  end   = last + MARGIN;  if (end > nBuf - 1) end = nBuf - 1;
  return (end - start + 1) >= MIN_SAMPLES;
}

void extractFeatures(int start, int end) {
  int n = end - start + 1;
  for (int t = 0; t < N_PTS; t++) {
    float pos = (n > 1) ? (float)t * (n - 1) / (N_PTS - 1) : 0.0f;
    int i0 = (int)pos;
    int i1 = (i0 + 1 < n) ? i0 + 1 : i0;
    float frac = pos - i0;
    for (int a = 0; a < N_AXES; a++) {
      feat[t * N_AXES + a] = buf[start + i0][a] * (1.0f - frac) + buf[start + i1][a] * frac;
    }
  }
  float mean[N_AXES] = {0};
  for (int t = 0; t < N_PTS; t++)
    for (int a = 0; a < N_AXES; a++) mean[a] += feat[t * N_AXES + a];
  for (int a = 0; a < N_AXES; a++) mean[a] /= N_PTS;

  float maxAcc = 1e-6f, maxGyr = 1e-6f;
  for (int t = 0; t < N_PTS; t++) {
    for (int a = 0; a < N_AXES; a++) {
      float v = feat[t * N_AXES + a] - mean[a];
      feat[t * N_AXES + a] = v;
      float av = fabsf(v);
      if (a < 3) { if (av > maxAcc) maxAcc = av; }
      else       { if (av > maxGyr) maxGyr = av; }
    }
  }
  for (int t = 0; t < N_PTS; t++) {
    for (int a = 0; a < 3; a++) feat[t * N_AXES + a] /= maxAcc;
    for (int a = 3; a < 6; a++) feat[t * N_AXES + a] /= maxGyr;
  }
}

#if MODEL_READY
static float h1[H1], h2[H2], probs[N_CLASSES];

int mlpPredict(const float *x, float *conf) {
  for (int j = 0; j < H1; j++) {
    float s = MODEL_B1[j];
    const float *w = MODEL_W1[j];
    for (int i = 0; i < N_FEAT; i++) s += w[i] * x[i];
    h1[j] = s > 0 ? s : 0;
  }
  for (int j = 0; j < H2; j++) {
    float s = MODEL_B2[j];
    const float *w = MODEL_W2[j];
    for (int i = 0; i < H1; i++) s += w[i] * h1[i];
    h2[j] = s > 0 ? s : 0;
  }
  float maxL = -1e30f;
  for (int j = 0; j < N_CLASSES; j++) {
    float s = MODEL_B3[j];
    const float *w = MODEL_W3[j];
    for (int i = 0; i < H2; i++) s += w[i] * h2[i];
    probs[j] = s;
    if (s > maxL) maxL = s;
  }
  float sum = 0;
  for (int j = 0; j < N_CLASSES; j++) { probs[j] = expf(probs[j] - maxL); sum += probs[j]; }
  int best = 0;
  for (int j = 0; j < N_CLASSES; j++) {
    probs[j] /= sum;
    if (probs[j] > probs[best]) best = j;
  }
  *conf = probs[best];
  return best;
}
#endif

void finishRecording() {
  state = ST_IDLE;
  startCount = 0;
  stillCount = 0;
  preCount = 0;

  int start, end;
  if (!trimGesture(start, end)) {
    Serial.println("E,short");
    return;
  }
  extractFeatures(start, end);

  Serial.print("F");
  for (int i = 0; i < N_FEAT; i++) {
    Serial.print(',');
    Serial.print(feat[i], 4);
  }
  Serial.println();

#if MODEL_READY
  float conf;
  int cls = mlpPredict(feat, &conf);
  Serial.printf("P,%s,%.3f\n", CLASS_NAMES[cls], conf);
#endif
}

void handleCommand(char c) {
  switch (c) {
    case 'b':
      if (mode == MODE_MANUAL && state == ST_IDLE) startRecording();
      break;
    case 'e':
      if (mode == MODE_MANUAL && state == ST_RECORDING) finishRecording();
      break;
    case 'a':
      mode = MODE_AUTO; state = ST_IDLE; startCount = 0;
      Serial.println("A");
      break;
    case 'm':
      mode = MODE_MANUAL; state = ST_IDLE;
      Serial.println("M");
      break;
    case 'd':
      debugW = !debugW;
      break;
    case '?':
      Serial.printf("I,mode=%s,state=%s,model=%d\n",
                    mode == MODE_AUTO ? "auto" : "manual",
                    state == ST_RECORDING ? "recording" : "idle",
                    (int)MODEL_READY);
      break;
    default:
      break;
  }
}

void i2cBusRecover() {
  pinMode(PIN_SDA, INPUT_PULLUP);
  pinMode(PIN_SCL, OUTPUT_OPEN_DRAIN);
  digitalWrite(PIN_SCL, HIGH);
  delayMicroseconds(10);
  for (int i = 0; i < 9 && digitalRead(PIN_SDA) == LOW; i++) {
    digitalWrite(PIN_SCL, LOW);  delayMicroseconds(10);
    digitalWrite(PIN_SCL, HIGH); delayMicroseconds(10);
  }
  pinMode(PIN_SDA, OUTPUT_OPEN_DRAIN);
  digitalWrite(PIN_SDA, LOW);  delayMicroseconds(10);
  digitalWrite(PIN_SDA, HIGH); delayMicroseconds(10);
}

void i2cScan() {
  Serial.printf("I,scan sda=%d scl=%d (both should read 1 when idle)\n",
                digitalRead(PIN_SDA), digitalRead(PIN_SCL));
  int found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    uint8_t err = Wire.endTransmission();
    if (err == 0) { Serial.printf("I,found device at 0x%02X\n", addr); found++; }
    else if (err == 5) { Serial.printf("I,timeout at 0x%02X (bus stuck?)\n", addr); break; }
  }
  if (!found) Serial.println("I,no I2C devices respond");
}

bool probeSensor() {
  if (readReg(0x6B, WHO_AM_I_AG) == 0x68) { agAddr = 0x6B; return true; }
  if (readReg(0x6A, WHO_AM_I_AG) == 0x68) { agAddr = 0x6A; return true; }
  return false;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  i2cBusRecover();
  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(400000);
  Wire.setTimeOut(50);

  int attempt = 0;
  while (!probeSensor()) {
    if (attempt % 5 == 0) {
      Serial.println("E,LSM9DS1 accel/gyro not found. Check wiring.");
      i2cScan();
      uint8_t who6B = readReg(0x6B, WHO_AM_I_AG), who6A = readReg(0x6A, WHO_AM_I_AG);
      Serial.printf("I,WHO_AM_I 0x6B=0x%02X 0x6A=0x%02X (expect 0x68)\n", who6B, who6A);
    }
    attempt++;
    delay(500);
    Wire.end();
    i2cBusRecover();
    Wire.begin(PIN_SDA, PIN_SCL);
    Wire.setClock(400000);
    Wire.setTimeOut(50);
  }
  Serial.printf("I,LSM9DS1 found at 0x%02X\n", agAddr);

  writeReg(CTRL_REG8, 0x05);
  delay(10);
  writeReg(CTRL_REG8, 0x44);
  writeReg(CTRL_REG1_G, 0x60);
  writeReg(CTRL_REG6_XL, 0x60);

  Serial.println("R");
}

void loop() {
  while (Serial.available()) handleCommand((char)Serial.read());

  if (!(readReg(agAddr, STATUS_REG) & 0x02)) return;

  int16_t a[3], g[3];
  readXYZ(OUT_X_L_XL, a);
  readXYZ(OUT_X_L_G,  g);

  float s[N_AXES];
  for (int i = 0; i < 3; i++) {
    s[i]     = a[i] * ACCEL_G_PER_LSB;
    s[i + 3] = g[i] * GYRO_DPS_PER_LSB;
  }
  float w = gyroMag(s);

  if (debugW && (++dbgCounter % 12 == 0)) Serial.printf("W,%.1f\n", w);

  if (state == ST_IDLE) {
    for (int i = 0; i < N_AXES; i++) pre[preHead][i] = s[i];
    preHead = (preHead + 1) % PREROLL;
    if (preCount < PREROLL) preCount++;

    if (mode == MODE_AUTO) {
      if (w > START_THRESH) {
        if (++startCount >= 3) startRecording();
      } else {
        startCount = 0;
      }
    }
  } else {
    if (nBuf < MAX_SAMPLES) {
      for (int i = 0; i < N_AXES; i++) buf[nBuf][i] = s[i];
      nBuf++;
    }
    bool full = (nBuf >= MAX_SAMPLES);
    if (mode == MODE_AUTO) {
      if (w < STOP_THRESH) stillCount++; else stillCount = 0;
      if (stillCount >= STOP_SAMPLES || full) finishRecording();
    } else if (full) {
      finishRecording();
    }
  }
}
