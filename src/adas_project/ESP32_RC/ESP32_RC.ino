/*
  ESP32 (ESP-WROOM-32) RC car actuator.  Flash once - all tuning lives in
  rc_controller.py on the laptop.

  This firmware does no steering maths. It just executes commands, which
  arrive as newline-separated text over USB serial (115200) AND/OR as UDP
  packets on port 4210 (both work at the same time):

    A <s1> <s2>   servo angles in degrees (decimals ok): s1 -> GPIO18,
                  s2 -> GPIO19. Angles are sent as-is (only clamped to the
                  physical 0..180 range).
    M <-255..255> motor PWM (negative = reverse)
    STOP          motor off
    PING          replies "PONG" on the channel it came from

  Failsafe: if no A/M/STOP arrives for 500 ms the motor is switched off.

  Boot banner: "ROVER READY reset=<reason>" (POWERON, BROWNOUT, WDT, PANIC, SW, EXT, OTHER) - the Pi's
  relay shows why the board restarted (a BROWNOUT means the supply dipped under the motor / servos).

  Radio: when the Pi drives the board over USB serial and nothing ever arrived over UDP, WiFi is switched
  off after RADIO_OFF_AFTER_SERIAL serial commands (the radio's transmit peaks are a load the car's weak
  supply does not need). A reset brings WiFi back. Set RADIO_OFF_WHEN_SERIAL to false to keep it on.

  WiFi: joins the network below; if that fails within 15 s it also starts
  its own "RC_CAR" access point. Board: "ESP32 Dev Module", core 3.x.
*/

#include <WiFi.h>
#include <WiFiUdp.h>
#include <ESPmDNS.h>
#include "esp_system.h"   // esp_reset_reason()

// ============================================================
// WIFI
// ============================================================

#include "secrets.h"   // WIFI_SSID, WIFI_PASSWORD, AP_SSID, AP_PASSWORD (AP: 8+ characters) - git-ignored, see secrets.example.h
const unsigned long WIFI_FALLBACK_MS = 15000;


const char* HOSTNAME = "rccar";         // reachable as rccar.local
const unsigned int UDP_PORT = 4210;


// ============================================================
// PINS
// ============================================================

const int MOTOR_PWM_PIN = 25;
const int MOTOR_DIR_PIN = 26;

const int SERVO1_PIN = 18;
const int SERVO2_PIN = 19;


// ============================================================
// PWM SETTINGS
// ============================================================

const uint32_t MOTOR_PWM_FREQ = 20000;   // 20 kHz, above audible range
const uint8_t  MOTOR_PWM_BITS = 8;       // duty 0..255

const uint32_t SERVO_PWM_FREQ = 50;
const uint8_t  SERVO_PWM_BITS = 14;
const float SERVO_MIN_US = 544.0;        // same mapping as Arduino Servo.h
const float SERVO_MAX_US = 2400.0;

const unsigned long COMMAND_TIMEOUT_MS = 500;

const bool RADIO_OFF_WHEN_SERIAL = true;
const unsigned long RADIO_OFF_AFTER_SERIAL = 250;   // serial commands (5 s at 50 Hz) with no UDP command ever seen
const unsigned int DIR_CHANGE_DEAD_MS = 2;          // motor PWM held at zero for this long around a direction change


// ============================================================
// STATE
// ============================================================

WiFiUDP udp;
char packetBuffer[128];

bool udpUp = false;
bool apUp = false;
unsigned long bootMs = 0;

unsigned long lastCommandTime = 0;

int targetMotor = 0;
int currentMotor = 0;

// Servo angles in tenths of a degree; -1 = not set yet
int target1 = -1, target2 = -1;
int current1 = -1, current2 = -1;

char lineBuffer[64];
int lineLength = 0;

unsigned long serialCommands = 0;   // drive commands received over USB serial
unsigned long udpCommands = 0;      // ... and over UDP
bool radioOff = false;
int motorDir = 0;                   // direction the driver is switched to: +1 / -1, 0 = not set yet

enum Source { FROM_SERIAL, FROM_UDP };


// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);

  pinMode(MOTOR_DIR_PIN, OUTPUT);

  ledcAttach(MOTOR_PWM_PIN, MOTOR_PWM_FREQ, MOTOR_PWM_BITS);
  ledcAttach(SERVO1_PIN, SERVO_PWM_FREQ, SERVO_PWM_BITS);
  ledcAttach(SERVO2_PIN, SERVO_PWM_FREQ, SERVO_PWM_BITS);

  setMotor(0);

  WiFi.setHostname(HOSTNAME);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);   // modem sleep adds latency to incoming packets
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  bootMs = millis();
  lastCommandTime = millis();

  Serial.print("ROVER READY reset=");
  Serial.println(resetReason());
}


const char* resetReason() {
  switch (esp_reset_reason()) {
    case ESP_RST_POWERON:  return "POWERON";
    case ESP_RST_BROWNOUT: return "BROWNOUT";
    case ESP_RST_SW:       return "SW";
    case ESP_RST_PANIC:    return "PANIC";
    case ESP_RST_INT_WDT:
    case ESP_RST_TASK_WDT:
    case ESP_RST_WDT:      return "WDT";
    case ESP_RST_EXT:      return "EXT";
    default:               return "OTHER";
  }
}


// ============================================================
// MAIN LOOP
// ============================================================

void loop() {

  readSerial();
  readUdp();
  manageWifi();

  // Apply only what changed, and only the newest value.
  if (target1 != current1 && target1 >= 0) {
    writeServo(SERVO1_PIN, target1);
    current1 = target1;
  }

  if (target2 != current2 && target2 >= 0) {
    writeServo(SERVO2_PIN, target2);
    current2 = target2;
  }

  if (targetMotor != currentMotor) {
    setMotor(targetMotor);
  }

  if (millis() - lastCommandTime > COMMAND_TIMEOUT_MS && currentMotor != 0) {
    targetMotor = 0;
    setMotor(0);
  }
}


// ============================================================
// INPUT
// ============================================================

void readSerial() {

  while (Serial.available() > 0) {

    char c = Serial.read();

    if (c == '\n') {

      lineBuffer[lineLength] = '\0';
      handleCommand(String(lineBuffer), FROM_SERIAL);
      lineLength = 0;
    }

    else if (c != '\r') {

      if (lineLength < (int)sizeof(lineBuffer) - 1) {
        lineBuffer[lineLength++] = c;
      } else {
        lineLength = 0;   // overlong garbage line, discard
      }
    }
  }
}


void readUdp() {

  if (!udpUp) return;

  while (udp.parsePacket() > 0) {

    int len = udp.read(packetBuffer, sizeof(packetBuffer) - 1);
    if (len <= 0) continue;
    packetBuffer[len] = '\0';

    char* line = strtok(packetBuffer, "\n");
    while (line != NULL) {
      handleCommand(String(line), FROM_UDP);
      line = strtok(NULL, "\n");
    }
  }
}


void manageWifi() {

  if (radioOff) return;

  // the Pi drives over USB and nobody uses the radio: switch it off (see the header)
  if (RADIO_OFF_WHEN_SERIAL && udpCommands == 0 && serialCommands >= RADIO_OFF_AFTER_SERIAL) {
    if (udpUp) udp.stop();
    WiFi.disconnect(true);
    WiFi.mode(WIFI_OFF);
    udpUp = false;
    radioOff = true;
    Serial.println("WIFI OFF (serial control)");
    return;
  }

  if (!udpUp && WiFi.status() == WL_CONNECTED) {

    udp.begin(UDP_PORT);
    MDNS.begin(HOSTNAME);
    udpUp = true;

    Serial.print("WiFi connected. IP address: ");
    Serial.println(WiFi.localIP());
  }

  if (!udpUp && !apUp && millis() - bootMs > WIFI_FALLBACK_MS) {

    WiFi.mode(WIFI_AP_STA);   // keeps trying the home network too
    WiFi.softAP(AP_SSID, AP_PASSWORD);
    apUp = true;

    udp.begin(UDP_PORT);
    MDNS.begin(HOSTNAME);
    udpUp = true;

    Serial.print("Home WiFi not found. Own access point ");
    Serial.print(AP_SSID);
    Serial.print(" at ");
    Serial.println(WiFi.softAPIP());
  }
}


// ============================================================
// COMMANDS
// ============================================================

void handleCommand(String command, Source source) {

  command.trim();
  command.toUpperCase();

  if (command.startsWith("A ")) {

    int split = command.indexOf(' ', 2);
    if (split < 0) return;

    // a line damaged on the way (a supply dip, a half-received packet) must not move the wheels: String.toFloat()
    // turns garbage into 0.0, i.e. full lock - so both angles have to be clean numbers or the line is ignored
    float a1, a2;
    if (!parseNumber(command.substring(2, split), a1) || !parseNumber(command.substring(split + 1), a2)) return;

    target1 = (int)round(constrain(a1, 0.0, 180.0) * 10.0);
    target2 = (int)round(constrain(a2, 0.0, 180.0) * 10.0);
    countCommand(source);
  }

  else if (command.startsWith("M ")) {

    float pwm;
    if (!parseNumber(command.substring(2), pwm)) return;

    targetMotor = constrain((int)pwm, -255, 255);
    countCommand(source);
  }

  else if (command == "STOP") {

    targetMotor = 0;
    countCommand(source);
  }

  else if (command == "PING") {

    if (source == FROM_SERIAL) {
      Serial.println("PONG");
    } else {
      udp.beginPacket(udp.remoteIP(), udp.remotePort());
      udp.print("PONG");
      udp.endPacket();
    }
  }
}


// A drive command arrived: feed the failsafe, remember which channel is in use.
void countCommand(Source source) {
  lastCommandTime = millis();
  if (source == FROM_SERIAL) serialCommands++; else udpCommands++;
}


// The whole text must be one number ("87", "-120", "76.5"); false for anything else.
bool parseNumber(String text, float &value) {
  text.trim();
  if (text.length() == 0) return false;
  char* end = NULL;
  value = strtof(text.c_str(), &end);
  return end != text.c_str() && *end == 0 && !isnan(value) && !isinf(value);
}


// ============================================================
// OUTPUTS
// ============================================================

void writeServo(int pin, int tenthsOfDegree) {

  float degrees = tenthsOfDegree / 10.0;

  float pulseUs = SERVO_MIN_US + degrees * (SERVO_MAX_US - SERVO_MIN_US) / 180.0;
  uint32_t duty = (uint32_t)(pulseUs * (float)(1UL << SERVO_PWM_BITS) / (1000000.0 / SERVO_PWM_FREQ));

  ledcWrite(pin, duty);
}


void setMotor(int speed) {

  speed = constrain(speed, -255, 255);

  currentMotor = speed;

  if (speed == 0) {
    ledcWrite(MOTOR_PWM_PIN, 0);
    return;
  }

  // Direction change: PWM to zero first, switch the direction pin, wait, then drive - the driver's bridge is
  // never asked to reverse under full duty (the Pi's brake pulse does reverse the motor at speed).
  int dir = speed > 0 ? 1 : -1;
  if (dir != motorDir) {
    ledcWrite(MOTOR_PWM_PIN, 0);
    digitalWrite(MOTOR_DIR_PIN, dir > 0 ? HIGH : LOW);
    delay(DIR_CHANGE_DEAD_MS);
    motorDir = dir;
  }
  ledcWrite(MOTOR_PWM_PIN, abs(speed));
}
