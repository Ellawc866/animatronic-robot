import queue
import sounddevice as sd
import json, subprocess, threading, os, sys
from vosk import Model, KaldiRecognizer
import board, busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

import re
import ast
import operator
import time

import cv2
import mediapipe as mp
from picamera2 import Picamera2

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = 'vosk-model-small-en-us-0.15'
LLM_MODEL = 'gemma:2b'
WAKE_WORD = 'robot'
RESET_PHRASE = 'robot terminate'
RATE = 16000

# -------------------------
# ANSI CLEANER
# -------------------------
ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def remove_ansi(text):
    return ansi_escape.sub('', text)

# -------------------------
# SERVO SETUP
# -------------------------
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

mouth = servo.Servo(pca.channels[0], min_pulse=500, max_pulse=2500)
track_servo = servo.Servo(pca.channels[1], min_pulse=500, max_pulse=2500)

# 🖐️ HAND SERVO (NEW - SG90 channel 2)
hand_servo = servo.Servo(pca.channels[2], min_pulse=500, max_pulse=2500)

eyelid_servo = servo.Servo(pca.channels[3], min_pulse=500, max_pulse=2500)

head_servo = servo.Servo(pca.channels[4], min_pulse=500, max_pulse=2500)

body_servo = servo.Servo(pca.channels[5], min_pulse=500, max_pulse=2500)

# -------------------------
# HEAD SERVO
# -------------------------
HEAD_LEFT = 160
HEAD_RIGHT = 20
HEAD_CENTER = 90
head_servo.angle = HEAD_CENTER

# -------------------------
# BODY SERVO
# -------------------------
BODY_LEFT = 160
BODY_RIGHT = 20
BODY_CENTER = 90
body_servo.angle = BODY_CENTER

# -------------------------
# MOUTH
# -------------------------
OPEN_ANGLE = 110
CLOSED_ANGLE = 70

def mouth_open():
    mouth.angle = OPEN_ANGLE

def mouth_closed():
    mouth.angle = CLOSED_ANGLE

mouth_closed()

# -------------------------
# TRACKING SERVO
# -------------------------
LEFT_ANGLE = 160
CENTER_ANGLE = 90
RIGHT_ANGLE = 20

current_angle = CENTER_ANGLE
track_servo.angle = current_angle

# -------------------------
# EYELIDS
# -------------------------
EYELID_OPEN = 30
EYELID_CLOSED = 120
eyelid_servo.angle = EYELID_OPEN

last_blink = time.time()

# -------------------------
# HAND WAVE STATE
# -------------------------
hand_waving = False
last_wave_time = 0

HAND_LEFT = 30
HAND_RIGHT = 150
hand_servo.angle = HAND_LEFT

# -------------------------
# CAMERA THREAD
# -------------------------
def camera_worker():
    global current_angle, last_blink, hand_waving, last_wave_time

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"size": (320, 240)}
        )
    )
    picam2.start()

    # FACE detection
    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)

    # HAND detection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )

    CENTER_LEFT = 110
    CENTER_RIGHT = 210

    while True:
        frame = picam2.capture_array()
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = face_detection.process(rgb)
        hand_results = hands.process(rgb)

        cv2.line(frame, (CENTER_LEFT, 0), (CENTER_LEFT, h), (255, 0, 0), 1)
        cv2.line(frame, (CENTER_RIGHT, 0), (CENTER_RIGHT, h), (255, 0, 0), 1)

        direction = "NO FACE"

        # -------------------------
        # BLINK
        # -------------------------
        if time.time() - last_blink >= 10:
            eyelid_servo.angle = EYELID_CLOSED
            time.sleep(0.4)
            eyelid_servo.angle = EYELID_OPEN
            last_blink = time.time()

        # -------------------------
        # FACE TRACKING
        # -------------------------
        if face_results.detections:
            d = face_results.detections[0]
            bbox = d.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            face_x = x + bw // 2

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            if face_x < CENTER_LEFT:
                direction = "LEFT"
                target_angle = LEFT_ANGLE
            elif face_x > CENTER_RIGHT:
                direction = "RIGHT"
                target_angle = RIGHT_ANGLE
            else:
                direction = "CENTER"
                target_angle = CENTER_ANGLE

            if target_angle != current_angle:
                track_servo.angle = target_angle
                current_angle = target_angle

        # -------------------------
        # HAND TRACKING (OPEN HAND → WAVE)
        # -------------------------
        global hand_waving, last_wave_time

        if hand_results.multi_hand_landmarks:
            hand = hand_results.multi_hand_landmarks[0]

            tips = [
                hand.landmark[4].y,
                hand.landmark[8].y,
                hand.landmark[12].y,
                hand.landmark[16].y,
                hand.landmark[20].y
            ]

            open_hand = all(t < hand.landmark[0].y for t in tips)

            if open_hand:
                direction = "OPEN HAND"

                if not hand_waving and time.time() - last_wave_time > 2:
                    hand_waving = True
                    last_wave_time = time.time()

                    def wave():
                        global hand_waving
                        for _ in range(3):
                            hand_servo.angle = HAND_LEFT
                            time.sleep(0.25)
                            hand_servo.angle = HAND_RIGHT
                            time.sleep(0.25)
                        hand_servo.angle = HAND_LEFT
                        hand_waving = False

                    threading.Thread(target=wave, daemon=True).start()

        cv2.putText(frame, direction, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Tracking Servo", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.05)

threading.Thread(target=camera_worker, daemon=True).start()

# -------------------------
# SPEECH RECOGNITION
# -------------------------
model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, RATE)
rec.SetWords(False)

audio_queue = queue.Queue(maxsize=50)
llm_queue = queue.Queue()

llm_busy = False
tts_process = None

# -------------------------
# MATH
# -------------------------
allowed_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod
}

def safe_eval(expr):
    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.BinOp):
            return allowed_operators[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return _eval(node.operand)
        raise ValueError()

    node = ast.parse(expr, mode='eval').body
    return _eval(node)

def is_math_expression(text):
    return bool(re.fullmatch(r"[0-9+\-*/().\s]+", text))

# -------------------------
# CLEAN TEXT
# -------------------------
def clean_text(text):
    text = remove_ansi(text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'[^a-zA-Z0-9.,!?\'"()\- ]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -------------------------
# REPEAT FIX
# -------------------------
GRAMMAR_WORDS = {
    "a", "i", "the", "is", "of", "to", "and", "in", "on", "for", "with", "was", "were", "are"
}

def fix_repetition(text):
    words = text.split()
    if not words:
        return ""

    cleaned = []

    for i, w in enumerate(words):
        lw = w.lower()

        if not cleaned:
            cleaned.append(w)
            continue

        prev = cleaned[-1]
        lp = prev.lower()

        # 1. Remove exact duplicates
        if lw == lp:
            continue

        # 2. Remove partial stutter (e.g. "cu cultural")
        if len(lp) <= 4 and w.lower().startswith(lp):
            cleaned[-1] = w  # replace fragment with full word
            continue

        # 3. Remove prefix overlap (e.g. "cult cultural")
        if lp in lw and len(lp) < len(lw):
            cleaned[-1] = w
            continue

        cleaned.append(w)

    # Final pass to ensure no duplicates remain
    final = []
    for w in cleaned:
        if not final or w != final[-1]:
            final.append(w)

    return " ".join(final)

# -------------------------
# AUDIO CALLBACK
# -------------------------
def audio_callback(indata, frames, time_info, status):
    try:
        audio_queue.put_nowait(bytes(indata))
    except queue.Full:
        pass

# -------------------------
# OLLAMA
# -------------------------
def ask_llm(prompt):
    try:
        r = subprocess.run(
            ['ollama', 'run', LLM_MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=180
        )

        out = r.stdout or ""
        out = remove_ansi(out)
        out = re.sub(r'\s+', ' ', out)

        return out.strip()

    except Exception:
        return "Sorry, I had a problem."

# -------------------------
# STOP SPEAKING
# -------------------------
def stop_speaking():
    global tts_process

    if tts_process and tts_process.poll() is None:
        try:
            tts_process.terminate()
            tts_process.kill()
        except:
            pass

    tts_process = None
    mouth_closed()

# -------------------------
# TTS
# -------------------------
def speak_blocking(text):
    global tts_process

    text = clean_text(text)

    wav_file = "/tmp/tts.wav"

    subprocess.run(
        ['espeak', '-v', 'en+f4', '-s', '128', '-w', wav_file, text],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    proc = subprocess.Popen(
        ['aplay', wav_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    tts_process = proc
    time.sleep(0.18)

    while proc.poll() is None:
        mouth_open()
        time.sleep(0.14)
        mouth_closed()
        time.sleep(0.14)

    tts_process = None
    mouth_closed()

# -------------------------
# LLM WORKER
# -------------------------
def llm_worker():
    global llm_busy

    while True:
        msg = llm_queue.get()
        if msg is None:
            return

        llm_busy = True

        if is_math_expression(msg):
            try:
                result = safe_eval(msg)
                reply = str(int(result) if result == result else result)
            except:
                reply = "Error"
        else:
            prompt = (
                "Answer in 1-2 short sentences. "
                "Do not repeat words. Do not restart or correct mid word. "
                "Speak cleanly and fluently.\n"
                + msg
            )

            raw = ask_llm(prompt)
            reply = clean_text(raw)
            reply = fix_repetition(reply)

        print("Robot:", reply)
        speak_blocking(reply)

        llm_busy = False

threading.Thread(target=llm_worker, daemon=True).start()

print("Say 'robot' to start. Say 'robot terminate' to reset. Say 'exit' to quit.\n")

# -------------------------
# AUDIO LOOP
# -------------------------
with sd.RawInputStream(
    samplerate=RATE,
    blocksize=4000,
    dtype='int16',
    channels=1,
    callback=audio_callback
):
    while True:
        data = audio_queue.get()

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get('text', '').strip().lower()

            if not text:
                continue

            print('Heard:', text)

            if RESET_PHRASE in text:
                stop_speaking()
                speak_blocking("Resetting now")
                os.execl(sys.executable, sys.executable, *sys.argv)

            if text == "exit":
                llm_queue.put(None)
                break

            if not text.startswith(WAKE_WORD):
                continue

            command = text[len(WAKE_WORD):].strip()

            if command == "look left":
                head_servo.angle = HEAD_LEFT
                speak_blocking("Looking left")
                continue

            if command == "look right":
                head_servo.angle = HEAD_RIGHT
                speak_blocking("Looking right")
                continue

            if command == "look forward":
                head_servo.angle = HEAD_CENTER
                speak_blocking("Looking forward")
                continue

            if command == "turn left":
                body_servo.angle = BODY_LEFT
                speak_blocking("Turning left")
                continue

            if command == "turn right":
                body_servo.angle = BODY_RIGHT
                speak_blocking("Turning right")
                continue

            if command == "turn forward":
                body_servo.angle = BODY_CENTER
                speak_blocking("Facing forward")
                continue

            if not command:
                speak_blocking("Yes?")
                continue

            if llm_busy:
                speak_blocking("One moment.")
                continue

            rec.Reset()

            print("Robot Thinking...")
            llm_queue.put(command)

            stop_speaking()

try:
    pca.deinit()
except:
    pass

print("Goodbye!")
