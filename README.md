# Animatronic Robot Project

This project is an AI-powered animatronic robotic system built using a Raspberry Pi.

## Features
- Speech recognition using Vosk
- AI responses using Ollama (Gemma 2B)
- Text-to-speech with eSpeak
- Mouth movement synced with speech
- Face tracking using camera module
- Eye movement and blinking
- Gesture recognition (hand wave)
- Head and body movement using servos

## Hardware Used
- Raspberry Pi 5
- PCA9685 PWM Servo Driver
- SG90 Servo (head, hand, eyes, mouth, eyelids, head turn)
- MG996R Servo (body turn)
- Raspberry Pi Camera Module 3
- Microphone and speaker
- Power Supply (5V 10A)

## Software Used
- Python
- OpenCV
- MediaPipe
- Vosk
- Ollama

## Setup Instructions

1. Install dependencies:
- Python libraries
- Vosk
- OpenCV
- MediaPipe

2. Download Vosk model:
https://alphacephei.com/vosk/models

Use:
vosk-model-small-en-us-0.15

Place it in the same folder as the Python file.

## Usage

Say:
- "robot" to activate
- "robot terminate" to reset

Example commands:
- "robot look left"
- "robot turn right"

## Author
Ella Winders 
