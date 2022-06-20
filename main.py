from unittest import result
from vosk import Model, KaldiRecognizer
import os
import pyaudio
import pyttsx3
import json
import core

engine = pyttsx3.init()  # sintaxe de fala

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[-1].id)


def speak(text):
    engine.say(text)
    engine.runAndWait()

# reconhecimento de fala


model = Model("model")
rec = KaldiRecognizer(model, 16000)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1,rate=16000, input=True, frames_per_buffer=2048)
stream.start_stream()

# loop do reconhecimento de fala

while True:
    data = stream.read(2048)
    if len(data) == 0:
        break

    if rec.AcceptWaveform(data):
        result = rec.Result()
        result = json.loads(result)

        if result is not None:
            text = result['text']

            print(text)
            
            if text == 'que horas são' or text == 'me diga as horas':
                speak(core.SystemInfo.get_time())
            