from html import entities
from unittest import result
from vosk import Model, KaldiRecognizer
import os
import pyaudio
import pyttsx3
import json
import core
from nlu.model import classify

engine = pyttsx3.init()  # sintaxe de fala

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[-1].id)


def speak(text):
    engine.say(text)
    engine.runAndWait()


def evaluate(text):
    
    # reconhecer entidade do texto
    entity = classify(text)
    
    if entity == 'time\\getTime':
        speak(core.SystemInfo.get_time())
    elif entity == 'time\\get_Date':
        speak(core.SystemInfo.get_date())
        
    # Abrir programas
    elif entity == 'open\\notepad':
        speak('Abrindo o bloco de notas')
        os.system('notepad.exe')    
        
    print('Text: {} Entity: {}'.format(text, entity))
    
    
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
            evaluate(text)

            