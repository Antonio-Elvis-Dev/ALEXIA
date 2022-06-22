from ast import While
from statistics import mode
from tkinter import W
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.utils import to_categorical

data = yaml.safe_load(open('nlu\\train.yml', 'r', encoding='utf-8').read())
inputs, outputs = [], []

for command in data['commands']:
    inputs.append(command['input'].lower())
    outputs.append('{}\{}'.format(command['entity'], command['action']))

# processar texto : palavras, caracteres, bytes

'''chars = set()

for input in inputs + outputs:
    for ch in input:
        if ch not in chars:
            chars.add(ch)'''

 # mapear char-id
chr2idx = {}
idx2chr = {}

'''for i, ch in enumerate(chars):
    chr2idx[ch] = i
    idx2chr[i] = ch

print('Número de chars:', len(chars))'''

max_seq = max([len(bytes(x.encode('utf-8'))) for x in inputs])

print('Max seq:', max_seq)

# criar dataset

input_data = np.zeros((len(inputs), max_seq,256), dtype='float32')

for i, inp in enumerate(inputs):
    for k, ch in enumerate(bytes(inp.encode('utf-8'))):
        input_data[i, k, int(ch)] = 1.0

# labels para classificador
'''
input_data = np.zeros((len(inputs), max_seq, len(chars)), dtype='int32')

for i, input in enumerate(inputs):
    for k, ch in enumerate(input):
        input_data[i, k] = chr2idx[ch]

'''

output_data = []
labels = set(outputs)
label2idx = {}
idx2label = {}

for k, label in enumerate(labels):
    label2idx[label] = k
    idx2label[k] = label
    
for output in outputs:
    output_data.append(label2idx[output])


output_data = to_categorical(output_data, len(output_data))

print(output_data[0])

model = Sequential()
model.add(LSTM(128))
model.add(Dense(len(output_data), activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])
# model.summary()

model.fit(input_data, output_data, epochs=128)

# Classificar texto em uma entidade
def classify(text):
    # criar um array de entrada
    x = np.zeros((1,max_seq,256), dtype='float32')
    
    # preencher o array com dados de texto
    for k, ch in enumerate(bytes(text.encode('utf-8'))):
        x[0,k,int(ch)]  = 1.0
    
#faz a previsão
    out = model.predict(x)
    idx = out.argmax()
    print(idx2label[idx])
    
while True:
    text = input('Digite algo: ')
    classify(text)
