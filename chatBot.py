from nltk import tokenize
from nltk import tag
from nltk.corpus import conll2000
from nltk .corpus import reuters

import nltk
from snowballstemmer import TurkishStemmer
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam

import numpy as np
import pandas as pd
import numpy
import random
import json
import speech_recognition as sr

from google_speech import Speech
import  re
import  keyboard

###PARAMAETERS VOICE
VOICE_TIMEOUT=10
VOICE_PHRASE_TIME_LIMIT=3

with open(r"data.json") as file:
    data = json.load(file)
stemmer=TurkishStemmer()
words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
    if intent["tag"] not in labels:
        labels.append(intent["tag"])
words = [stemmer.stemWord(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stemWord(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stemWord(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

model = Sequential()
model.add(Dense(28, input_shape=(len(training[0]),), activation="relu"))
model.add(Dense(28, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(8, activation="softmax"))
model.summary()
model.compile(Adam(lr=.001), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(training, output, epochs=200, verbose=2, batch_size=8)

def checkInputMail(tag,inp):
    if tag == "mail":
        if "mail gönder" in inp:
            print("Lütfen mail atılacak ismi söyleyiniz")
            textToSound("Lütfen mail atılacak ismi söyleyiniz")
            return True
        if "ismi Jone" in inp:
            print("xxx adresine mail atılıyor.")
            textToSound("xxx adresine mail atılıyor")
            return True
        if "ismi Burak" in inp:
            print("xxx@gsu.edu.tr adresine mail atılıyor.")
            textToSound("xxx@gsu.edu.tr adresine mail atılıyor.")
            return True
    return False

def  checkInputKargo(tag,inp):
    if tag == "kargo":
        kargo_number = int(re.search(r'\d+', inp).group())
        print(str(kargo_number) + "numaralı kayıt bulundu")
        textToSound(str(kargo_number) + "numaralı kayıt bulundu")

def chat(inp):
    results = model.predict(np.asanyarray([bag_of_words(inp, words)]))[0]
    #print(results)
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    if (checkInputMail(tag, inp) == True):
        return
    if results[results_index] > 0.85:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        print("\r\n")
        choice = random.choice(responses)
        print(choice)
        textToSound(str(choice))
        print("\r\n")
        checkInputKargo(tag,inp)
    else:
        print("Tam olarak anlayamadım")
        textToSound("Tam olarak anlayamadım")



def recognizeVoice():
    r = sr.Recognizer()
    while True:
        print(" enter  speech estimation time in second");
        inp = input("You: ")
        if inp == "q":
            textToSound("program kapanıyor")
            break
        with sr.Microphone() as source:
            try:
                print("Sizi dinliyorum");
                #auido = r.listen(source)
                auido = r.listen(source, timeout=VOICE_TIMEOUT, phrase_time_limit=int(inp))
                print("Time is over recognizing words")
                speech_recognition_result= r.recognize_google(auido, language='tr')
                print("Recognized data: "+speech_recognition_result)
                chat(speech_recognition_result)
            except:
                pass;

def textToSound(text):
    # say "Hello World"
    lang = "tr"
    speech = Speech(text, lang)
    sox_effects = ("speed", "1")
    speech.play(sox_effects)


textToSound("Selam size nasıl yardımcı olabilirim")
##chat()
recognizeVoice()

