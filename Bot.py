import nltk
from nltk.stem.lancaster import LancasterStemmer
from torch import classes, true_divide
from xgboost import train
stemmer = LancasterStemmer()
import pickle
import numpy
import tflearn
import tensorflow
import random
import json
import speech_recognition as sr


with open("intent.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f: #Openting data try
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    #taking data from json
    for intent in data ["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)
    # one hot encoding or bag of words

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for i in words:
            if i in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    
    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output),f)

#Training the model (ML), using softmax activation output
tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=100, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# Making Predictions





def AudioToText():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print('Please say Something...')
        audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            print("You have said: \n " + text)
            return text

        except Exception as e:
            print("Error : " + str(e))

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

def chat():
    responses = ''
    print("Start Talking with the bot! (say quit to exit)")
    while True:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print('Please say Something...')
            audio = r.listen(source)

            try:
                text = r.recognize_google(audio)
                print("You have said: \n " + text)
                if text.lower() == "quit":
                    break
                
                print("check1")
                results = model.predict([bag_of_words(text,words)])[0]
                results_index = numpy.argmax(results) #gives highest values from the list
                tag = labels[results_index]
                print("check2")
                if results[results_index] > 0.6:
                    for tg in data["intents"]:
                        print("check3")
                        if tg["tag"] == tag:
                            responses = tg["responses"]
                        print(random.choice(responses))
                

            except Exception as e:
                print("Error : " + str(e))
                break
        

   

chat()
#AudioToText()
#TF_Model()