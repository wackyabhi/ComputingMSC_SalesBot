# importing libraries

import string
import json
import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout
from torch import dropout
nltk.download("punkt")
nltk.download("wordnet")

# loading the dataset: intents.json

with open("intent.json") as file:
    data = json.load(file)

# identying Target and Feature for NLP

words = [] #for Bag of words model/vocabulary for patterns
classes = [] #For BOW model/vocabulart for tags
data_X = [] #for storing each pattern
data_y = [] #for storing tag corresponding to each pattern in data_X


#Iterating over all the intents
for intent in data["intents"]:
    for pattern in intent["patterms"]:
        tokens = nltk.word_tokenize(pattern) #tokenize each pattern
        words.extend(tokens) #and append tokens to words
        data_X.append(pattern) #appending pattern to data_X
        data_y.append(intent["tag"]) #appending the associated tag to each pattern

        #adding the tag to the classes if its not already there
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

#initializing lematizer to get stem of words
lematizer = WordNetLemmatizer()

#lematize all the words in the vocab and convert them to lowercase
# if the words dont appear in punctuation
words = [lematizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
#sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
words = sorted(set(words))
classes = sorted(set(classes))

#One hot encoding / text to numbers

training = []
out_empty = [0] * len(classes)
# creating the bag of words model

for idx, doc in enumerate(data_X):
    bow = []
    text = lematizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    
    #mark the index of class that the current pattern is associated to

    output_row = list(out_empty)
    output_row[classes.index(data_y[idx])] = 1
    # add the one hot encoded BOW and associated classes to training
    training.append([bow,output_row])
#shuffle the data and convert it to an array

random.shuffle(training)
training = np.array(training, dtype=object)
#split the features and target labels
train_X = np.array(list(training[:,0]))
train_Y = np.array(list(training[:,1]))



#Building Neural Network Model

model = Sequential()
model.add(Dense(128, input_shape=(len(train_X[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_Y[0]), activation="softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=["accuracy"])
print(model.summary())
model.fit(x=train_X,y=train_Y,epochs=1000, verbose=1)