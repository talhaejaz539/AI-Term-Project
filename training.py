import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD


lemmatizer = WordNetLemmatizer()

contents = json.loads(open("contents.json").read())

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ",", "<"]

# Add words in documents
for content in contents["contents"]:
    for pattern in content["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, content["tag"]))
        if content["tag"] not in classes:
            classes.append(content["tag"])
#print(documents)

# Lemmatizing the words
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
output_empty = [0] * len(classes)

# Add patterns in documents
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Make Numpy array of Training
random.shuffle(training)
training = np.array(training)
trainx = list(training[:, 0])
trainy = list(training[:, 1])

# Making Neural Network
model = Sequential()
model.add(Dense(128, input_shape=(len(trainx[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(trainy[0]), activation="softmax"))

# compile the model, and Save it
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
hist = model.fit(np.array(trainx), np.array(trainy), epochs=200, batch_size=5, verbose=1)
model.save("chatbotmodel.h5", hist)
print("Done")