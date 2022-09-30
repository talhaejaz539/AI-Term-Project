import json
import pickle
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import pyttsx3

# Lemmatizing the words
lemmatizer = WordNetLemmatizer()
contents = json.loads(open("contents.json").read())

# Load Pickle files
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbotmodel.h5")

# Clean Up sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Create Bag of Words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict the class of the string given
def predict_class(sentence):
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    sol = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]
    sol.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in sol:
        return_list.append({"content": classes[r[0]], "probability": str(r[1])})
    return return_list

# Get Response on the basis of prediction
def get_response(contents_list, contents_json):
    tag = contents_list[0]["content"]
    list_of_contents = contents_json["contents"]
    for i in list_of_contents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


# Running the Bot
completion_msg = "Hurray! Bot is running! :)"
print(completion_msg)
engine = pyttsx3.init()
engine.setProperty("rate", 200)
engine.say(completion_msg)
engine.runAndWait()

# Query Handler
print("Enter [e, q, exit, quit] to STOP")
while True:
    message = input("Question: ")
    if message.lower() in ["e", "q", "exit", "quit"]:
        print("Thanks for using Chatbot JD, hope to see you soon, Take care :)")
        engine = pyttsx3.init()
        engine.setProperty("rate", 200)
        engine.say("Thanks for using Chatbot JD, hope to see you soon, Take care :)")
        engine.runAndWait()
        break
    else:
        ints = predict_class(message)
        res = get_response(ints, contents)
        print(res)
        engine = pyttsx3.init()
        engine.setProperty("rate", 200)
        engine.say(res)
        engine.runAndWait()
