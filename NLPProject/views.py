from django.shortcuts import render,HttpResponse

from keras.models import load_model
import tensorflow  as tf
from nltk.stem import WordNetLemmatizer 
import json
import string
import random 
import nltk
import numpy as np
# from mydb import databaseconnect
# data for intent
data = {"intents": [
             {"tag": "greeting",
              "patterns": ["Hello", "How are you?", "Hi there", "Hi", "Whats up"],
              "responses": ["Hii !! First of all  i would need some details .Let's start with your name","Hey!! What is your name?"],
             },
             {"tag": "age",
              "patterns": ["my name is manish kumar", "name is suraj singh", "dinesh","neha","suman"],
              "responses": ["Okay!! what is your age?"]
             },
             {"tag": "email",
              "patterns": ["age is 21","my age 21","age 27","21"],
              "responses": ["what is your email?"]
             },
             {"tag": "phone",
              "patterns": ["my email is jan@gmail.com","gd@gerni.co","email is sud@gmail.com","sura@gmail.com"],
              "responses": ["what is you phone number?"]
             },
             {"tag": "query",
              "patterns": [ "my phone number is 7898787968","7787966534","phone number 6698563210","8875637098"],
              "responses": ["what is your query?"]
             },
	      {"tag": "thanku",
              "patterns": [ "what you know","what can you do for me","nothing"],
              "responses": ["Okay you want to save the details?"]
             },
             {"tag": "confirmation",
              "patterns": [ "yes","no"],
              "responses": ["what changes you want"]
             },
             {"tag": "changing",
              "patterns": [ "i want to change my name","name","age","want to change email","change phone","phone"],
              "responses": ["sure"]
             }

]}

lemmatizer = WordNetLemmatizer()
# Each list to create
words = []
classes = []
doc_X = []
doc_y = []
# Loop through all the intents
# tokenize each pattern and append tokens to words, the patterns and
# the associated tag to their associated list
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent["tag"])
    
    # add the tag to the classes if it's not there already 
    if intent["tag"] not in classes:
        classes.append(intent["tag"])
# lemmatize all the words in the vocab and convert them to lowercase
# if the words don't appear in punctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
# sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
words = sorted(set(words))
classes = sorted(set(classes))


training = []
out_empty = [0] * len(classes)
# creating the bag of words model
for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    # mark the index of class that the current pattern is associated
    # to
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1
    # add the one hot encoded BoW and associated classes to training 
    training.append([bow, output_row])
# shuffle the data and convert it to an array
random.shuffle(training)
training = np.array(training, dtype=object)
# split the features and target labels
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

#import model
model=load_model('model.h5')
def clean_text(text): 
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  return tokens

def bag_of_words(text, vocab): 
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens: 
    for idx, word in enumerate(vocab):
      if word == w: 
        bow[idx] = 1
  return np.array(bow)

def pred_class(text, vocab, labels): 
  bow = bag_of_words(text, vocab)
  result = model.predict(np.array([bow]))[0]
  thresh = 0.2
  y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

  y_pred.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]])
  return return_list

def get_response(intents_list, intents_json): 
  tag = intents_list[0]
  list_of_intents = intents_json["intents"]
  for i in list_of_intents: 
    if i["tag"] == tag:
      result = random.choice(i["responses"])
      break
  return result

  
def index(request):
    return render(request,'index.html')
def User(request):
      while True:
        message=request.GET['userinput']
        intents = pred_class(message, words, classes)
        result = get_response(intents, data)
        
        context={
                "Response":result,
                }
        
        
              
        return render(request,'index.html',context)

# Name = request.GET['userinput']
# Age = request.GET['userinput']
# Email = request.GET['userinput']
# Phone = request.GET['userinput']
# Query = request.GET['userinput']
# databaseconnect(Name, Age, Email, Phone, Query)


