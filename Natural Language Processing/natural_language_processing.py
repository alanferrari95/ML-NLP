#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:13:51 2019

@author: Alan
"""

# Natural Language Processing

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

# Limpieza de texto
import re #Regular Expression
import nltk #Natural Languaje ToolKit
nltk.download('stopwords') #palabras irrelevantes
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #infinitivos de la palabra
corpus = []
for i in range(0, 1000):
    #elimina TODO lo que no sea texto (puntos, parentesis, comas, etc)
    #[^a-zA-Z] = elimina todo menos lo que está entre parentesis
    #re.SUB sustituye X elemento por Y elemento. En este caso todo lo que no sea a-zA-Z, por un espacio
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) 
    review = review.lower() #pone todo en minuscula
    review = review.split() #separa la oración creando una lista
    ps = PorterStemmer() #transofmr las palabras conjugadas a infinitivo
    #"review" va a estar compuestas por todas las palabras MENOS las que aparezcan en el stopwords
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review) #unimos la lista
    corpus.append(review)
    
# Crear el Bag of Words
from sklearn.feature_extraction.text import CountVectorizer #de palabra a vector
cv = CountVectorizer(max_features = 1500) #max_features es maximo de columnas. Palabras mas frecuentes
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

(55+91)/200