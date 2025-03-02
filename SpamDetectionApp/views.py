from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from genetic_selection import GeneticSelectionCV
from sklearn import linear_model


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

global rf, tfidf_vectorizer

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def UploadDataset(request):
    if request.method == 'GET':
       return render(request, 'UploadDataset.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})    

def SpamDetection(request):
    if request.method == 'GET':
       return render(request, 'SpamDetection.html', {})    

def Login(request):
    if request.method == 'GET':
       return render(request, 'Login.html', {})
    
def UserLogin(request):
    if request.method == 'POST':
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        if username == 'admin' and password == 'admin':
            context= {'data':"Welcome "+username}
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':'Invalid login details'}
            return render(request, 'Login.html', context)        
        
def UploadDatasetAction(request):
    if request.method == 'POST':
        file = request.FILES['t1']
        dataset = pd.read_csv("Dataset/spam_ham_dataset.csv",encoding='iso-8859-1',nrows=50)
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Class Label</font></th>'
        output+='<th><font size=3 color=black>Email Message</font></th>'
        for i in range(len(dataset)):
            msg = dataset.get_value(i, 'text')
            label = dataset.get_value(i, 'label')
            output+='<tr><td><font size=3 color=black>'+str(label)+'</font></td>'
            output+='<td><font size=3 color=black>'+msg+'</font></td>'
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'ViewDataset.html', context)

def TrainDataGA(request):
    if request.method == 'GET':
        global rf, tfidf_vectorizer
        Y = np.load("model/Y.txt.npy")
        X = np.load("model/X.txt.npy")
        estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr") #BUILDING GENETIC ALGORITHM WITH NAME CALLED SELECTOR
        selector = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=1,
                                  scoring="accuracy",
                                  max_features=5,
                                  n_population=50,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=10,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  n_gen_no_change=10,
                                  caching=True,
                                  n_jobs=-1)
        selector = selector.fit(X, Y)#OPTIMIZING FEATURES WITH GENETIC ALGORITHM OBJECT SELECTOR
        print(selector.support_)
        X_selected_features = X[:,selector.support_==True]#SELECTING IMPORTANT FEATURES
        data = X[:,selector.support_==True]#assiging all seleccted features to data
        X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.2, random_state = 0)
        rf = RandomForestClassifier() #now training random forest with selected features
        rf.fit(data, Y)
        predict = rf.predict(X_test)
        acc = accuracy_score(y_test,predict)*100
        p = precision_score(y_test,predict,average='macro') * 100
        r = recall_score(y_test,predict,average='macro') * 100
        f = f1_score(y_test,predict,average='macro') * 100
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Algorithm Name</font></th>'
        output+='<th><font size=3 color=black>Accuracy</font></th>'
        output+='<th><font size=3 color=black>Precision</font></th>'
        output+='<th><font size=3 color=black>Recall</font></th>'
        output+='<th><font size=3 color=black>FScore</font></th></tr>'

        output+='<tr><td><font size=3 color=black>Random Forest with Genetic Algorithm</font></td>'
        output+='<td><font size=3 color=black>'+str(acc)+'</font></td>'
        output+='<td><font size=3 color=black>'+str(p)+'</font></td>'
        output+='<td><font size=3 color=black>'+str(r)+'</font></td>'
        output+='<td><font size=3 color=black>'+str(f)+'</font></td>'
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'TrainData.html', context)      
    

def TrainData(request):
    if request.method == 'GET':
        global rf, tfidf_vectorizer
        Y = np.load("model/Y.txt.npy")
        X = np.load("model/X.txt.npy")
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)
        with open('model/tfidf.txt', 'rb') as file:
            tfidf_vectorizer = pickle.load(file)
        file.close()

        if os.path.exists('model/rf.txt'):
            with open('model/rf.txt', 'rb') as file:
                rf = pickle.load(file)
            file.close()
        else:
            rf = RandomForestClassifier()
            rf.fit(X_train, y_train)
            with open('model/rf.txt', 'wb') as file:
                pickle.dump(rf, file)
            file.close()
        predict = rf.predict(X_test)
        acc = accuracy_score(y_test,predict)*100
        p = precision_score(y_test,predict,average='macro') * 100
        r = recall_score(y_test,predict,average='macro') * 100
        f = f1_score(y_test,predict,average='macro') * 100
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Algorithm Name</font></th>'
        output+='<th><font size=3 color=black>Accuracy</font></th>'
        output+='<th><font size=3 color=black>Precision</font></th>'
        output+='<th><font size=3 color=black>Recall</font></th>'
        output+='<th><font size=3 color=black>FScore</font></th></tr>'

        output+='<tr><td><font size=3 color=black>Random Forest</font></td>'
        output+='<td><font size=3 color=black>'+str(acc)+'</font></td>'
        output+='<td><font size=3 color=black>'+str(p)+'</font></td>'
        output+='<td><font size=3 color=black>'+str(r)+'</font></td>'
        output+='<td><font size=3 color=black>'+str(f)+'</font></td>'
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'TrainData.html', context)      


def SpamDetectionAction(request):
    if request.method == 'POST':
        global rf, tfidf_vectorizer
        message = request.POST.get('t1', False)
        msg1 = message.strip().lower()
        clean = cleanPost(msg1)
        tfidf = tfidf_vectorizer.transform([clean]).toarray()
        predict = rf.predict(tfidf)
        predict = predict[0]
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Email Message</font></th>'
        output+='<th><font size=3 color=black>Detection Result</font></th></tr>'
        if predict == 0:
            output+='<tr><td><font size=3 color=black>'+message+'</font></td><td><font size=3 color=black>HAM</td></tr>'
        if predict == 1:
            output+='<tr><td><font size=3 color=black>'+message+'</font></td><td><font size=3 color=black>SPAM</td></tr>'
        context= {'data':output}
        return render(request, 'ViewResult.html', context)       
        



            
