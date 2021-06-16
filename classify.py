import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def label():
    fake_data = pd.read_csv("pol-fake.csv") #read from csv file
    fake_data['isreal'] = 0 #add one column to show the data is real
    fake_data.to_csv("fake.csv",index=False)

    real_data = pd.read_csv("pol-real.csv") #read from csv file
    real_data['isreal'] = 1 #add one column to show the data is fake
    real_data.to_csv("real.csv",index=False)


def merge():
    first = 0
    realfile = open("real.csv", "a")
    fakefile = open("fake.csv", "r")
    for line in fakefile: #write all the lines of fake data in real data
        if(first != 0): #except the first line bcz it includes lables
            realfile.write(line)
        first = first + 1
    realfile.close()
    fakefile.close()


def shuffle():
    featureData = []
    targetData = []
    labels = (pd.read_csv("real.csv", header=None).values)[0] #labels are in the first row
    allData = (pd.read_csv("real.csv")).values.tolist() #list all rows
    random.shuffle(allData) #shuffle all rows
    for i in range(len(allData)): #seperate target and feature
        featureData.append(allData[i][:-1]) #feature all are the columns except the last
        targetData.append(allData[i][-1]) #target is the "isReal" column

    return featureData, targetData, labels

label()
merge()
x, y, labels = shuffle()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


###########################################
#2
def plotgini(clf,labels):
    fig = plt.figure(figsize=(5, 5))
    a = tree.plot_tree(clf,feature_names = labels,class_names=['fake', 'real'],filled=True)

    plt.show()
    fig.savefig("ginitree.png")

def q2():
    giniclf = DecisionTreeClassifier(criterion = 'gini')
    giniclf.fit(x_train, y_train)
    y_pred = giniclf.predict(x_test)
    print("accuracy: ",giniclf.score(x_test, y_test))
    print("confusion matrix: ", confusion_matrix(y_test, y_pred))
    print("precision score: ", precision_score(y_test, y_pred))
    print("recall score: ", recall_score(y_test, y_pred))
    print("f1 score: ", f1_score(y_test, y_pred))

    plotgini(giniclf,labels)


###########################################
#3
def plotgain(clf,labels):
    fig = plt.figure(figsize=(5, 5))
    a = tree.plot_tree(clf,feature_names = labels,class_names=['fake', 'real'],filled=True)

    plt.show()
    fig.savefig("gaintree.png")

def q3():
    giniclf = DecisionTreeClassifier(criterion = 'entropy')
    giniclf.fit(x_train, y_train)
    y_pred = giniclf.predict(x_test)
    print("accuracy: ",giniclf.score(x_test, y_test))
    print("confusion matrix: ", confusion_matrix(y_test, y_pred))
    print("precision score: ", precision_score(y_test, y_pred))
    print("recall score: ", recall_score(y_test, y_pred))
    print("f1 score: ", f1_score(y_test, y_pred))

    plotgain(giniclf,labels)

###########################################
#4
def crossValidationDepths(x, y, tree_depths, cv=10, scoring='accuracy'):
    scores_mean = []
    for depth in tree_depths:
        tree_model = DecisionTreeClassifier(max_depth=depth)
        scores = cross_val_score(tree_model, x, y, cv=cv, scoring=scoring)
        scores_mean.append(scores.mean())
    scores_mean = np.array(scores_mean)
    return scores_mean

def q4():
    tree_depths = range(5, 21)
    scores_mean= crossValidationDepths(x_train, y_train,tree_depths)
    mymax = 0
    index = 0
    for i in range(len(scores_mean)):
        var = mymax
        mymax = max(scores_mean[i],mymax)
        if(var != mymax):
            index = i
        print("Depth " + str(i+5) + " score : " + str(scores_mean[i]))

    print("The max is in depth : " + str(index+5))

######################################################
#6
def q6():
    model = RandomForestClassifier(criterion="gini")
    model.fit(x_train,y_train)
    print("RandomForest accuracy " + str(model.score(x_test,y_test)))

q2()
q3()
q4()
q6()