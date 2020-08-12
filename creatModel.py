# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 23:07:09 2020

@author: Harsh Parashar
"""
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn import preprocessing
from sklearn import model_selection
import classifiers
from Neural_Network import ANN
from keras.models import load_model


#Importing the dataset
dataset = sklearn.datasets.load_breast_cancer()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#Data Preprocessing
#Standardization
sc = preprocessing.StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Classifiers
#Linear Regression
cn_LinR, accuracy_LinR, accuracies_LinR = classifiers.Linear_Regression(X_train, y_train, X_test, y_test)
print("The accuracy of Linear Regression is: {:.2f} %".format(accuracies_LinR.mean()*100))
print("Standard Deviation of Linear Regression is {:.2f} %".format(accuracies_LinR.std()*100))

#Logostic Regresion
cn_LR, accuracy_LR, accuracies_LR = classifiers.Logistic_Regression(X_train, y_train, X_test, y_test)
print("The accuracy of Logistic Regression is: {:.2f} %".format(accuracies_LR.mean()*100))
print("Standard Deviation of Logistic Regression is {:.2f} %".format(accuracies_LR.std()*100))

#KNN
cn_KNN, accuracy_KNN, accuracies_KNN = classifiers.KNN(X_train, y_train, X_test, y_test)
print("The accuracy of KNN is: {:.2f} %".format(accuracies_KNN.mean()*100))
print("Standard Deviation of KNN is {:.2f} %".format(accuracies_KNN.std()*100))

#SVM
cn_SVM, accuracy_SVM, accuracies_SVM = classifiers.SVM(X_train, y_train, X_test, y_test)
print("The accuracy of SVM is: {:.2f} %".format(accuracies_SVM.mean()*100))
print("Standard Deviation of SVM is {:.2f} %".format(accuracies_SVM.std()*100))

#Naive Bayes
cn_GNB, accuracy_GNB, accuracies_GNB = classifiers.Naive_Bayes(X_train, y_train, X_test, y_test)
print("The accuracy of Naive Bayes is: {:.2f} %".format(accuracies_GNB.mean()*100))
print("Standard Deviation of Naive Bayes is {:.2f} %".format(accuracies_GNB.std()*100))

#Decision Tree Classification
cn_DTC, accuracy_DTC, accuracies_DTC = classifiers.Decision_Tree_Classifier(X_train, y_train, X_test, y_test)
print("The accuracy of Decision Tree Classifier is: {:.2f} %".format(accuracies_DTC.mean()*100))
print("Standard Deviation of Decision Tree Classifier is {:.2f} %".format(accuracies_DTC.std()*100))

#Random Forest Classification
cn_RFC, accuracy_RFC, accuracies_RFC = classifiers.Random_Forest_Classifier(X_train, y_train, X_test, y_test)
print("The accuracy of Random Forest Classifier is: {:.2f} %".format(accuracies_RFC.mean()*100))
print("Standard Deviation of Random Forest Classifier is {:.2f} %".format(accuracies_RFC.std()*100))

#Artificial Neural Network
accuracy_NN = ANN(X_train, y_train, X_test, y_test)
print("Accuracy of the neural Network is: "+str(accuracy_NN))

#XGBOOST
cn_XGB, accuracy_XGB, accuracies_XGB = classifiers.XGBoost(X_train, y_train, X_test, y_test)
print("The accuracy of XGBoost Classifier is: {:.2f} %".format(accuracies_XGB.mean()*100))
print("Standard Deviation of XGBoost Classifier is {:.2f} %".format(accuracies_XGB.std()*100))
 
#Accuracies
print("The accuracy of Linear Regression on test set is: {:.2f} %".format(accuracy_LinR*100))
print("The accuracy of RFC algorithm on test set is: {:.2f} %".format(accuracy_RFC*100))
print("The accuracy of DTC algorithm on test set is: {:.2f} %".format(accuracy_DTC*100))
print("The accuracy of Naive Bayes algorithm on test set is: {:.2f} %".format(accuracy_GNB*100))
print("The accuracy of SVM algorithm on test set is: {:.2f} %".format(accuracy_SVM*100))
print("The accuracy of KNN algorithm on test set is: {:.2f} %".format(accuracy_KNN*100))
print("The accuracy of Logistic Regression on test set is: {:.2f} %".format(accuracy_LR*100))
print("The accuracy of Artficial Neural Network on test set is: {:.2f} %".format(accuracy_NN[1]*100))


