import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

#Classifiers
#Linear Regression
def Linear_Regression(X_train, y_train, X_test, y_test):
    classifier_linearRegression = LinearRegression()
    classifier_linearRegression.fit(X_train, y_train)
    predictor_linearRegression = np.array(classifier_linearRegression.predict(X_test), dtype='int')
    cn_LinR = confusion_matrix(y_test, predictor_linearRegression)
    accuracy_LinR = accuracy_score(y_test, predictor_linearRegression)
    accuracies_LinR = cross_val_score(estimator=classifier_linearRegression, X=X_train, y=y_train, cv=10)
    
    return cn_LinR, accuracy_LinR, accuracies_LinR 

#Logistic Regression
def Logistic_Regression(X_train, y_train, X_test, y_test):
    classifier_logisticRegression = LogisticRegression()
    classifier_logisticRegression.fit(X_train, y_train)
    predictor_logisticRegression = np.array(classifier_logisticRegression.predict(X_test), dtype='int')
    cn_LR = confusion_matrix(y_test, predictor_logisticRegression)
    accuracy_LR = accuracy_score(y_test, predictor_logisticRegression)
    accuracies_LR = cross_val_score(estimator=classifier_logisticRegression, X=X_train, y=y_train, cv=10)
    
    return cn_LR, accuracy_LR, accuracies_LR

#KNN
def KNN(X_train, y_train, X_test, y_test):
    classifier_KNeighbors = KNeighborsClassifier(n_neighbors=5)
    classifier_KNeighbors.fit(X_train, y_train)
    predictor_KNeighbors = np.array(classifier_KNeighbors.predict(X_test), dtype='int')
    cn_KNN = confusion_matrix(y_test, predictor_KNeighbors)
    accuracy_KNN = accuracy_score(y_test, predictor_KNeighbors)
    accuracies_KNN = cross_val_score(estimator=classifier_KNeighbors, X=X_train, y=y_train, cv=10)
    
    
    return cn_KNN, accuracy_KNN, accuracies_KNN

#SVM
def SVM(X_train, y_train, X_test, y_test):
    classifier_SVM = SVC(kernel='rbf')
    classifier_SVM = classifier_SVM.fit(X_train, y_train)
    predictor_SVM = classifier_SVM.predict(X_test)
    cn_SVM = confusion_matrix(y_test, predictor_SVM)
    accuracy_SVM = accuracy_score(y_test, predictor_SVM)
    accuracies_SVM = cross_val_score(estimator=classifier_SVM, X=X_train, y=y_train, cv=10)
    
    return cn_SVM, accuracy_SVM, accuracies_SVM

#Naive Bayes
def Naive_Bayes(X_train, y_train, X_test, y_test):
    classifier_GNB = GaussianNB()
    classifier_GNB = classifier_GNB.fit(X_train, y_train)
    predictor_GNB = classifier_GNB.predict(X_test)
    cn_GNB = confusion_matrix(y_test, predictor_GNB)
    accuracy_GNB = accuracy_score(y_test, predictor_GNB)
    accuracies_GNB = cross_val_score(estimator=classifier_GNB, X=X_train, y=y_train, cv=10)
    
    return cn_GNB, accuracy_GNB, accuracies_GNB

#Decision Tree Classifier
def Decision_Tree_Classifier(X_train, y_train, X_test, y_test):
    classifier_DTC = DecisionTreeClassifier(criterion='gini')
    classifier_DTC.fit(X_train, y_train)
    predictor_DTC = np.array(classifier_DTC.predict(X_test), dtype='int')
    cn_DTC = confusion_matrix(y_test, predictor_DTC)
    accuracy_DTC = accuracy_score(y_test, predictor_DTC)
    accuracies_DTC = cross_val_score(estimator=classifier_DTC, X=X_train, y=y_train, cv=10)
    
    return cn_DTC, accuracy_DTC, accuracies_DTC

#Random Forest Classification
def Random_Forest_Classifier(X_train, y_train, X_test, y_test):
    classifier_RFC = RandomForestClassifier(n_estimators=10)
    classifier_RFC.fit(X_train, y_train)
    predictor_RFC = np.array(classifier_RFC.predict(X_test), dtype='int')
    cn_RFC = confusion_matrix(y_test, predictor_RFC)
    accuracy_RFC = accuracy_score(y_test, predictor_RFC)
    accuracies_RFC = cross_val_score(estimator=classifier_RFC, X=X_train, y=y_train, cv=10)
    
    return cn_RFC, accuracy_RFC, accuracies_RFC

#XGBoost Classification
def XGBoost(X_train, y_train, X_test, y_test):
    classifier_xgboost = XGBClassifier()
    classifier_xgboost.fit(X_train, y_train)
    predictor_XGB = np.array(classifier_xgboost.predict(X_test), dtype='int')
    cn_XGB = confusion_matrix(y_test, predictor_XGB)
    accuracy_XGB = accuracy_score(y_test, predictor_XGB)
    accuracies_XGB = cross_val_score(estimator=classifier_xgboost, X=X_train, y=y_train, cv=10)
    
    return cn_XGB, accuracy_XGB, accuracies_XGB



