#-------------------------------------------------------------------------
# AUTHOR: Jose Pavon
# FILENAME: svm.py
# SPECIFICATION: Uses SVM to determine handwritten numbers
# FOR: CS 4210- Assignment #3
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here
accuracy = 0
for strict in c:
    for value in degree:
        for types in kernel:
           for shape in decision_function_shape:
                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                svmTest = svm.SVC(C = strict, degree=value, kernel=types, decision_function_shape=shape)

                #Fit SVM to the training data
                svmTest.fit(X_training, y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                #to make a prediction do: clf.predict([x_testSample])
                correct = 0
                for(x_testcase, y_testcase) in zip(X_test, y_test):
                    predicted = svmTest.predict([x_testcase])
                    if predicted == y_testcase:
                        correct += 1

                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                print(correct/len(X_test))
                if correct/len(X_test) > accuracy:
                    accuracy = correct/len(X_test)
                    print(f"Highest SVM accuracy so far: {round(accuracy, 4)}, Parameters: a={strict}, degree={value}, kernel = {types}, decision_function_shape = {shape}")
