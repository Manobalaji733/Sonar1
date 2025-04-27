# importing the  libraries
import numpy as np # for arrays
import pandas as pd # for several data processing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# data collection  and data processing 
#loading the dataset to the pandas Dataframe
sonar_data = pd.read_csv('C:/Users/91936/OneDrive/Desktop/ml/Copy of sonar data.csv',header=None)
sonar_data.head()

# number of rows and columns.
sonar_data.shape
print(sonar_data.shape)

# describe the statistical measure of the data.
#this gives the better understanding of the data.
sonar_data.describe()
print(sonar_data.describe())

#this gives the how many rock and mine are present
sonar_data[60].value_counts()
print(sonar_data[60].value_counts()) #here the "60" is the coloum index

# Mean for each and ever coloum
sonar_data.groupby(60).mean()
print(sonar_data.groupby(60).mean())

# Seperating data and lables as its a Supervised Machine Learning Project
X = sonar_data.drop(columns=60,axis=1)
Y = sonar_data[60] 
print(X,Y)

# Training and test data 
X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=4)
print(X.shape,X_train.shape,X_test.shape)

# Model training --> Logistic Regresion
model = LogisticRegression()

# Training the model with the training data
model.fit(X_train,Y_train)

# Model evulation , accuracy on training data
X_train_predection = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predection,Y_train )
print("Accuracy on training data:",training_data_accuracy )

# accuracy on test data
X_train_predection = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_train_predection ,Y_test )
print("Accuracy on testind data:",testing_data_accuracy )

#Making a predictive System
input_data = (0.0229,0.0369,0.004,0.0375,0.0455,0.1452,0.2211,0.1188,0.075,0.1631,0.2709,0.3358,0.4091,0.44,0.5485,0.7213,0.8137,0.9185,1,0.9418,0.9116,0.9349,0.7484,0.5146,0.4106,0.3443,0.6981,0.8713,0.9013,0.8014,0.438,0.1319,0.1709,0.2484,0.3044,0.2312,0.1338,0.2056,0.2474,0.279,0.161,0.0056,0.0351,0.1148,0.1331,0.0276,0.0763,0.0631,0.0309,0.024,0.0115,0.0064,0.0022,0.0122,0.0151,0.0056,0.0026,0.0029,0.0104,0.0163)
# Changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predecting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
predection = model.predict(input_data_reshaped)
print(predection)
if(predection[0]=="R"):
    print("The object is a rock")
else:
    print("The object is mine")