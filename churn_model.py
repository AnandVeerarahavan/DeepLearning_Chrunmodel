# Import all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Import tensorflow libraries
import tensorflow 
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense

# Read the csv data
data = pd.read_csv('C:/Users/Anand/Downloads/Churn_Modelling.csv')
print("====================================================")
print()
print("Top five columns")
print(data.head(5))

# Data infromation
print("====================================================")
print()
print("Infromation about the dataset")
print(data.info())

# Seperate features and target from the data
X = data.iloc[:,3:13]
y= data.iloc[:,13]

# Operations on data (Scaling and feature extraction)
geography =pd.get_dummies(X['Geography'], drop_first=True)
Gender =pd.get_dummies(X['Gender'], drop_first=True)

X = pd.concat([X, geography, Gender], axis =1)
X.drop(columns =['Geography','Gender'], axis =1,inplace = True)
print("====================================================")
print()
print("Revised top 5 rows")
print(X.head())

# Splitting the training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2,random_state = 101)

# Feature Scaling
sc= StandardScaler()
X_train =sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Process to follow :
# 1. Create an architechture
# 2. Compilation
# 3. Fit

# Creating a architechture

model = Sequential()
model.add(Dense(300,activation='relu', kernel_initializer = 'he_normal'))# Hidden layer 1
model.add(Dense(150,activation='sigmoid',kernel_initializer = 'glorot_uniform'))# Hidden layer 2
model.add(Dense(75,activation='relu',kernel_initializer = 'he_normal'))# Hidden layer 3
model.add(Dense(37,activation='sigmoid',kernel_initializer = 'glorot_uniform'))# Hidden layer 4
model.add(Dense(1,activation='sigmoid',kernel_initializer = 'glorot_uniform'))# Output Layer


# Compilation
# Stochiastic Gradient descent
model.compile(optimizer='SGD', loss ='binary_crossentropy', metrics =['accuracy'])

# Fitting the model
history = model.fit(X_train,y_train, epochs = 50, batch_size =32, validation_split=0.2)

# Printing dictionary keys used in history
print("====================================================")
print()
print("Dictionary keys available")
print(history.history.keys())

# Plotting the graph - Accuracy vs Epoch
print("====================================================")
print()
print("Plotting the graph - Accuracy vs Epoch")
plot1 =plt.plot(history.history['accuracy'])
plot1 =plt.plot(history.history['val_accuracy'])
plot1 =plt.ylabel('Accuracy')
plot1 =plt.xlabel('Epoch')
plot1 =plt.legend(['train' , 'test'])
plt.show()

# Plotting the graph - Model loss vs Epoch
print("====================================================")
print()
print("Plotting the graph - Model loss vs Epoch")
plot2 =plt.plot(history.history['loss'])
plot2 =plt.plot(history.history['val_loss'])
plot2 =plt.ylabel('Model loss')
plot2 =plt.xlabel('Epoch')
plot2 =plt.legend(['train' , 'test'])
plt.show()

# Prediction
y_pred = model.predict(X_test)
print("====================================================")
print()
print("Prediction output")
print(y_pred)
print()
print("====================================================")
print()
y_pred = (y_pred>0.5)
print("Prediction based on True or False")
print(y_pred)

#Accuracy
from sklearn.metrics import accuracy_score
print("====================================================")
print()
print("Accuracy score :  ", accuracy_score(y_pred, y_test))

#Saving the model
model.save('Deep_learning_churn_save_model.h5')

#Model summary
print("====================================================")
print()
print(model.summary())
print("====================================================")
print()