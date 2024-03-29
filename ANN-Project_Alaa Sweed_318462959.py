# USE THIS TO RUN THE CODE IN TERMINAL!!!
#FIRST RUN: pip install numpy pandas tensorflow scikit-learn
#SECOND RUN: python "ANN-Project_Alaa Sweed_318462959.py"

# # Artificial Neural Network Project - Alaa Sweed - 318462959

# ## 0) Importing the relevant libraries

import numpy as np
import pandas as pd
import tensorflow as tf

# ## 1) Data Preprocessing

# ### 1.1) Importing the dataset


# loading iris dataset from csv file format
# i provided dataset with low accuracy for testing (iris-data-with-lower-accuracy.csv) or use deafult one with  high accuracy (iris-data.csv)
irisDataSet = pd.read_csv('iris-data.csv')
# loading input values from dataset
X = irisDataSet.iloc[:, :-1].values
# loading output values from dataset
y = irisDataSet.iloc[:, -1].values

print(X)

print(y)


# ### 1.2) Encoding categorical data

# #### 1.2.1) There is no need to encode the input columns because everything in numeric

# #### 1.2.2) One-hot encoding the output column: class

from sklearn.preprocessing import OneHotEncoder
# we use one-hot encoding because it most efficent way to encode categorial labels from class to numeric that will be used by ann (to make it suitable for ANN models)
one_hot_encoder = OneHotEncoder()
y_encoded = one_hot_encoder.fit_transform(y.reshape(-1, 1)).toarray()

print(y_encoded)


# ### 1.3) Splitting the dataset into training and test sets

from sklearn.model_selection import train_test_split
# we split randomly the data: 80% of data is used for training and 20% for testing 
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# ### 1.4) Normalize the input data (X) to improve ANN performance (i prefered to use Standardization over normalization because it rescales data for any distribution, while normalization is for normally distributed data. so overall Standardization is better)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## 2) Building the ANN

# ### 2.1) Initializing the ANN with the Input layer

# in input layer: input_shape must match number of inputs columns 
ann = tf.keras.models.Sequential([tf.keras.Input(shape=(X_train.shape[1],))])


# ### 2.2) Adding the first hidden layer

# first hidden layer uses ReLU activation function
ann.add(tf.keras.layers.Dense(units=8, activation='relu'))


# ### 2.3) Adding the second hidden layer

# second hidden layer also uses ReLU activation function
ann.add(tf.keras.layers.Dense(units=8, activation='relu'))


# ### 2.4) Adding the output layer (we use softmax for categorial output)

# output layer uses Softmax activation function for multi-class classification(number of units matches number of classes)
ann.add(tf.keras.layers.Dense(units=3, activation='softmax'))


# ## 3) Training the ANN

# ### 3.1) Compiling the ANN

# we use "adam" optimizer and "categorical_crossentropy" loss function for multi-class classification
ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# ### 3.2) Training the ANN on the Training set

ann.fit(X_train, y_train, epochs=100, batch_size=8)


# ## 4) Making the predictions and evaluating the model

# ### 4.1) Predicting the Test set results

# to solve an warning i changed X_test to TensorFlow tensor then passing it to ANN's prediction function
X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32) 
# making predictions 
y_pred_probs = ann.predict(X_test_tensor)
# converting predicted probabilities to class labels
y_pred = np.argmax(y_pred_probs, axis=1)
# extracting actual class labels
y_actual_values = np.argmax(y_test, axis=1)

print(y_pred)
print(y_actual_values)


# ### 4.2) Evaluating the model by Making the Confusion Matrix and calculate statistical metrics

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
cm = confusion_matrix(y_actual_values, y_pred)
print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(y_actual_values, y_pred)
precision = precision_score(y_actual_values, y_pred, average='macro')
recall = recall_score(y_actual_values, y_pred, average='macro')
f1 = f1_score(y_actual_values, y_pred, average='macro')

print("Statistical Metrics:")
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')