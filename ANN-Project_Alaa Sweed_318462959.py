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
irisDataSet = pd.read_csv('iris-data.csv')
X = irisDataSet.iloc[:, 1:-1].values
y = irisDataSet.iloc[:, -1].values

print(X)
print(y)

# ### 1.2) Encoding categorical data

# #### 1.2.1) One-hot encoding the input columns: sepal length ,sepal width ,petal length ,petal width
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()
X_encoded = one_hot_encoder.fit_transform(X).toarray()

print(X_encoded)

# #### 1.2.2) One-hot encoding the output column: class
y_encoded = pd.get_dummies(y).values

print(y_encoded)

# ### 1.3) Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# ### 1.4) There is no need to use standardization or normalization in the dataset because all the input we encoded using one-hot encoded

# ## 2) Building the ANN

# ### 2.1) Initializing the ANN
ann = tf.keras.models.Sequential()

# ### 2.2) Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=8, activation='relu', input_shape=(X_train.shape[1],)))

# ### 2.3) Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=8, activation='relu'))

# ### 2.4) Adding the output layer (we use softmax for categoral output)
ann.add(tf.keras.layers.Dense(units=3, activation='softmax'))

# ## 3) Training the ANN

# ### 3.1) Compiling the ANN
ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ### 3.2) Training the ANN on the Training set
ann.fit(X_train, y_train, epochs=100, batch_size=8)

# ## 4) Making the predictions and evaluating the model

# ### 4.1) Predicting the Test set results (then converting predictions from one-hot encoded to class integers)
y_pred_probs = ann.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print(y_pred)
print(y_true)

# ### 4.2) Making the Confusion Matrix and calculate statistical metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print("Statistical Metrics:")
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
