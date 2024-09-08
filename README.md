# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## PPOBLEM STATEMENT

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## NEURAL NETWORK MODEL

![image](https://github.com/user-attachments/assets/087237d8-8891-485f-9559-5174ebe59fc5)

## DESIGN STEPS

### STEP 1:
Prepare and preprocess the customer data by cleaning, encoding categorical features, and splitting it into training and testing sets.
### STEP 2:
Build and train a neural network using TensorFlow/Keras to predict customer segments based on the preprocessed data.
### STEP 3:
Evaluate the model’s performance, save the trained model, and use it to predict customer segments for new data.

## PROGRAM
### Name: SASIDEVI.V
### Register Number: 212222230136
#### Importing Libraries
```
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,load_model
import pickle
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,OneHotEncoder,OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt
```
#### Loading and Exploring Data
```
df = pd.read_csv('/content/customers.csv')
df.columns
df.dtypes
df.shape
df.isnull().sum()
df_cleaned=df.dropna(axis=0)
df_cleaned.shape
df_cleaned.nunique()
df_cleaned=df_cleaned.drop(columns=['ID','Var_1'],axis=1)
```
#### Encoding Categorical Data
##### Ordinal Encoder
```
columns_to_encode = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score']
ordinal_encoder = OrdinalEncoder()
df_cleaned[columns_to_encode] = ordinal_encoder.fit_transform(df_cleaned[columns_to_encode])
df_cleaned.head()
df_cleaned.dtypes
```
##### Label Encoder
```
le = LabelEncoder()
df_cleaned['Segmentation'] = le.fit_transform(df_cleaned['Segmentation'])
df_cleaned.dtypes
```
#### Data Visualization
```
corr = df_cleaned.corr()
sns.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns,cmap="BuPu",annot= True)
sns.pairplot(df_cleaned)
sns.distplot(df_cleaned['Age'])
plt.figure(figsize=(7,5))
sns.boxplot(x='Family_Size',y='Age',data=df_cleaned)
plt.figure(figsize=(7,5))
sns.scatterplot(x='Family_Size',y='Spending_Score',data=df_cleaned)
plt.figure(figsize=(7,5))
sns.scatterplot(x='Family_Size',y='Age',data=df_cleaned)
```
#### Preparing Data for Model
```
X=df_cleaned[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values
y=df_cleaned[['Segmentation']].values
one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y)
y.shape
y = one_hot_enc.transform(y).toarray()
y.shape
X.shape
```
#### Splitting Data
```
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=50)
X_train[0]
X_train.shape
```
#### Scaling Age Feature 
```
scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:,2].reshape(-1,1))
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)
X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)
```
#### Building Model
```
early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

model = Sequential([
    Dense(6, activation='relu', input_shape=[8]),
    Dropout(0.3), 
    Dense(10, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='relu'),
    Dense(4, activation='softmax')
])
model.compile(optimizer='adam',loss='categorical_crossentropy' ,metrics=['accuracy'])
```
#### Training the Model
```
model.fit(x=X_train_scaled,y=y_train,
             epochs= 2000,
             batch_size= 32,
             validation_data=(X_test_scaled,y_test),
             callbacks=[early_stop] )
```
#### Evaluating the Model
```
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['loss','val_loss']].plot()
```
#### Predicting and Evaluating Test Data
```
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape
print(confusion_matrix(y_test_truevalue,x_test_predictions))
print(classification_report(y_test_truevalue,x_test_predictions))
```
#### Saving and Loading Model
```
model.save('customer_classification_model.h5')
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,df_cleaned,df_cleaned,scaler_age,ordinal_encoder,one_hot_enc,le], fh)
```
#### Making Predictions on New Data
```
model = load_model('customer_classification_model.h5')
with open('customer_data.pickle', 'rb') as fh:
   [X_train_scaled,y_train,X_test_scaled,y_test,df_cleaned,df_cleaned,scaler_age,ordinal_encoder,one_hot_enc,le]=pickle.load(fh)

x_single_prediction = np.argmax(model.predict(X_test_scaled[1:2,:]), axis=1)
print(x_single_prediction)

print(le.inverse_transform(x_single_prediction))
```

## Dataset Information

![image](https://github.com/user-attachments/assets/a48133e1-b853-46f9-a05b-e62d78448e34)

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/1a4c8239-7347-4ccc-b295-abe11dac01f4)

### Classification Report

![image](https://github.com/user-attachments/assets/438ea0d4-84e5-47da-a3dd-b7fbe61469f8)

### Confusion Matrix

![image](https://github.com/user-attachments/assets/b8e2415b-4428-49e4-83b2-bb419b40ac11)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/372d0bd6-e1e7-4a41-831e-bada8b3a34f7)

## RESULT
A neural network classification model is developed for the given dataset. 
