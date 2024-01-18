#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Detection

# ## Import necessary Libraries.

# In[40]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# # Load the Dataset

# In[41]:


data=pd.read_csv('C:/Users/Effat/Desktop/SystemTronInternship/Breast_Cancer_Detection/data.csv')


# # Top 5 rows

# In[42]:


data.head()


# # Shape of the dataset

# In[43]:


print("Shape of dataset:",data.shape)


# # EDA - Exploratory Data Analysis

# In[44]:


data=data.drop('Unnamed: 32',axis=1)


# In[45]:


data.head()


# ### Dataset Information

# In[46]:


data.info()


# ### Statistics of data

# In[47]:


data.describe()


# ### Checking for Null Values

# In[48]:


data.isnull().sum()


# ### Target variable ( Diagnosis )count

# In[49]:


sns.countplot(x ='diagnosis', data=data)


# ### Correlation of variables

# In[50]:


plt.figure(figsize=(24,13))

d = data
corr = d.corr()
sns.heatmap(corr, annot=True, fmt=".2f");


# In[51]:


fig =px.box(data,y=data.radius_mean,points='all',color=data.diagnosis)
fig.show()


# # Model Building and Evaluation

# ### 1. Training Data Extraction

# In[52]:


X=data.drop(['id','diagnosis'], axis=1)
y=data.diagnosis


# In[53]:


X.head()


# ### 2. Splitting of Data

# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### 3. Data Standardization 

# In[56]:


from sklearn.preprocessing import StandardScaler


# In[57]:


# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[58]:


# Reshape data for CNN
X_train = X_train.reshape((-1, X.shape[1], 1, 1))
X_test = X_test.reshape((-1, X.shape[1], 1, 1))


# ### 4. Label Encoding

# In[59]:


from sklearn.preprocessing import LabelEncoder

# Encode the target labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# ### 5. CNN Model Building

# In[60]:


import tensorflow as tf
from tensorflow.keras import layers, models


# In[61]:


# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train_encoded)
y_test = tf.keras.utils.to_categorical(y_test_encoded)


# In[62]:


# Build the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 1), activation='relu', input_shape=(X.shape[1], 1, 1)))
model.add(layers.MaxPooling2D((2, 1)))
model.add(layers.Conv2D(64, (3, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))  # 2 output classes for binary classification


# In[63]:


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[64]:


# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1)


# In[65]:


# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[75]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(['training data', 'validation data'], loc = 'upper right')


# ### 6. Evaluation of the model

# In[66]:


print(X_test.shape)
print(X_test[0])


# In[67]:


# Make predictions on the test set
predictions = model.predict(X_test)


# In[68]:


print(predictions.shape)
print(predictions[0])


# In[69]:


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')


# ### We got 96.5% Accuarcy

# In[70]:


#  argmax function

my_list = [0.25, 0.56]

index_of_max_value = np.argmax(my_list)
print(my_list)
print(index_of_max_value)


# In[71]:


Y_pred_labels = [1 if np.argmax(prob) == 1 else 0 for prob in predictions]
print(Y_pred_labels)


# ## Breast Cancer Detection System

# In[72]:


def predict_breast_cancer(input_data, model, scaler):
    try:
        # Change the input_data to a numpy array
        input_data_as_numpy_array = np.asarray(input_data)

        # Reshape the numpy array for prediction
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Standardize the input data
        input_data_std = scaler.transform(input_data_reshaped)

        # Reshape the standardized data for CNN
        input_data_std_reshaped = input_data_std.reshape(1, input_data_std.shape[1], 1, 1)

        # Make predictions using the CNN model
        prediction = model.predict(input_data_std_reshaped)

        # Convert predictions to class labels
        predicted_class = np.argmax(prediction)

        if predicted_class == 1:
            print('The tumor is Malignant')
        else:
            print('The tumor is Benign')

    except Exception as e:
        print(f"An error occurred: {e}")



# In[73]:


# Example usage:
input_data2 = np.array([17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,
                        153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,
                        0.6656,0.7119,0.2654,0.4601,0.1189 ])

predict_breast_cancer(input_data2, model, scaler)


# In[74]:


# Example usage
input_data = np.array([11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657, 0.01115, 0.1495, 0.05888, 
                       0.4062, 1.21, 2.635, 28.47, 0.005857, 0.009758, 0.01168, 0.007445, 0.02406, 
                       0.001769, 12.98, 25.72, 82.98, 516.5, 0.1085, 0.08615, 0.05523, 0.03715, 0.2433, 0.06563])

predict_breast_cancer(input_data, model, scaler)


# ## Thank You :)
