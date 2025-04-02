import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import keras 
import os
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical 
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Dense, Dropout 
from keras.layers import Flatten, BatchNormalization


import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from sklearn.metrics import confusion_matrix
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


#from google.colab import files
#uploaded = files.upload()
train_df = pd.read_csv('crop_recommendation.csv')#LiverDatasetForCNN-train.csv')
#train_df=train_df.drop(['label'],axis=1)
# Convert categorical values in 'label' column to numerical codes
train_df['label'] = pd.factorize(train_df['label'])[0] + 1 

#uploaded = files.upload()
test_df = train_df
train_df =train_df.iloc[1:,:] #11001
test_df =test_df.iloc[1:,:]

#df = df.dropna() 
#colcnt=len()
#,'first_name','last_name','email','gender'
#train_df= train_df.drop(columns=['sno'])#, axis=1)
#test_df= test_df.drop(columns=['sno'])#, axis=1)
#train_df= train_df.drop(columns=['bmi'])#, axis=1)
#test_df= test_df.drop(columns=['bmi'])#, axis=1)
#train_df= train_df.drop(columns=['weight'])#, axis=1)
#test_df= test_df.drop(columns=['weight'])#, axis=1)
"""
for i in range(0,len(df.columns.values)): 
 
 if i!=len(df.columns.values)-1:
    df[df.columns.values[i]]= pd.to_numeric(df[df.columns.values[i]], errors='coerce')
""" 

print(train_df.head())


train_data = np.array(train_df.iloc[:, :-1])
test_data = np.array(test_df.iloc[:, :-1])


train_labels = to_categorical(train_df.iloc[:, 8]) #8])
test_labels = to_categorical(test_df.iloc[:, 8]) # 8])


rows, cols = 8, 1

train_data = train_data.reshape(train_data.shape[0], rows, cols, 1)
test_data = test_data.reshape(test_data.shape[0], rows, cols, 1)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

#train_data /= 255.0
#test_data /= 255.0
"""
files=os.listdir('./dataset')
for obj in os.listdir('./dataset'):
    print("Folder : " + obj)
"""
files=["Yes","No","Unknown"]
labels=files
#labels.append('Unknown')
print(len(labels))

NUM_CLASSES=3  #Yes,No,Unknown #len(labels) #6#10#9#len(files)
train_x, val_x, train_y, val_y = train_test_split(train_data, train_labels, test_size=0.25)

train_x =train_data
val_x= test_data
train_y = train_labels
val_y =test_labels
batch_size = 32# 256#32#256
epochs = 50  #10
input_shape = (rows, cols, 1)

def baseline_model():
    model = Sequential()
    """
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(0.25))
    """
    model.add(Conv2D(32, (4, 4), padding='same', activation='relu'))
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    model.add(Conv2D(64, (4, 4), padding='same', activation='relu'))
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    

      
   
    """
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    """
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


model = baseline_model()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

history = model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(val_x, val_y))

predictions= model.predict(test_data)


model.summary()

def create_trace(x,y,ylabel,color):
        trace = go.Scatter(
            x = x,y = y,
            name=ylabel,
            marker=dict(color=color),
            mode = "markers+lines",
            text=x
        )
        return trace


def plot_accuracy_and_loss(train_model):
    hist = train_model.history
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = list(range(1,len(acc)+1))

    trace_ta = create_trace(epochs,acc,"Training accuracy", "Green")
    trace_va = create_trace(epochs,val_acc,"Validation accuracy", "Red")
    trace_tl = create_trace(epochs,loss,"Training loss", "Blue")
    trace_vl = create_trace(epochs,val_loss,"Validation loss", "Magenta")

    fig = tools.make_subplots(rows=1,cols=2, subplot_titles=('Training and validation accuracy',
                                                             'Training and validation loss'))
    fig.append_trace(trace_ta,1,1)
    fig.append_trace(trace_va,1,1)
    fig.append_trace(trace_tl,1,2)
    fig.append_trace(trace_vl,1,2)
    fig['layout']['xaxis'].update(title = 'Epoch')
    fig['layout']['xaxis2'].update(title = 'Epoch')
    fig['layout']['yaxis'].update(title = 'Accuracy', range=[0,1])
    fig['layout']['yaxis2'].update(title = 'Loss', range=[0,1])


    iplot(fig, filename='accuracy-loss')


predictions = model.predict(test_data)
y_pred_classes = np.argmax(predictions, axis=1)  # Convert probabilities to class labels
y_true = np.argmax(test_labels, axis=1)  # Assuming test_labels are one-hot encoded

# Step 2: Calculate the Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Step 3: Extract TP, FP, TN, FN from Confusion Matrix (for binary classification)
TP = cm[0, 0]  # True Positive: Correctly predicted positive
TN = cm[1, 0]  # True Negative: Incorrectly predicted negative
FP = cm[0, 1]  # False Positive: Incorrectly predicted positive
FN = cm[1, 1]  # False Negative: Correctly predicted negative

# Step 4: Calculate Precision, Recall, and F1 Score using TP, FP, TN, and FN

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Output the accuracy
print(f"Accuracy: {accuracy:.4f}")
# Output the Confusion Matrix and Scores
print(f"Confusion Matrix:\n{cm}")
print(f"True Positive (TP): {TP}")
print(f"False Positive (FP): {FP}")
print(f"True Negative (TN): {TN}")
print(f"False Negative (FN): {FN}")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

import matplotlib.pyplot as plt
labels = ['True Positive', 'False Positive', 'True Negative', 'False Negative']

# Values for the y-axis (corresponding to TP, FP, TN, FN)
values = [TP, FP, TN, FN]

# Create a bar plot
plt.bar(labels, values, color=['green', 'red', 'blue', 'orange'])

# Add labels and title
plt.xlabel('Confusion Matrix Components')
plt.ylabel('Values')
plt.title('Confusion Matrix Bar Plot')

# Display the plot
plt.show()


labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Values for the y-axis (corresponding to TP, FP, TN, FN)
values = [accuracy, precision, recall, f1_score]

# Create a bar plot
plt.bar(labels, values, color=['green', 'red', 'blue', 'orange'])

# Add labels and title

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Metrics Comparision')

# Display the plot
plt.show()

exit()
#---

"""
#plot_accuracy_and_loss(history)
y_pred=predictions
y_pred_classes = np.argmax(y_pred, axis=1)
y_test= test_data#val_y
y_true = np.argmax(y_test, axis=1)  # Assuming one-hot encoding for y_test

# Step 3: Generate classification report
print("Classification Report:\n", classification_report(y_true, y_pred_classes))

# Step 4: Alternatively, calculate precision, recall, and F1-score separately
precision = precision_score(y_true, y_pred_classes, average='weighted')  # 'weighted' for multi-class
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

# Display the calculated scores
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
"""

score = model.evaluate(val_x, val_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('./_models/my_liver_model.keras')

import torch
torch.save(model,"liver.pt")
#model.save_state_dict("mushroomdict.pt")
print("Model Saved")


"""


y =train_labels# traindata['classification']
indices =range(len(train_data))
X_train=train_data #, X_test, y_train, y_test,tr,te = train_test_split(traindata, y, indices,test_size = 0.25, random_state = 42)
X_test  = test_data
traindata=X_train
y_train =train_labels
trainlabel= y_train#traindata.iloc[1:,11]
testlabel=test_labels#  y_test
print(trainlabel)
model.fit(traindata, trainlabel)

# make predictions
expected = testlabel
#np.savetxt('classical/expected.txt', expected, fmt='%01d')
predicted = model.predict(X_test)
#proba = model.predict_proba(X_test)

#np.savetxt('classical/predictedlabelLR.txt', predicted, fmt='%01d')
#np.savetxt('classical/predictedprobaLR.txt', proba)
y_pred_classes = np.argmax(predicted, axis=1)
y_train1 = expected
y_pred = predicted
y_true = np.argmax(test_data, axis=1)
#accuracy = accuracy_score(y_true, y_pred_classes)
print(y_true)
print(y_pred_classes)
exit()
precision = precision_score(y_true, y_pred_classes, average='weighted')  # 'weighted' for multi-class
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

# Display the calculated scores
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
exit()
#-------
"""

#To load the model from .pt file use this code
""" 
model = baseline_model()
#model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model2=torch.load("braintumor.pt")
history2 = model2.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(val_x, val_y))

predictions2= model2.predict(test_data)
print(predictions2)
"""
#---------------END---------------------

#model = CNN.CNN(39)    
#https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A
#model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
#model.eval()

"""
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
target_names = ["Class {} ({}) :".format(i,labels[i]) for i in range(NUM_CLASSES)]
print(classification_report(val_y, predictions, target_names=target_names))
"""
"""# The confusion matrix(both numpy array type and pictoral representation)"""
"""
#labels = [ "T-shirt/top", "Trouser", "Pullover",  "Dress", "Coat",           "Sandal", "Shirt",  "Sneaker",  "Bag",  "Ankle Boot"]
cm = confusion_matrix(val_y, predictions)
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                colorbar=True,
                                show_absolute=False,
                                show_normed=True,
                                class_names=labels)
plt.show()
cm
"""