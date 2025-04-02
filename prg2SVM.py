#!/usr/bin/env python
#python 3.7 32 bit 
# coding: utf-8
#python -m pip install datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import timedelta
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv('crop_recommendation.csv')
df = pd.DataFrame(data = dataset)
#df['classfactor']=0

#print(df.columns.values)
#df=df.iloc[:, [0,10000]] #13 is date column 
#pd.DataFrame(np.c_[X['Item_Weight'], X['Item_Visibility'], X=df_Sales
#df['classfactor']=df['label']
#df=df.drop(['label'],axis=1)

#----------
# Calculate the average rainfall for each label
average_rainfall = df.groupby('label')['rainfall'].mean()

# Plot the average rainfall as a bar chart
average_rainfall.plot(kind='bar', color='skyblue')
plt.title('Average Rainfall by Crop Type')
plt.xlabel('Crop Type')
plt.ylabel('Average Rainfall')
plt.show()
#----------
# Calculate the average temperature for each label
average_rainfall = df.groupby('label')['temperature'].mean()

# Plot the average rainfall as a bar chart
average_rainfall.plot(kind='bar', color='skyblue')
plt.title('Average Temperature by Crop Type')
plt.xlabel('Crop Type')
plt.ylabel('Average Temperature')
plt.show()

#----------
# Calculate the average humidity for each label
average_rainfall = df.groupby('label')['humidity'].mean()

# Plot the average rainfall as a bar chart
average_rainfall.plot(kind='bar', color='skyblue')
plt.title('Average Humidity by Crop Type')
plt.xlabel('Crop Type')
plt.ylabel('Average Humidity')
plt.show()

df['label'] = pd.factorize(df['label'])[0] + 1 
df = df.iloc[: , 1:]
rowscount =len(df)

"""
# Get the counts of each unique value in the 'label' column
label_counts = df['label'].value_counts()

# Plot the counts as a bar chart
label_counts.plot(kind='bar', color='skyblue')
plt.title('Count of Records by Crop Name')
plt.xlabel('Crop Name')
plt.ylabel('Count')
plt.show()
"""
"""
for i in range(0,rowscount):
   if (df.at[i,'Production']>=35650000):
       df.at[i,'classfactor']=1
   else:
       df.at[i,'classfactor']=2
"""

""" 
for  i in range(1,rowscount):
 if df.at[i,'classfactor']<=20:
   df.at[i, 'classfactor'] = 1
 else:
   df.at[i,'classfactor']=2
"""
df_Sales =df
print(df.columns.values)

X=df_Sales
print(X.head())
print(X.columns.values)

print(X.shape)
print(X.columns)
print(X['classfactor'])
print(X['classfactor'].value_counts())
# As we can see, we have 212 - Malignant, and 357 - Benign
#  Let's visulaize our counts
# In[9]:
#sns.countplot(X['classfactor'], label = "Count") 


# # Let's check the correlation between our features 

# In[10]:


#plt.figure(figsize=(20,12)) 
#sns.heatmap(df_Sales.corr(), annot=True) 



#X = X.drop(['classfactor'], axis = 1) # We drop our "target" feature and use all the remaining features in our dataframe to train the model.
#print(X.head())


# In[12]:


y = X['classfactor']
X = X.drop(['classfactor'], axis = 1)
print(y.head())


from sklearn.model_selection import train_test_split


# Let's split our data using 80% for training and the remaining 20% for testing.

# In[14]:

indices =range(len(X))
X_train, X_test, y_train, y_test,tr,te = train_test_split(X, y, indices,test_size = 0.2, random_state = 20)


# Let now check the size our training and testing data.

# In[15]:

#f=open('FlaskDeployedApp/templates/outputsvm.html',"w")
#f.write('<html><head><title>Output</title>');
#s="<link rel='stylesheet' href='css/bootstrap.min.css'><script src='css/jquery.min.js'></script>  <script src='css/bootstrap.min.js'></script><link rel='stylesheet' href='css/all.css'>"
#f.write(s)
#f.write("</head><body>")

strcont="<center><h2>TRAINING/TEST DATASET SIZE</h2></center><font color='red' size='4' face='Tahoma'>"
#f.write("<center><h2>TRAINING/TEST DATASET SIZE</h2></center><font color='red' size='4' face='Tahoma'>")

print ('The size of our training "X" (input features) is', X_train.shape)
#f.write('The size of our training "X" (input features) is' + str(X_train.shape))
strcont=strcont+'The size of our training "X" (input features) is' + str(X_train.shape)
#f.write("<br/>")
strcont=strcont+"<br/>"

print ('\n')
print ('The size of our testing "X" (input features) is', X_test.shape)
#f.write('The size of our testing "X" (input features) is' + str(X_test.shape))
strcont=strcont+'The size of our testing "X" (input features) is' + str(X_test.shape)
#f.write("<br/>")
strcont=strcont+"<br/>"
print ('\n')
print ('The size of our training "y" (output feature) is', y_train.shape)
#f.write('The size of our training "y" (output feature) is' + str( y_train.shape))
strcont=strcont+'The size of our training "y" (output feature) is' + str( y_train.shape)
#f.write("<br/>")
strcont=strcont+"<br/>"
print ('\n')
print ('The size of our testing "y" (output features) is', y_test.shape)
#f.write('The size of our testing "y" (output features) is'+  str(y_test.shape))
strcont=strcont+'The size of our testing "y" (output features) is'+  str(y_test.shape)
#f.write("<br/><br/></font>")
strcont=strcont+"<br/>"


# # Import Support Vector Machine (SVM) Model 

# In[16]:


from sklearn.svm import SVC


# In[17]:


svc_model = SVC()


# # Now, let's train our SVM model with our "training" dataset.

# In[18]:


svc_model.fit(X_train, y_train)


# # Let's use our trained model to make a prediction using our testing data

# In[19]:


y_predict = svc_model.predict(X_test)
print(y_predict)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[21]:


cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_Low', 'is_High'],
                         columns=['predicted_Low','predicted_High'])
print(confusion)

cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_Low', 'is_High'],
                         columns=['predicted_Low','predicted_High'])
ac = accuracy_score(y_test,y_predict)
leng=len(te)
print("------");

print('Accuracy:')
print(str(round(ac,3)))

tp, fp, fn, tn = confusion_matrix(y_test, y_predict).ravel()
#Accuracy = (TP+TN)/(TP+FP+FN+TN)

print(tp)
print(tn)
print(fp)
print(fn)
precision_score_val = tp / (tp + fp)
#Recall = (TP/(TP+FN))
recall_score_val = tp / (tp + fn)
print("Precision Score:")
print(precision_score_val)
print("Recall Score:")
print(recall_score_val)

#F1 = (2 * P * R)/(P+R)
#f1 = f1_score(y_test, y_predict)#, average="binary")
f1= (2*precision_score_val * recall_score_val) / (precision_score_val+recall_score_val)
print("F1 Score:")
print(f1)


#---------------
#---


#exit()
#f.write(   'Accuracy Score<br/>')
strcont=strcont+   'Accuracy Score<br/>'
#f.write(str(round(ac,3)) + "<br/>")
strcont=strcont+str(round(ac,3)) + "<br/>"

print('Confusion Matrix')
print(confusion_matrix(y_test, y_predict))
#f.write('Confusion Matrix [SVM]<br/>')
strcont=strcont+'Confusion Matrix [SVM]<br/>'
#f.write('----------------------<br/>')
strcont=strcont+'----------------------<br/>'





Mat=confusion_matrix(y_test, y_predict)
X=Mat.flatten(order='C')
Y=X#[ ['Item_Outlet_Sales']]
#values=X.groupby(['Item_Outlet_Sales'])['Item_Outlet_Sales'].count() #.plot.bar(figsize=(8, 6));

#f.write(str(confusion_matrix(y_test, y_predict)) + "<br/>")
strcont=strcont+str(confusion_matrix(y_test, y_predict)) + "<br/>"
strcont=strcont+"<center><img src='{{output_image}}' width='1000' height='600'/><br/>"
#f.write("<center><img src='{{output_image}}' width='1000' height='600'/><br/>")
strcont=strcont+"<center><table class='table table-bordered table-striped table-hover' border='1' style='border-radius:5px;width:50%'><tr><th>Index</th><th>Category</th></tr>"

#f.write("<center><table class='table table-bordered table-striped table-hover' border='1' style='border-radius:5px;width:50%'><tr><th>Index</th><th>Category</th></tr>")
for i in range(0,leng):
   s= "<tr><td>" + str(te[i]) + "</td><td>"+  str(y_predict[i]) + "</td>"
   #f.write(s)
   strcont=strcont+s

#f.write("</table>")
strcont=strcont+"</table>"

#str2=str2+ "</div></div></div><script>const actualBtn = document.getElementById('actual-btn');const fileChosen = document.getElementById('file-chosen');    actualBtn.addEventListener('change', function () {        fileChosen.textContent = this.files[0].name    })</script>{% endblock body %}"
#f.write(str2)
#f.write("</center></body></html>")
#f.close()

"""
f1=open("tmp.html","r", errors='replace')
contents=f1.readlines()
f1.close()

f1=open("FlaskDeployedApp/templates/outputsvm.html","w", errors='replace')
strcont=strcont+ "<img src='{{output_image}}' width='1000' height='600'/>"
strtmp=''
for x1 in contents:
  strtmp =strtmp + x1
contents=str(strtmp).replace("#mydata#",strcont)
f1.write (contents)
f1.close()
"""

values=X#.groupby(  pd.cut(X['Item_Outlet_Sales'],[500,1000,1500,2000,2500,3000,3500]))['Item_Outlet_Sales'].count()
#print(values)
#print(values[0])
#values.plot.bar(figsize=(8, 6))

xaxistitles=['True Positive','True Negative','False Positive','False Negative']#['HH','HM','HS','MH','MM','MS','SH','SM','SS']#['True Positive','True Negative','False 
#xaxistitles=['True Positive','True Negative','False Negative','False Positive']#[ '500-1000','1001-1500','1501-2000','2001-2500','2501-3000','3001-3500']
#for i in range(0,len(values)):
  #xaxistitles.append(str(dominantcloudservers[i]))
#plt.bar(xaxistitles,values)
#plt.scatter(DCCap, df['y'], color=colors, alpha=0.5, edgecolor='k')
#for idx, centroid in enumerate(centroids):
#    plt.scatter(*centroid, color=colmap[idx+1])
#plt.xlim(0, 7)
#plt.ylim(0, 40)
#plt.show()

def autolabel(rects,ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

ind = np.arange(4)#9)  # the x locations for the groups
width = 0.35   
men_std = (2, 2, 2, 2)#,2,2,2,2,2)
fig, ax = plt.subplots()
rects1 = ax.bar(ind, values, width, color='r', yerr=men_std)
 #women_means = (25, 32, 34, 20, 25,25, 32, 34, 20, 25)
 #women_std = (3, 5, 2, 3, 3,2,2,2,2,2)
 #rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)
 # add some text for labels, title and axes ticks
ax.set_xlabel('CATEGORY')
ax.set_ylabel('RECORDS COUNT')
ax.set_title('CONFUSION MATRIX VALUES')
ax.set_xticks((ind + width / 2 ) )
ax.set_xticklabels(xaxistitles) #  ('1', '2', '3', '4', '5','6','7','8','9','10'))
 #ax.legend((rects1[0], rects2[0]), ('Men','Women'))
ax.legend(['Value'])
autolabel(rects1,ax)
 #autolabel(rects2)
#plt.savefig('FlaskDeployedApp/static/outputimages/prgsvm.png')
plt.show()
"""  for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
"""
#exit()
#End of module SVM
# In[32]:
"""
import webbrowser
import os
os.chdir('FlaskDeployedApp/templates')
filename='file:///'+os.getcwd()+'/' + 'outputsvm.html'
webbrowser.open_new_tab(filename)
"""
exit()
sns.heatmap(confusion,annot=True,fmt="d")
# In[33]:

"""
print(classification_report(y_test,y_predict))



param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 


# In[35]:


from sklearn.model_selection import GridSearchCV


# In[36]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


# In[37]:


grid.fit(X_train_scaled,y_train)


# **Let's print out the "grid" with the best parameter**

# In[38]:


print (grid.best_params_)
print ('\n')
print (grid.best_estimator_)


# **As we can see, the best parameters are "C" = 100, "gamma" = "0.01" and "kernel" = 'rbf'**

# In[39]:


grid_predictions = grid.predict(X_test_scaled)


# In[40]:


cm = np.array(confusion_matrix(y_test, grid_predictions, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_Low', 'is_High'],
                         columns=['predicted_Low','predicted_High'])
confusion


# In[41]:


sns.heatmap(confusion, annot=True)


# In[42]:


print(classification_report(y_test,grid_predictions))


# **As we can see, our best model is SVM with Normalized data, followed by our Gridsearch model**
"""
