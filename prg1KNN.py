#!/usr/bin/env python
#python 3.7 32 bit 
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import timedelta

#get_ipython().run_line_magic('matplotlib', 'inline')

#Import Cancer data from the Sklearn library
# Dataset can also be found here (http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29)

#from sklearn.datasets import load_breast_cancer
#cancer = load_breast_cancer()
#dataset = pd.read_csv('BigMartSales1000Records.csv')
dataset = pd.read_csv('crop_recommendation.csv')
df = pd.DataFrame(data = dataset)
#X = X.filter(['Item_Weight', 'Item_Visibility', 'Outlet_Establishment_Year','Item_Outlet_Sales', 'classfactor'])
# In[2]:
#X
print(df.columns.values)
#df=df.iloc[:, [1,3,5,7,11,12]]
df = pd.DataFrame(data = dataset)
#print(df.columns.values)
#df=df.iloc[:, [0,10000]] #13 is date column 
#df_Sales =df#pd.DataFrame(np.c_[X['Item_Weight'], X['Item_Visibility'], X=df_Sales
#df['classfactor']=df['label']
#df = df.iloc[: , 1:]
#df['classfactor']=df['label']
df=df.drop(['label'],axis=1)
df = df.iloc[: , 1:]
rowscount =len(df)
"""
for  i in range(1,rowscount):
 if df.at[i,'classfactor']<=20:
   df.at[i, 'classfactor'] = 1
 else:
   df.at[i,'classfactor']=2
"""   
df_Sales =df
print(df.columns.values)

# As we can see above, not much can be done in the current form of the dataset. We need to view the data in a better format.

# # Let's view the data in a dataframe.

# In[3]:


#df_Sales = pd.DataFrame(np.c_[X['Item_Outlet_Sales'], X['classfactor']], columns = np.append(X['Item_Outlet_Sales'], ['classfactor']))
df_Sales =df#pd.DataFrame(np.c_[X['Item_Weight'], X['Item_Visibility'], X['Outlet_Establishment_Year'],X['Item_Outlet_Sales'], X['classfactor']])
#,
#columns = np.append(['Item_Weight','Item_Visibility','Outlet_Establishment_Year','Item_Outlet_Sales','classfactor'])
X=df_Sales
print(X.head())
print(X.columns.values)

# # Let's Explore Our Dataset

# In[4]:
print(X.shape)
# As we can see,we have 596 rows (Instances) and 31 columns(Features)
# In[5]:
print(X.columns)

# Above is the name of each columns in our dataframe.

# # The next step is to Visualize our data

# In[6]:


# Let's plot out just the first 5 variables (features)
#sns.pairplot(df_Sales)#, vars = ['Item Weight', 'Item Visibility', 'Outlet Establishment Year', 'Item Outlet Sales'] )


# The above plots shows the relationship between our features. But the only problem with them is that they do not show us which of the "dots" is Malignant and which is Benign. 
# 
# This issue will be addressed below by using "target" variable as the "hue" for the plots.

# In[7]:


# Let's plot out just the first 5 variables (features)
#sns.pairplot(df_Sales, hue = 'classfactor', vars = ['Item_Weight', 'Item_Fat_Content', 'Outlet_Establishment_Year','Item_Outlet_Sales']  )


# **Note:** 
#     
#   1.0 (Orange) = Benign (No Cancer)
#   
#   0.0 (Blue) = Malignant (Cancer)

# # How many Benign and Malignant do we have in our dataset?

# In[8]:

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
X_train, X_test, y_train, y_test,tr,te = train_test_split(X, y, indices,test_size = 0.1, random_state = 20)
# Let now check the size our training and testing data.
# In[15]:
#f=open('FlaskDeployedApp/templates/outputknn.html',"w")


"""
str2=""#<html>{% extends 'base.html' %}"#{% block pagetitle %}AI Engine{% endblock pagetitle%}"
#str2=str2+"<head><link rel='stylesheet' type='text/css' href='{{ url_for('static', filename='css/mystyle') }}' />"
str2=str2+"<head><link rel='stylesheet' type='text/css' href='../static/css/mystyle' />"
str2=str2+"    <script src='https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.js'>  "
str2=str2+"    </script>  "
str2=str2+"<script src='https://ajax.googleapis.com/ajax/libs/jqueryui/1.8.16/jquery-ui.js'>  "
str2=str2+"    </script>  "
str2=str2+"    <link href='http://ajax.googleapis.com/ajax/libs/jqueryui/1.8.16/themes/ui-lightness/jquery-ui.css' rel='stylesheet' type='text/css' />  "
#str2=str2+"{% block body %}"
str2=str2+""
str2=str2+"<style>.bg {background-image: url('images/background.jpeg');height: 100%;/*Center and scale the image nicely */background-position: center;background-repeat: no-repeat;background-size: cover;}.file-upload input[type='file'] {display: none;}body {background-color: #77a4f7;/* background: -webkit-linear-gradient(to right, #66a776, #043610);background: linear-gradient(to right, #5d9c6a, #059b0d); */height: 100vh;}.rounded-lg {border-radius: 1rem;}.custom-file-label.rounded-pill {            border-radius: 50rem;}.custom-file-label.rounded-pill::after {border-radius: 0 50rem 50rem 0;}label {/*background-color: rgb(2, 46, 15);*/background-color:#010f87;color: white;padding: 0.5rem;font-family: sans-serif;border-radius: 0.3rem;cursor: pointer;margin-top: 1rem;}#file-chosen {margin-left: 0.3rem;font-family: sans-serif;}.footer-basic {padding: 40px 0;background-color: #77a4f7;color: #f1f3f5;}"
str2=str2+".footer-basic ul {padding: 0;            list-style: none;text-align: center;font-size: 18px;line-height: 1.6;margin-bottom: 0;}.footer-basic li {padding:0 10px;}.footer-basic ul a {color: inherit;text-decoration: none;opacity: 0.8;}"
str2=str2+".footer-basic ul a:hover {opacity: 1;}.footer-basic .social {           text-align: center;            padding-bottom: 25px;        }"
str2=str2+".footer-basic .social>"
str2=str2+"a {font-size: 24px; width: 40px;            height: 40px;            line-height: 40px;            display: inline-block; text-align: center;            border-radius: 50%;border: 1px solid #ccc;            margin: 0 8px;            color: inherit;opacity: 0.75;        }.footer-basic .social>a:hover {opacity: 0.9;}"
str2=str2+".footer-basic .copyright {margin-top: 15px;text-align: center;font-size: 13px;color: #aaa;margin-bottom: 0;}</style>"

str2=str2+"</head><body><div>    <div class='container'><div class='row mb-5 text-center text-white'>            <div class='col-lg-10 mx-auto'>                <h1 class='display-4' style='padding-top: 2%;font-weight: 400;color: rgb(4, 54, 4);'><b>Convolutional Neural Network</b></h1>                <p class='lead' style='font-weight: 500;color: black;'>Let Convolutional Neural Network Will Help You To Predict Beverage Sales Price</p>            </div>        </div><!--		<img src='{{ output_image }}' alt='User Image'> -->	        <!-- End -->        <div class='row '>            <div class='col mx-auto'>                <div class='p-5' style='height: 95%;'><h5><b>We will learn how to apply deep convolutional networks for predicting sales prices in python. This model could be easily applied to the stock-price prediction problem. Deep CNNs have been quite popular in areas such as Image Processing, Computer Vision, etc. Recently, the research community has been showing a growing interest in using CNNs for time-series forecasting problems. This article will be useful for a wide range of readers including deep learning enthusiasts, finance professionals, academicians, and data-science hobbyists.                    </p>                </div>            </div>"

str2=str2+'''<script>if (window.matchMedia('(max-width: 767px)').matches){
        // The viewport is less than 768 pixels wide 
        document.write('This is a mobile device.');
        window.location.replace('/mobile-device');
    } else {
        // The viewport is at least 768 pixels wide 
        document.write('This is a tablet or desktop.'); 
    }
</script>'''
f.write(str2)
"""
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
#f.write("<br/>")
strcont=strcont+"<br/>"
print ('\n')
print ('The size of our testing "y" (output features) is', y_test.shape)
#f.write('The size of our testing "y" (output features) is'+  str(y_test.shape))
#f.write("<br/><br/></font>")
strcont=strcont+'The size of our testing "y" (output features) is'+  str(y_test.shape)+"<br/><br/></font>"

#f.write(   'Accuracy Score<br/>')


# # Import Support Vector Machine (SVM) Model 

from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#iris = datasets.load_iris() 
#X, y = iris.data[:, :], iris.target
indices =range(len(X))
Xtrain, Xtest, y_train, y_test,tr,te = train_test_split(X, y, indices,stratify = y, random_state = 0, train_size = 0.7)

scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(Xtrain, y_train)
y_pred = knn.predict(Xtest)
y_predict=y_pred
# In[16]:
leng=len(te)
print("------");

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
#f.write(   'Accuracy Score<br/>')
strcont=strcont+   'Accuracy Score<br/>'
#f.write(str(round(ac,3)) + "<br/>")
strcont=strcont+str(round(ac,3)) + "<br/>"

print('Confusion Matrix')
print(confusion_matrix(y_test, y_predict))
#f.write('Confusion Matrix [SVM]<br/>')
strcont=strcont+'Confusion Matrix [KNN]<br/>'
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

f1=open("FlaskDeployedApp/templates/outputknn.html","w", errors='replace')
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

#xaxistitles=['HH','HM','HS','MH','MM','MS','SH','SM','SS']#['True Positive','True Negative','False Negative','False Positive']#[ '500-1000','1001-1500','1501-2000','2001-2500','2501-3000','3001-3500']
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
#plt.savefig('FlaskDeployedApp/static/outputimages/prgknn.png')
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