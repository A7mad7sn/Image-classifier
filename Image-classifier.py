from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print('----------------------------------------------------------------------')
ImSize = (64,64)
fds = []
labels = []
# Reading the Training images
print('Reading & Preprocessing the Training Images:')
print('--------------------------------------------')

#accordian
for i in range(1,22):
    if(i == 4 or i == 5 or i == 6 or i == 15 or i == 17 or i == 19 or i == 20):
        continue
    curnum = str(i)
    if(len(curnum) != 2):
        curnum = '0'+curnum
    img = imread('Assignment dataset/train/accordian/image_00'+curnum+'.jpg',as_gray=True)
    img = resize(img, ImSize)
    fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, multichannel=False, feature_vector=True)
    fds.append(fd)
    labels.append('Accordian')

#dollar_bill
for i in range(1,19):
    if(i == 4 or i == 5 or i == 6 or i == 17):
        continue
    curnum = str(i)
    if(len(curnum) != 2):
        curnum = '0'+curnum
    img = imread('Assignment dataset/train/dollar_bill/image_00'+curnum+'.jpg',as_gray=True)
    img = resize(img, ImSize)
    fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, multichannel=False, feature_vector=True)
    fds.append(fd)
    labels.append('Dollar Bill')
    
#motorbike
for i in range(1,17):
    if(i == 5 or i == 6):
        continue
    curnum = str(i)
    if(len(curnum) != 2):
        curnum = '0'+curnum
    img = imread('Assignment dataset/train/motorbike/image_00'+curnum+'.jpg',as_gray=True)
    img = resize(img, ImSize)
    fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, multichannel=False, feature_vector=True)
    fds.append(fd)
    labels.append('Motorbike')
    
#Soccer_Ball
for i in range(11,47):
    if(i == 17 or i == 19 or i == 20 or i == 23 or i == 26 or i in range(30,46)):
        continue
    curnum = str(i)
    if(len(curnum) != 2):
        curnum = '0'+curnum
    img = imread('Assignment dataset/train/Soccer_Ball/image_00'+curnum+'.jpg',as_gray=True)
    img = resize(img, ImSize)
    fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, multichannel=False, feature_vector=True)
    fds.append(fd)
    labels.append('Soccer Ball')

#Training dataframe
mydict = {'Features':fds,'Labels':labels}
Images_Training_DF = pd.DataFrame(mydict)

X_train = np.vstack(Images_Training_DF['Features'].values)#Features
Y_train = Images_Training_DF['Labels']#Label

lbl = LabelEncoder()
Y_train = lbl.fit_transform(Y_train)
Images_Training_DF['Labels'] = Y_train

print('Training Images:')
print(Images_Training_DF)
print('----------------------------------------------------------------------')


# Reading the Testing images
print('Reading & Preprocessing the Testing Images:')
print('--------------------------------------------')

fds = []
labels = []
#accordian
for i in [23,26]:
    curnum = str(i)
    if(len(curnum) != 2):
        curnum = '0'+curnum
    img = imread('Assignment dataset/test/accordian/image_00'+curnum+'.jpg',as_gray=True)
    img = resize(img, ImSize)
    fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, multichannel=False, feature_vector=True)
    fds.append(fd)
    labels.append('Accordian')

#dollar_bill
for i in [40,48]:
    curnum = str(i)
    if(len(curnum) != 2):
        curnum = '0'+curnum
    img = imread('Assignment dataset/test/dollar_bill/image_00'+curnum+'.jpg',as_gray=True)
    img = resize(img, ImSize)
    fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, multichannel=False, feature_vector=True)
    fds.append(fd)
    labels.append('Dollar Bill')
    
#motorbike
for i in [30,44]:
    curnum = str(i)
    if(len(curnum) != 2):
        curnum = '0'+curnum
    img = imread('Assignment dataset/test/motorbike/image_00'+curnum+'.jpg',as_gray=True)
    img = resize(img, ImSize)
    fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, multichannel=False, feature_vector=True)
    fds.append(fd)
    labels.append('Motorbike')
    
#Soccer_Ball
for i in [13,32,46]:
    curnum = str(i)
    if(len(curnum) != 2):
        curnum = '0'+curnum
    img = imread('Assignment dataset/test/Soccer_Ball/image_00'+curnum+'.jpg',as_gray=True)
    img = resize(img, ImSize)
    fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, multichannel=False, feature_vector=True)
    fds.append(fd)
    labels.append('Soccer Ball')


#Training dataframe
mydict = {'Features':fds,'Labels':labels}
Images_Testing_DF = pd.DataFrame(mydict)

X_test = np.vstack(Images_Testing_DF['Features'].values)#Features
Y_test = Images_Testing_DF['Labels']#Label

Y_test = lbl.transform(Y_test)
Images_Testing_DF['Labels'] = Y_test

print('Testing Images:')
print(Images_Testing_DF)
print('----------------------------------------------------------------------')



#SVM Implementing:
print('SVM Evaluating:')
print('---------------')
svc = SVC(kernel='rbf',class_weight= {0:2,1:2,2:2,3:3},C=10)
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print('Confusion Matrix:')
confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
cm_display.plot()
plt.show()
print('SVM Accuracy -->',metrics.accuracy_score(Y_test,Y_pred))