!pip3 install sklearn 

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


import cv2
from keras import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import keras
from keras.utils import np_utils
from sklearn.utils import shuffle
import os
from scipy import stats
from keras import backend as K
K.common.set_image_dim_ordering('tf')
from keras.layers import Conv2D, MaxPooling2D,Convolution2D,Activation
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split
#from sklearn import cross_validation
from sklearn.model_selection import cross_validate
#from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD,RMSprop,adam
from sklearn.metrics import classification_report,confusion_matrix
import itertools
#from sklearn import preprocessing  

# Retourner le chemin de la base de donnee ORL et leurs sujets 
Path = os.getcwd()
print(Path)
data_path = Path+'/ORL'
print(data_path)
data_dir_list = sorted(os.listdir(data_path))
#data_dir_list = os.listdir(data_path)
print("data_dir_list=",data_dir_list)
#definer des constantes
num_class = 40
img_cols= 56
img_rows= 46
num_channel = 1 #backend tensorflow = 1 pour niveau de gray

nb_epoch = 10

img_data_list = []

for data_dir in data_dir_list : 
    print("data_dir=",data_dir)
    img_list = sorted(os.listdir(data_path+'/'+data_dir))   
    for img in img_list :
        input_img = cv2.imread(data_path+'/'+data_dir+'/'+img)
        
        #input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img = cv2.resize(input_img, (img_rows,img_cols))
        img_data_list.append(input_img)

#print(label_name)
img_data = np.array(img_data_list)
#print(img_data)
img_data = img_data.astype('float32')
img_data/=255
print("img_data.shape=",img_data.shape)

num = img_data.shape[0]
label = np.ones((num,40),dtype='int64')


label[0:11] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0]
label[11:21] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0]
label[21:31] = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0]
label[31:41] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0]
label[41:51] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0]
label[51:61] = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0]
label[61:71] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0]
label[71:81] = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0]
label[81:91] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0]
label[91:101] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0]
label[101:111] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[111:121] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[121:131] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[131:141] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[141:151] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[151:161] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[161:171] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[171:181] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[181:191] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[191:201] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[201:211] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[211:221] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[221:231] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[231:241] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[241:251] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[251:261] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[261:271] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[271:281] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[281:291] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[291:301] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[301:311] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[311:321] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                  0, 0, 0, 0, 0, 0]
label[321:331] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 0, 0]
label[331:341] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                  0, 0, 0, 0, 0, 0]
label[341:351] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 0, 0, 0, 0, 0]
label[351:361] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 1, 0, 0, 0, 0]
label[361:371] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 0, 0, 0]
label[371:381] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0]
label[381:391] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 1, 0]
label[391:401] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 1]
#names=[s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,s25,s26,s27,s28,s29,s30,s31,s32,s33,s34,s35,s36,s37,s38,s39,s40]
#print(label)
Y=label
print(Y.shape)
#Y = np_utils.to_categorical(label, num_class)
print(Y)

if num_channel == 1 :
   if K.common.image_dim_ordering()== 'th' :
      img_data = np.expand_dims(img_data, axis=1)
      print(img_data.shape)
   else : 
      img_data = np.expand_dims(img_data, axis=3)
      print(img_data.shape)
else : 
   if K.common.image_dim_ordering()== 'th' :
      img_data = np.rollaxis(img_data,3,1)
      print(img_data.shape)

if K.common.image_dim_ordering == 'th':
   img_data = img_data.reshape(img_data.shape[0],3,img_cols,img_rows)   
else :
   img_data = img_data.reshape(img_data.shape[0],img_cols,img_rows,3)   
   input_shape = (3, img_cols,img_rows)
    
x,y = shuffle(img_data, Y, random_state=2)
#x,y = shuffle(img_data, Y)
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=4)

input_shape = img_data[0].shape
print(input_shape)
"""print(X_test)
print(X_train)
print(y_test)
print(y_train)"""

# define model 
model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape ))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Activation('relu'))
model.add(Conv2D(28,kernel_size=(3,3),activation='relu'))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
#model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class,activation='softmax'))
#model.add(Activation('softmax'))
# complie model
model.compile(loss='categorical_crossentropy', optimizer='adam' ,metrics=["accuracy"])
print("model.summary::::::::::::::\n")
model.summary()
"""
model.get_config()

model.layers[0].get_config()

model.layers[0].input_shape()

model.layers[0].output_shape()

model.layers[0].get_weights()

np.shape(model.layers[0].get_weights()[0])

model.layers[0].trainable
"""
# train model
hist = model.fit(X_train,y_train,batch_size=32, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, y_test))
#hist = model.fit(X_train,y_train,batch_size=32, nb_epoch=10, verbose=1, validation_split=0.2)

score = model.evaluate(X_test,y_test,True,verbose=1)
print(score[0])
print(score[1])

# tester le model
print("tester le model::::::")
test_image = X_test[1:2]
print("test_image:\n",test_image)
print(model.predict(test_image))
print(model.predict_classes(test_image))
print("y_test::\n",y_test[0:1])

# tester le model on utilisant une image
Path = os.getcwd()
img =Path+'2.pgm'
test_img = cv2.imread(img)

test_img=cv2.imread('/home/pc_dell/Bureau/1.pgm')
print("input_test_img")
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
#test_img = cv2.resize(test_img, (img_rows,img_cols))
test_img = np.array(test_img)
test_img = test_img.astype('float32')

if num_channel == 1 :
   if K.common.image_dim_ordering()== 'th' :
      test_img = np.expand_dims(test_img, axis=0)
      test_img = np.expand_dims(test_img, axis=0)
      print(test_img.shape)
   else : 
      test_img = np.expand_dims(test_img, axis=3)
      test_img = np.expand_dims(test_img, axis=0)
      print(test_img.shape)
else : 
   if K.common.image_dim_ordering()== 'th' :
      test_img = np.rollaxis(test_img,2,0)
      test_img = np.expand(test_img,axis=0)
   else :
      test_img = np.expand(test_img,axis=0)
    

# Visualizing lossees and accuracy
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(nb_epoch)
plt.figure(1, figsize=(8, 6))
plt.plot(xc,xc, train_loss)
plt.plot(xc,xc,val_loss)
plt.xlabel('num of epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train', 'val'])
plt.style.use(['classic'])
plt.figure(2, figsize=(8, 6))
plt.plot(xc,xc, train_acc)
plt.plot(xc,xc,val_acc)
plt.xlabel('num of epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train', 'val'], loc=4)
plt.style.use(['classic'])
score = model.evaluate(X_test, y_test,True, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
test_img = X_test[0:1]
print(test_img.shape)
print(model.predict(test_img))
print(model.predict_classes(test_img))
print(y_test[0:1])

#confusion matrix
Y_pred = model.predict(X_test)
print("Y_pred= ", Y_pred)
Y_pred = np.argmax(Y_pred, axis=1)
print("array_Y_pred :",Y_pred)
labels = ['s1', 's2', 's3', 's4', 's5', 's6', 's7' ,'s8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25','s26', 's27', 's28', 's29', 's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39','s40']
#print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names))
classification_rep = classification_report(np.argmax(y_test, axis=1), Y_pred, target_names=labels)
print(classification_rep)
print(confusion_matrix(np.argmax(y_test, axis=1), Y_pred))



def plot_confusion_matrix(cm,classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("Normalised confusion matrix")
    else:
        print("Conusion matrix, without normalization")
    print(cm)
    thresh = cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j], horizontalalignment="center",color="white" if cm[i,j]> thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
#compute confusion matrix
cnf_matrix=(confusion_matrix(np.argmax(y_test,axis=1),Y_pred))
np.set_printoptions(precision=10)
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=labels, normalize=False,title='Confusion matrix', cmap=plt.cm.Blues)
plt.show()
