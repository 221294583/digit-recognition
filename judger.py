import numpy as np
from PIL import Image
from ImageDigit import  ImageDigit

img=Image.open(r"无标题.png")
imageToDigit=ImageDigit(img)
imageToDigit.histShow()
thr=int(input('请输入背景阈值:'))
imageToDigit.convert_to_bw(thr)
digits=imageToDigit.split()
imageToDigit.to_32_32("C:\\Users\\a\\PycharmProjects\\610\\result_22")
X,y=imageToDigit.featureExtract()
print('1234',X.shape,y.shape)

import numpy as np

from keras.utils import to_categorical

#y=to_categorical(y)
#print(y)

from sklearn.model_selection import train_test_split

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=.1)
from keras.models import Sequential
model=Sequential()
from keras.layers import Dense
model.add(Dense(units=20,activation='relu',input_dim=256))

model.add(Dense(units=12,activation='relu'))

model.add(Dense(units=10,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.fit(Xtrain, ytrain, epochs=100, batch_size=5)
loss_and_metrics = model.evaluate(Xtrain, ytrain, batch_size=10)
classes = model.predict(Xtest, batch_size=5)
predict=np.argmax(classes,axis=1)
ytrue=np.argmax(ytest,axis=1)
err=ytrue-predict
print(model.output_shape)
print(ytrue,'     ',predict)
print(err)
