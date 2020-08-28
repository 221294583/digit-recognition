from sklearn.datasets.samples_generator import make_circles
import numpy as np

X,y=make_circles(100,factor=.1,noise=.1)

from keras.utils import to_categorical

y=to_categorical(y)

from sklearn.model_selection import train_test_split

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=.1)

from keras.models import Sequential
model=Sequential()
from keras.layers import Dense
model.add(Dense(units=10,activation='relu',input_dim=2))

model.add(Dense(units=2,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.fit(Xtrain, ytrain, epochs=100, batch_size=5)
loss_and_metrics = model.evaluate(X, y, batch_size=10)
classes = model.predict(Xtest, batch_size=5)
predict=np.argmax(classes,axis=1)
ytrue=np.argmax(ytest,axis=1)
err=ytrue-predict
print(model.output_shape)
print(err)
