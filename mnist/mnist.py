import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import numpy as np
from keras import layers
import matplotlib.pyplot as plt

import keras.datasets.mnist as mnist
(train_image,train_label),(test_image,test_label) = mnist.load_data()
'''print(train_image.shape)
plt.imshow(train_image[0])
plt.show()
print(train_label[0])'''

model = keras.Sequential()
model.add(layers.Flatten())             #展平 （60000,28,28）>>（60000,28*28）
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])

model.fit(train_image,train_label,epochs=50,batch_size=512)

z = np.argmax(model.predict(test_image[:10]),axis=1)
print(z)
print(test_label[0:10])

#模型的优化