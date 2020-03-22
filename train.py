from comet_ml import Experiment
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout
from keras.layers import Dense, Activation, Input, Softmax
import keras.optimizers
import sklearn
from sklearn import model_selection
from datetime import datetime
import pickle

from nsynth_batch_loader import DynamicBatchLoader

experiment = Experiment(api_key="OKhPlin1BVQJFzniHu1f3K1t3",
                        project_name="audiofingerprintclassifier", workspace="cm5409a")

################ HYPER-PARAMETERS ###########################

batch_size = 8
epochs = 2
test_split = 0.2

# label_path = './data/label-set-FINAL.csv'
# data_path = './data/data-set-FINAL.csv'
# train_path = '/Volumes/Remote_BU/Data/nsynth-train/nsynth-test/'
train_path = './data/'

################ READ TRAINING DATA ###########################

batchLoader = DynamicBatchLoader()

with open(train_path + 'num_examples.npy', 'rb') as f:
    num_examples = pickle.load(f)

with open(train_path + 'dimensions_X.npy', 'rb') as f:
    dimensions = pickle.load(f)

with open(train_path + 'dimensions_Y.npy', 'rb') as f:
    num_classes = pickle.load(f)

print(dimensions)
print(num_classes)
input_shape = dimensions[2]

################ BUILD MODEL ###########################

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(input_shape,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

################ TRAIN MODEL ###########################

steps_per_epoch = np.floor(num_examples/batch_size)

model.fit_generator(batchLoader.batchGenerator(batch_size, train_path, num_classes),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=1,
            shuffle=True)


################ SAVE WEIGHTS ###########################

now = datetime.now()
timestamp = str(datetime.timestamp(now))
model.save_weights('./model/weights_' + timestamp + '.h5')
model.save('./model/model_' + timestamp + '.h5')
print("Saved model to disk")

################ EVALUATE ###########################
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
