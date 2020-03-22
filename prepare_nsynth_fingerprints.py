import numpy as np
import glob
from PIL import Image
import librosa
import json
import fingerprint



def generate_fingerprint(filename):
    audio, sr = librosa.load(filename)
    print = fingerprint.fingerprint(audio, sr,
                    wsize=4096,
                    wratio=0.5,
                    amp_min=10,
                    peak_neighborhood=10)
    return print


def loadDataset(self, path, label, batch_size):
    # return an array with labels and an array of filenames
    json_path = path + 'examples_copy.json'
    with open(json_path, 'r') as f:
        train = json.load(f)
    x_train_fn = []
    y_train = []

    idx = 0

    for key in train:
        obj = train[key]
        filename = path + 'audio/' + obj['note_str'] + '.wav'
        inst_family = obj[label]
        x_train_fn.append(filename)
        y_train.append(inst_family)

    # lb = LabelBinarizer()
    # y_train = lb.fit_transform(list(y_train))
    x_train_fn = np.asarray(x_train_fn)
    y_train = np.asarray(y_train)

    # returns an array with x_train filenames and y_train labels

    num_examples = len(x_train_fn)
    steps_per_epoch = int(num_examples/batch_size)
    print(len(x_train_fn))
    print(len(y_train))
    return x_train_fn, y_train, steps_per_epoch





def imageLoader(self, x_train_fn, y_train, batch_size):
    # global batch_start
    # global batch_end
    #this line is just to make the generator infinite, keras needs that
    while True:

        # print(batch_end)

        self.batch_start = self.batch_end
        self.batch_end = self.batch_start + batch_size

        while self.batch_start < self.batch_end:
            limit = min(self.batch_end, self.num_examples)

            x_train = []
            print(str(batch_start) + ' batch start')
            print(str(batch_end) + ' batch end')
            print(x_train_fn[batch_start:limit])

            for file in x_train_fn[batch_start:limit]:
                print('generating mel spec')
                print(f)
                audio, sr = librosa.load(f)
                spec = librosa.feature.melspectrogram(audio, sr, n_fft = 1024)
                x_train.append(x)

            X = np.array(x_train)
            Y = y_train[batch_start:limit]
            Y = keras.utils.to_categorical(Y, num_classes=num_classes)
            yield (X,Y) #a tuple with two numpy arrays with batch_size samples

            self.batch_start += self.batch_size
            self.batch_end += self.batch_size
