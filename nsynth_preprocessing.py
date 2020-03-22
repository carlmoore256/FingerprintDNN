import json
import numpy as np
import librosa
import pickle
import matplotlib.pyplot as plt
import joblib
import os.path
import glob
import fingerprint
from keras.utils import to_categorical

def generate_fingerprint(audio, sr):
    print = fingerprint.fingerprint(audio, sr,
                    wsize=1024,
                    wratio=0.5,
                    amp_min=10,
                    peak_neighborhood=10)
    return print

def extract_features(inpath, example_win, hop, flatten=True):

    audio, sr = librosa.load(inpath, res_type='kaiser_fast')
    data = generate_fingerprint(audio, sr)

    endIdx = hop
    startIdx = 0
    output_arr = []

    while True:
        endIdx = startIdx + example_win
        if endIdx > data.shape[0]:
            break
        if flatten:
            thisData = data[startIdx:endIdx, :]
            thisData = np.reshape(thisData, (thisData.shape[0] * thisData.shape[1]))
        else:
            thisData = data[startIdx:endIdx, :]

        output_arr.append(thisData)
        startIdx += hop

    output_arr = np.asarray(output_arr)

    return output_arr

def saveMemmap(X, Y, batch_idx, batch_save):
    print('save memmap - input x shape: ' + str(X.shape))
    print('save memmap - input y shape: ' + str(Y.shape))

    # offset_x = int((batch_idx * batch_save) * X.shape[0] * X.shape[1] * 32 / 8)
    # offset_y = int((batch_idx * batch_save) * Y.shape[0] * Y.shape[1] * 32 / 8)

    offset_x = int((batch_idx * X.shape[0]) * X.shape[1] * 32 / 8)
    offset_y = int((batch_idx * X.shape[0]) * Y.shape[1] * 32 / 8)


    x_file = np.memmap(path + 'train_data_x.memmap', dtype='float32', mode='r+', shape=X.shape, offset=offset_x)
    x_file[:] = X
    del x_file

    y_file = np.memmap(path + 'train_data_y.memmap', dtype='float32', mode='r+', shape=Y.shape, offset=offset_y)
    y_file[:] = Y
    del y_file

def parse(path, num_classes, numfiles, batch_save, num_threads, example_win, hop, init_memmap=False):
    # batch_save determines the size of the batch saved before a new array is started
    json_path = path + 'examples_copy.json'
    with open(json_path, 'r') as f:
        train = json.load(f)

    # get the dimensions of fingerprint using dummy file
    sample_print = extract_features('./sample/sample0.wav', example_win, hop)
    dimensions = sample_print.shape

    examplesPerPrint = dimensions[0]
    # expand size of numfiles to number of actual examples per file
    numfiles = numfiles * examplesPerPrint

    print('dimensions: ' + str(dimensions))
    print('total number of examples: ' + str(numfiles))
    print('examples per print: ' + str(examplesPerPrint))

    dimsX = [dimensions[0], example_win, dimensions[1]]
    print(dimsX)
    # save size of x_train and y_train
    with open(path + 'dimensions_X.npy', 'wb') as f:
        pickle.dump(dimsX, f)

    with open(path + 'dimensions_Y.npy', 'wb') as f:
        pickle.dump(num_classes, f)

    with open(path + 'num_examples.npy', 'wb') as f:
        pickle.dump(numfiles, f)

    # exit()

    if init_memmap:
        x_file_init = np.memmap(path + 'train_data_x.memmap', dtype='float32', mode='w+', shape=((numfiles, dimensions[1])))
        y_file_init = np.memmap(path + 'train_data_y.memmap', dtype='float32', mode='w+', shape=((numfiles, num_classes)))
        del x_file_init
        del y_file_init
        print('initialized memmap files')
    else:
        print('init_memmap=False, skipping memmap init')

    batch_idx=0
    loaded_files_batch = 0
    loaded_files_batch_counter = 0
    loaded_files_total = 0

    x_train = np.zeros((batch_save * examplesPerPrint, dimensions[1]))
    print('x_train shape ' + str(x_train.shape))
    y_train = np.zeros((batch_save * examplesPerPrint, num_classes))

    for key in train:
        obj = train[key]

        thisPath = path + 'audio/' + obj['note_str'] + '.wav'
        thisClass = to_categorical(obj['pitch'], num_classes=num_classes)

        instFingeprints = extract_features(thisPath, example_win, hop)
        # idx = 0
        # for f in instFingeprints:
        #     print('current fingerprint idx: ' + str(idx))
        #     idx += 1
        #     print(np.count_nonzero(f))
        # maxX = np.count_nonzero(X)
        # print('non-zero values in x ' + str(maxX))

        if instFingeprints.shape[0] == examplesPerPrint:
            x_train[loaded_files_batch: loaded_files_batch + examplesPerPrint, :] = instFingeprints
            y_train[loaded_files_batch: loaded_files_batch + examplesPerPrint, :] = np.tile(thisClass, (examplesPerPrint, 1))

            print(str(loaded_files_total) + '/' + str(numfiles) + ' loaded ' + str((loaded_files_total/numfiles)*100) + '%')

            # use loaded files instead of verified to avoid issues
            if loaded_files_total >= numfiles:
                print('last batch, saving to memmap')
                saveMemmap(x_train, y_train, batch_idx, batch_save)
                break

            if loaded_files_batch/examplesPerPrint % batch_save == batch_save - 1:
                print('saving to memmap')
                saveMemmap(x_train, y_train, batch_idx, batch_save)
                x_train = np.zeros((batch_save * examplesPerPrint, dimensions[1]))
                print('x_train shape ' + str(x_train.shape))
                y_train = np.zeros((batch_save * examplesPerPrint, num_classes))
                loaded_files_batch = 0
                batch_idx += 1

            # loaded_files += 1
            loaded_files_batch += examplesPerPrint
            loaded_files_total += examplesPerPrint
        else:
            print('null file, skipping')
            numfiles -= examplesPerPrint


    print('closing file...')

if __name__ == '__main__':

    path = '/Volumes/Remote_BU/Data/nsynth-train/nsynth-test/'
    total_files = len(glob.glob(path + "audio/*.wav"))
    print(str(total_files) + ' found in ' + path + "audio/")

    parse(path, 128, total_files, 128, num_threads=1, example_win=256, hop=32)
