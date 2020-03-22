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

def extract_features(inpath, subSampleIndicies, dimensions, example_win):
    # output_arr = np.zeros((example_win * len(subSampleIndicies), dimensions[1]))
    # prepare an array sized as a flattened array
    # output_arr = np.zeros((len(subSampleIndicies), example_win * dimensions[1]))
    output_arr = []

    audio, sr = librosa.load(inpath, res_type='kaiser_fast')
    data = generate_fingerprint(audio, sr)

    endIdx = hop
    startIdx = 0
    examplesPerPrint = 0
    subSampleIndicies = []

    while True:
        endIdx = startIdx + example_win
        if endIdx > dimensions[0]:
            break
        subSampleIndicies.append([startIdx, endIdx])
        startIdx += hop

    for s in subSampleIndicies:
        output_arr.append(data[s[0]:s[1],:])
        # output_arr[s[0]:s[1],:] = data[s[0]:s[1],:]

    return output_arr

def saveMemmap(X, Y, batch_idx, batch_save):
    # X = np.asarray(x)
    # Y = np.asarray(y)

    print('save memmap - input x shape: ' + str(X.shape))
    print('save memmap - input y shape: ' + str(Y.shape))

    l = list(X.shape)

    # offset_x = int((batch_idx * batch_save)*int(l[1])*int(l[2])*32/8)
    # offset_y = int((batch_idx * batch_save)*32/8)
    # offset_x = int((batch_idx * batch_save)*int(l[0])*int(l[1])*32/8)
    offset_x = int((batch_idx * batch_save) * X.shape[1] * 32 / 8)
    offset_y = int((batch_idx * batch_save) * Y.shape[1] * 32 / 8)

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
    audio, sr = librosa.load('./sample/sample0.wav', res_type='kaiser_fast')
    sample_print = generate_fingerprint(audio, sr)
    dimensions = sample_print.shape

    print('dimensions: ' + str(dimensions))

    endIdx = hop
    startIdx = 0
    examplesPerPrint = 0
    subSampleIndicies = []

    while True:
        endIdx = startIdx + example_win
        if endIdx > dimensions[0]:
            break
        subSampleIndicies.append([startIdx, endIdx])
        startIdx += hop

    examplesPerPrint = len(subSampleIndicies)
    # expand size of numfiles to number of actual examples per file
    numfiles = numfiles * examplesPerPrint
    print('total number of examples: ' + str(numfiles))
    print('examples per print: ' + str(examplesPerPrint))

    with open(path + 'dimensions_X.npy', 'wb') as f:
        pickle.dump((example_win * dimensions[1]), f)

    with open(path + 'dimensions_Y.npy', 'wb') as f:
        pickle.dump(Y_train.shape, f)

    with open(path + 'num_examples.npy', 'wb') as f:
        pickle.dump(numfiles, f)

    # output_print = np.reshape(output_print, (output_print.shape[0] * output_print.shape[1]))

    loaded_files = 0
    verified_files = 0
    # initialize the memmap files to write to
    # x_file_init = np.memmap(path + 'train_data_x.memmap', dtype='float32', mode='r+', shape=((numfiles,176,176)))
    # y_file_init = np.memmap(path + 'train_data_y.memmap', dtype='float32', mode='r+', shape=((numfiles,)))

    if init_memmap:
        x_file_init = np.memmap(path + 'train_data_x.memmap', dtype='float32', mode='w+', shape=((numfiles, example_win * dimensions[1])))
        y_file_init = np.memmap(path + 'train_data_y.memmap', dtype='float32', mode='w+', shape=((numfiles, num_classes)))

        del x_file_init
        del y_file_init
        print('initialized memmap files')
    else:
        print('init_memmap=False, skipping memmap init')

    batch_idx=0

    x_train = np.zeros((batch_save * examplesPerPrint, example_win * dimensions[1]))
    y_train = np.zeros((batch_save * examplesPerPrint, num_classes))

    for key in train:
        obj = train[key]

        thisPath = path + 'audio/' + obj['note_str'] + '.wav'
        thisClass = to_categorical(obj['pitch'], num_classes=num_classes)

        instFingeprints = extract_features(thisPath, subSampleIndicies, dimensions, example_win)

        # flatten fingerprint

        x_train[loaded_files:loaded_files + examplesPerPrint, :] = instFingeprints

        print(str(loaded_files) + '/' + str(numfiles) + ' loaded ' + str((loaded_files/numfiles)*100) + '%')

        # use loaded files instead of verified to avoid issues
        if loaded_files == numfiles:
            print('last batch, saving to memmap')
            saveMemmap(x_train, y_train, batch_idx, batch_save)
            break

        if loaded_files % batch_save == batch_save - 1:
            print('saving to memmap')
            saveMemmap(x_train, y_train, batch_idx, batch_save)

            x_train = np.zeros((batch_save * examplesPerPrint, example_win * dimensions[1]))
            y_train = np.zeros((batch_save * examplesPerPrint, num_classes))

            batch_idx += 1

        loaded_files += examplesPerPrint

    print('closing file...')
    return x_train, y_train, verified_files

if __name__ == '__main__':

    path = '/Volumes/Remote_BU/Data/nsynth-train/nsynth-test/'
    total_files = len(glob.glob(path + "audio/*.wav"))
    print(str(total_files) + ' found in ' + path + "audio/")

    X_train, Y_train, num_examples = parse(path, 128, total_files, 128, num_threads=1, example_win=256, hop=32)
    # 289205
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)


#
# x_file = np.memmap(path + 'train_data_x.memmap', dtype='float32', mode='w+', shape=x_train.shape)
# x_file[:] = x_train[:]
# del x_file
# #
# y_file = np.memmap(path + 'train_data_y.memmap', dtype='float32', mode='w+', shape=y_train.shape)
# y_file[:] = y_train[:]
# del y_file
