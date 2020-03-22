import numpy as np
import pickle
import keras

class DynamicBatchLoader:
    def batchGenerator(self, batch_size, path, num_classes):

        batch_start = 0
        batch_end = batch_size

        with open(path + 'dimensions_X.npy', 'rb') as f:
            dimensions_X = pickle.load(f)

        with open(path + 'num_examples.npy', 'rb') as f:
            num_examples = pickle.load(f)

        #this line is just to make the generator infinite, keras needs that
        while True:
            print('initializing generator')
            batch_start = 0
            batch_end = batch_size

            while batch_start < num_examples:
                limit = min(batch_end, num_examples)
                # print('batch start ' + str(batch_start) + path)
                # index = batch_start
                x_train = []
                y_train = []
                # print('starting new batch, batch_start: ' + str(batch_start))
                for i in range(batch_size):
                    offset_x = int((batch_start + i) * dimensions_X[2] * 32 / 8)
                    offset_y = int((batch_start + i) * num_classes * 32 / 8)
                    # numpy.memmap.take

                    x_train.append(np.memmap(path + 'train_data_x.memmap', dtype='float32', mode='r', shape=((dimensions_X[2],)), offset=offset_x))
                    # print(np.memmap(path + 'train_data_x.memmap', dtype='float32', mode='r', shape=((dimensions_X[2],)), offset=offset_x).shape)
                    # print(np.memmap(path + 'train_data_y.memmap', dtype='float32', mode='r', shape=((num_classes,)), offset=offset_y).shape)
                    y_train.append(np.memmap(path + 'train_data_y.memmap', dtype='float32', mode='r', shape=((num_classes,)), offset=offset_y))

                # keep dimensions happy
                x_train = np.asarray(x_train)
                y_train = np.asarray(y_train)
                # x_train = np.repeat(x_train[:, :, :, np.newaxis],1,axis=2)

                yield (x_train, y_train) #a tuple with two numpy arrays with batch_size samples
                batch_start += batch_size
                batch_end += batch_size
