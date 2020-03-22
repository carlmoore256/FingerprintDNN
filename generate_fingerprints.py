import numpy as np
import librosa
import os
from os import walk
import fingerprint



N = 2048;
frameHop = 64; # this is the frame hop within a fingerprint
printHop = 2048; # this is the hop between full multi-frame prints
printDur = 20; # desired fingerprint length in milliseconds
minFreq = 150;
maxFreq = 18000;
maxExtrema = 8;     # 8
minProm = 3;        # 3
numClasses = 37;
plotFlag = False;


def load_files_classes(path):
    files = []
    classes = []
    for (dirpath, dirnames, filenames) in walk(path):
        this_class = os.path.basename(os.path.normpath(dirpath))
        for a in filenames:
            if a.endswith('.wav'):
                classes.append(this_class)
                files.append(dirpath + '/' + a)
    return files, classes

def generate_spectrogram(filename, sr, fft_size, num_mels, hop_size):
    audio, sr = librosa.load(filename)
    spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=fft_size, n_mels=num_mels,
                                    hop_length=hop_size, window='hann',
                                    center=True)

    S_dB = librosa.power_to_db(spec, ref=np.max)
    S_dB = np.uint8((S_dB + 80) * 3.2)
    spectrogram = Image.fromarray(S_dB.astype(np.uint8))
    return spectrogram


def generate_fingerprint(filename):
    audio, sr = librosa.load(filename)
    print = fingerprint.fingerprint(audio, sr,
                    wsize=4096,
                    wratio=0.5,
                    amp_min=10,
                    peak_neighborhood=10)
    return print


if __name__ == '__main__':

    path = '/Users/carl/Documents/samples'

    files, classes = load_files_classes(path)

    for f, c in zip(files, classes):
        print(c)
        print = generate_fingerprint(f)

        break

# this creates a struct with the string file name of every .wav file in
# bounces/
#
# allFiles = dir(strcat(audioDir, '*.wav'));
#
# numDatabaseFiles = length(allFiles);
#
# # make a full path file name for the dataset and labelset CSVs, which will end up in the
# # root directory
# dataSetFileName = strcat(projectDir, 'data-set.csv');
# dataLabelFileName = strcat(projectDir, 'label-set.csv');
#
# # this just initializes the file to contain no data
# dlmwrite(dataSetFileName, [], 'delimiter', ',');
# dlmwrite(dataLabelFileName, [], 'delimiter', ',');
#
# # iterate through all files in bounces/ directory
# for file=1:numDatabaseFiles
#
#     # pull the .name field from the allFiles struct and concatenate with the bounces/ directory
#     thisFile = allFiles(file).name;
#     # thisFolder = strcat(projectDir, 'bounces/');
#     thisFile = strcat(audioDir, thisFile);
#
#     # display it so we know which file we're on as processing is being done
#     disp(thisFile);
#
#     [x, R] = audioread(thisFile);
#
#     printDurSamps = round((printDur/1000) * R);
#
#     # grab the name of this file (not the full path)
#     thisFile = allFiles(file).name;
#     # remove the final 9 characters from the audio file name (-mic1.wav)
#     # because those characters aren't used in the corresponding label file
#     labelFileName = thisFile(1:end-9);
#     # append .txt to it to create the label file name automatically
#     labelFileName = strcat(labelFileName, '.txt');
#     # add the label directory to get a full path
#     labelFileName = strcat(labelDir, labelFileName);
#
#     disp(labelFileName);
#
#     xLabels = load(labelFileName, '-ascii');
#
#     # see how many transients there are in this label file
#     numLabels = size(xLabels, 1);
#
#     for i=1:numLabels
#         # for each label, set the start/end sample the ~50ms range to analyze.
#         # make this begin a half window BEFORE the actual onset
#         # make it end half a window AFTER the end of the label
#         startSamp = round(xLabels(i, 1)*R) - (N/2);
#         labelEndBound = round(xLabels(i, 2)*R) + (N/2);
#         endSamp = startSamp + printDurSamps;
#
#         MIDInote = xLabels(i, 3);
#         printNum = 1;
#
#         while endSamp < labelEndBound
#
#             xWindow = x(startSamp:endSamp);
#
#             # should consider getting the magnitude spectrum flux GROWTH only here,
#             # then looking for local maxima in that to get the fingerprint
#
#             [~, printDouble, printBins, numFrames] = ...
#                 extractFingerprint(xWindow, R, N, frameHop, minFreq, maxFreq, minProm, maxExtrema);
#
#             if plotFlag
#                 binStart = round(freq2bin(minFreq, N, R))+1;
#                 binEnd = round(freq2bin(maxFreq, N, R))+1;
#
#                 # plot the input constellation
#                 plot(1:numFrames, printBins, 'x'); axis([1 numFrames binStart binEnd]);
#
#                 # title the figure
#                 titleString = sprintf("Note %i, printNum %i", MIDInote, printNum);
#                 title(titleString);
#
#                 pause(0.1);
#                 # waitforbuttonpress;
#             end
#
#             # flatten out the fingerprint to one long row
#             printDouble = reshape(printDouble, 1, []);
#
#             # append a single line to our data file with the single fingerprint
#             # for this note (in future, need to add multiple fingerprints at
#             # different moments of the note transient, based on a hop size of
#              # ~10ms?)
#             dlmwrite(dataSetFileName, printDouble, '-append', ...
#                 'delimiter', ',', 'precision', '%i');
#
#             # need to simultaneously add a line to the label file
#             dlmwrite(dataLabelFileName, oneHot(MIDInote-52, numClasses), ...
#                 '-append', 'delimiter', ',', 'precision', '%i');
#
#             # advance by printHop samples
#             startSamp = startSamp + printHop;
#             endSamp = startSamp + printDurSamps;
#             printNum = printNum+1;
#
#         end
#     end
#
# end
