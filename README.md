# FingerprintDNN

Fast pitch tracking for musical instruments using a Keras MLP trained on the NSynth Dataset (Engel et. al) 

NSynth Dataset: https://magenta.tensorflow.org/datasets/nsynth

Audio fingerprinting is an algorithm typically used for quick database matching of audio files, and is the primary method used by Shazam for song recognition. This implementation does not make use of the hashing aspect of fingerprinting, and instead uses the raw binarized spectrogram data to train the network. This project experiments with these forms of simplified spectrograms in order to explore the validity of using this highly effective and ubiquitous processing method for deep learning.

Fingerprint extraction based on DejaVu audio fingerprinting in python: https://github.com/worldveil/dejavu

Jesse Engel, Cinjon Resnick, Adam Roberts, Sander Dieleman, Douglas Eck,
  Karen Simonyan, and Mohammad Norouzi. "Neural Audio Synthesis of Musical Notes
  with WaveNet Autoencoders." 2017.
