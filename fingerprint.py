import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure,
                                      iterate_structure, binary_erosion)

def fingerprint(channel_samples, Fs=44100,
                wsize=4096,
                wratio=0.5,
                amp_min=10,
                peak_neighborhood=20):
    # FFT the signal and extract frequency components
    channel_samples = (channel_samples * 32768).astype(np.int16)


    arr2D = mlab.specgram(
        channel_samples,
        NFFT=wsize,
        Fs=Fs,
        window=mlab.window_hanning,
        noverlap=int(wsize * wratio))[0]

    # print('length of spectrogram' + str(arr2D.shape))
    # apply log transform since specgram() returns linear array
    arr2D = 10 * np.log10(arr2D)
    arr2D[arr2D == -np.inf] = 0  # replace infs with zeros

    # find local maxima
    local_maxima = get_2D_peaks(arr2D, peak_neighborhood, plot=False, amp_min=amp_min)

    # return hashes
    return local_maxima


def get_2D_peaks(arr2D, peak_neighborhood, plot=False, amp_min=10):
    #  http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.iterate_structure.html#scipy.ndimage.iterate_structure
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, peak_neighborhood)

    # find local maxima using our filter shape
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
    background = (arr2D == 0)
    eroded_background = binary_erosion(background, structure=neighborhood,
                                       border_value=1)

    # Boolean mask of arr2D with True at peaks (Fixed deprecated boolean operator by changing '-' to '^')
    detected_peaks = local_max ^ eroded_background

    # extract peaks
    amps = arr2D[detected_peaks]
    j, i = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten()
    peaks = zip(i, j, amps)
    peaks_filtered = filter(lambda x: x[2]>amp_min, peaks) # freq, time, amp
    # get indices for frequency and time
    frequency_idx = []
    time_idx = []
    output_print = np.zeros(arr2D.shape)

    for x in peaks_filtered:
        # print(str(x[1]) + ' x1')
        # print(str(x[0]) + ' x0')
        output_print[x[1], x[0]] = 1
        frequency_idx.append(x[1])
        time_idx.append(x[0])

    if plot:
        # scatter of the peaks
        fig, ax = plt.subplots()
        ax.imshow(output_print)
        # ax.scatter(time_idx, frequency_idx)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title("Spectrogram")
        plt.gca().invert_yaxis()
        plt.show()

    # output_print = np.reshape(output_print, (output_print.shape[0] * output_print.shape[1]))

    return output_print
