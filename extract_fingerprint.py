import numpy as np


def extractFingerprint(x, R, N, hop, minFreq, maxFreq, minProm, maxExtrema):
    # % front pad x with N/2 zeros so that the audio sample is dead center in the
    # % Hann window
    x = [np.zeros((N/2, 1)); x]

    # % break the signal x into sub-windows, contained in xFrames
    [xFrames, numFrames] = audioFrames(x, N, hop)

    # % create a Hann window, to fade in/out each subwindow before FFT
    h = hann(N)

    # % apply the Hann window to all columns (subwindows) of xFrames
    xFrames = xFrames .* h

    # % do Fourier analysis (FFT) on every subwindow to get its spectrum
    # % information
    XFrames = fft(xFrames)

    # % find the bin number for maxFreqHz, since that's going to be our cutoff.
    # % remember to round the fractional bin number, and add 1 to make it
    # % useable as a MATLAB array  index
    binStart = round(freq2bin(minFreq, N, R))+1
    binEnd = round(freq2bin(maxFreq, N, R))+1

    # % get the magnitudes using abs(), and only keep frequency bands 1 through
    # % binCutoff
    magSpecFrames = abs(XFrames(binStart:binEnd, :))

    # % take the growth-only flux (this didn't work out)
    # % magSpecFrames = specFlux(magSpecFrames);
    # % numFrames = numFrames-1;

    # % find the local maxima
    printLog = islocalmax(magSpecFrames, 1, 'MinProminence', minProm, 'MaxNumExtrema', maxExtrema)
    # % 'MaxNumExtrema'
    # % use this argument if we want to limit the number of maxima per column

    # % trim the first and final rows off of the fingerprint since those rows
    # % cannot have local maxima (missing one neighbor)
    printLog = printLog(2:end-1, :)

    # % this version of the constellation map is comprised of double-precision
    # % floating point numbers rather than "logicals", so we can use it with
    # % xcorr2() to do our matching test
    printDouble = printLog*1

    # % replace each 1 in the logical results with its corresponding bin number
    # % so that we can plot it
    printBins = printLog .* (binEnd-1:-1:binStart+1)'

    return printLog, printDouble, printBins, numFrames
