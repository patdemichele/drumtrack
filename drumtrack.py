import librosa
import numpy as np
import matplotlib.pyplot as plt

def jumps(plp, lo, hi): # 0 < lo < hi < 1
    prev = np.roll(plp, 1) # roll 1 forward
    prev[0] = 0
    return (prev <= lo) & (plp >= hi)


def compute_onset_error(filename):
    y, sr = librosa.load(filename)
    onsets = librosa.onset.onset_detect(y,sr,units='time')
    l = len(onsets)
    A = np.array([[1] * l, [i for i in range(l)]]).T
    b = onsets.reshape(l, 1)
    x, residuals, _, _ = np.linalg.lstsq(A, b)
    errors = b - np.dot(A,x)
    return errors, residuals

