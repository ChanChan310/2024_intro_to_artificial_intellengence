import os

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

max_time_steps = 16000
upsample_conditional_features = True
hop_length = 256

class HW5Dataset(Dataset):
    def __init__(self, meta):
        self.meta = os.path.realpath(meta)
        self.paths, self.labels = self.collect_files()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """When iterating through the dataset, the first returned value
        is the original wav file, the second is the mel spectrogram, and
        the third is the label.
        e.g. wav, mel_spectrogram, label = next(iter(val_dataset))
        """
        wav, sr = librosa.load(self.paths[idx], sr=22050)

        wav = wav / np.abs(wav).max() * 0.999
        n_fft = 1024
        hop_length = 256

        mel_spectrogram = librosa.feature.melspectrogram(
            y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=8,
            fmin=125, fmax=7600).T

        mel_spectrogram = torch.tensor(mel_spectrogram)
        label = torch.FloatTensor([self.labels[idx]]).long()
        return mel_spectrogram, label

    def collect_files(self):
        paths = []
        labels = []
        with open(self.meta, "r") as f:
            for line in f:
                file, label = line.strip().split(',')
                paths.append(os.path.realpath(file))
                labels.append(float(label))

        return paths, labels