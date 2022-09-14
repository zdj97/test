import os
import math
import random
import torch
import torchaudio
from torchaudio import transforms
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import Dataset
from typing import Tuple, Optional, Union
from torch import Tensor

EPS = 1e-9
SAMPLE_RATE = 16000


# default labels from GSC dataset
DEFAULT_LABELS = [
    '1',
    '0',
    '-1',
]

N_CLASS = len(DEFAULT_LABELS)
HASH_DIVIDER = "_nohash_"

def load_item(filepath):
   # print(filepath)
    x=filepath.strip().split(' ')
   # print(x)
    # relpath = os.path.relpath(filepath, path)
    label, filename = filepath.strip().split(' ')
    # speaker, _ = os.path.splitext (filename)
    # speaker, _ = os.path.splitext (speaker)

    # speaker_id, utterance_number = speaker.split (HASH_DIVIDER)
    # utterance_number = int (utterance_number)

    # Load audio
    # print(filepath)
    waveform, sample_rate = torchaudio.load (filename)
    return waveform, sample_rate, label   #,speaker_id, utterance_number

def prepare_wav(waveform, sample_rate):
    if sample_rate != SAMPLE_RATE: 
        resampler = transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    to_mel = transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, f_max=8000, n_mels=40)
    log_mel = (to_mel(waveform) + EPS).log2()
    return log_mel


class SubsetSC(Dataset):
    def __init__(self, subset: str, path="./HIXIAOWEN/"):
        self.to_mel = transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, f_max=8000, n_mels=40)
        self.subset = subset
        self.path=path


        def load_list(filename):
            filepath = os.path.join(self.path, filename)
            with open(filepath) as fh:
                return [
                    line.strip() for line in fh
                ]

        # def load_list(filename):
        #     S=[]
        #     # print(self.path)
        #     filepath = os.path.join(self.path, filename)
        #     # print(filepath)
        #     with open(filepath) as fh:
        #         #print(fh)
        #         for line in fh:
        #             # print(line)
        #             s= line.split(' ')[1]
        #             # print(s)
        #             S.append(s)
        #     return S

        self._noise = []

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
            # print(type(self._walker[0]))
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            # excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            # excludes = set(excludes)
            # self._walker = [w for w in self._walker if w not in excludes]
            self._walker = load_list ("train_list.txt")
            noise_paths = [w for w in os.listdir("./HIXIAOWEN/_background_noise_") if w.endswith(".wav")]
            for item in noise_paths:
                noise_path =  os.path.join(self.path, "_background_noise_", item)
                noise_waveform, noise_sr = torchaudio.sox_effects.apply_effects_file(noise_path, effects=[])
                noise_waveform = transforms.Resample(orig_freq=noise_sr, new_freq=SAMPLE_RATE)(noise_waveform)
                self._noise.append(noise_waveform)
        else:
            raise ValueError(f"Unknown subset {subset}. Use validation/testing/training")

    def _noise_augment(self, waveform):
        noise_waveform = random.choice(self._noise)

        noise_sample_start = 0
        if noise_waveform.shape[1] - waveform.shape[1] > 0:
            noise_sample_start = random.randint(0, noise_waveform.shape[1] - waveform.shape[1])
        noise_waveform = noise_waveform[:, noise_sample_start:noise_sample_start+waveform.shape[1]]

        signal_power = waveform.norm(p=2)
        noise_power = noise_waveform.norm(p=2)

        snr_dbs = [20, 10, 3]
        snr = random.choice(snr_dbs)

        snr = math.exp(snr / 10)
        scale = snr * noise_power / signal_power
        noisy_signal = (scale * waveform + noise_waveform) / 2
        return noisy_signal

    def _shift_augment(self, waveform):
        shift = random.randint(-1600, 1600)
        waveform = torch.roll(waveform, shift)
        if shift > 0:
            waveform[0][:shift] = 0
        elif shift < 0:
            waveform[0][shift:] = 0
        return waveform

    def _augment(self, waveform):
        if random.random() < 0.8:
            waveform = self._noise_augment(waveform)
        
        waveform = self._shift_augment(waveform)

        return waveform

    def __getitem__(self, n):
        # waveform, sample_rate, label, _, _ = self().__getitem__(n)
        fileid = self._walker[n]
        # print(type(fileid))
        # a,b=torchaudio.load('/Users/zmj1997/fsdownload/HIXIAOWEN/data_tar/mobvoi_hotword_dataset/b5526d1af19a5cd8caa2ea696b6cfcae.wav')
        # a,b=torchaudio.load(fileid.strip())
        # print(a,b)
        # print(type(fileid))
        waveform, sample_rate, label=load_item (fileid)
        if sample_rate != SAMPLE_RATE: 
            resampler = transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
        if self.subset == "training":
            waveform = self._augment(waveform)
        log_mel = (self.to_mel(waveform) + EPS).log2()

        return log_mel, label
    def __len__(self) -> int:
        return len(self._walker)

_label_to_idx = {label: i for i, label in enumerate(DEFAULT_LABELS)}
print(_label_to_idx)
_idx_to_label = {i: label for label, i in _label_to_idx.items()}


def label_to_idx(label):
    return _label_to_idx[label]


def idx_to_label(idx):
    return _idx_to_label[idx]


def pad_sequence(batch):
    batch = [item.permute(2, 1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return batch.permute(0, 3, 2, 1) 


def collate_fn(batch):
    tensors, targets = [], []
    for log_mel, label in batch:
        tensors.append(log_mel)
        targets.append(label_to_idx(label))

    tensors = pad_sequence(tensors)
    targets = torch.LongTensor(targets)

    return tensors, targets
