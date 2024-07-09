import math
import os
import random
import torch
import torch.utils.data
import numpy as np
import librosa
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0
# WAV 文件的最大幅度值。标准的 16 位 PCM 音频格式的范围是 -32768 到 32767，
# MAX_WAV_VALUE用来将音频归一化到 -1 到 1 之间。

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)
# 动态范围压缩函数，将输入的信号 x 进行压缩，避免信号范围过大。
# C是压缩系数，clip为裁剪函数，保证值在 [clip_val, +$\infty$]

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C
#动态范围解压缩函数，用于恢复被压缩的信号。

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output
#对幅度谱进行规范化（动态范围压缩）。

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output
# 从压缩的状态解回到原始的幅度范围

mel_basis = {}   # 存储梅尔滤波器组
hann_window = {} # 汉宁窗函数

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        #mel = librosa.filters.mel(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax) # [80, 513]
        #print(mel.shape) (80, 513)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device) 
        # 如果 fmax 不在 mel_basis 中（即未计算对应的梅尔滤波器），则调用 librosa_mel_fn 函数计算梅尔滤波器，并将结果存储在 mel_basis 中。
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
        # print(hann_window['cpu'].shape) - 依赖于采样率、FFT大小、梅尔滤波器数量、频率范围等参数。

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    # 对输入的音频数据 y 进行填充（padding），使其长度适合进行STFT（短时傅里叶变换）
    #print(y.shape) # [1,1,8960]
    y = y.squeeze(1)
    #print(y.shape)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    # 输入信号 y 的短时傅里叶变换。这一步得到的 spec 是复数形式的频谱。

    # 窗口大小 (n_fft)，即每个窗口中包含的样本数, 是频域精度，通常是2的幂次方（如256、512、1024等），较大的 n_fft 可以提供更高的频率分辨率，但也会导致计算量增加。

    # 帧移 (hop_size):  指在时间上相邻窗口之间的间隔，即每次取样时窗口的滑动步长。，较小的 hop_size 提供更高时间分辨率，但增加计算量，通常为窗口大小n_fft的一半。
    # 默认等于 n_fft/4

    # 窗口函数长度 (win_size): win_size 是指窗口函数的长度，用于加权每个窗口内的样本。 默认等于 n_fft

    # 窗口函数（如汉宁窗、汉明窗等）会在输入信号的每一帧上乘以这个窗口函数，减少窗口边缘的频谱泄露（spectral leakage）。

    spec = spec.abs()
    #print(spec.shape)
    #spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    # 复数形式的频谱转换为能量谱。这里使用了平方和开方的方法来计算能量，并添加了一个小的常数（1e-9）以避免除以零的情况。

    spec = spec.transpose(1, 2).to(y.device)  # 或者 spec = spec.permute(1, 0)
    #print(spec.shape)
    mel_basis[str(fmax)+'_'+str(y.device)] = mel_basis[str(fmax)+'_'+str(y.device)].transpose(0,1)
    #print(mel_basis[str(fmax)+'_'+str(y.device)].shape)
    spec = torch.matmul(spec, mel_basis[str(fmax)+'_'+str(y.device)]) # 将能量谱乘以预先计算的梅尔滤波器矩阵 mel_basis，得到最终的梅尔频谱图。
    spec = spectral_normalize_torch(spec) # 对梅尔频谱进行归一化处理。
    spec = spec.transpose(1, 2)
    return spec

def get_dataset_filelist(a): # a获取到是一个文件对象, 指定训练文件和验证文件中读取音频文件的路径列表，并返回这些路径的集合。
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files # 包含训练文件路径的列表。
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files) # 控制是否对音频进行分割和是否打乱训练文件列表
        self.segment_size = segment_size # 分段大小，用于将音频切成统一的段（配合split参数）
        self.sampling_rate = sampling_rate # 
        self.split = split #  # 将不同长度的音频截取固定长度的连续采样点（其实点随机的8192个点）
        self.n_fft = n_fft # 
        self.num_mels = num_mels # 80
        self.hop_size = hop_size # 
        self.win_size = win_size # 
        self.fmin = fmin # 0
        self.fmax = fmax # 8000
        self.fmax_loss = fmax_loss
        self.cached_wav = None #加载和处理后的音频数据缓存到 self.cached_wav 中，以便后续重复使用
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0 # 控制缓存重用的次数
        self.device = device # 
        self.fine_tuning = fine_tuning # 是否进行微调
        self.base_mels_path = base_mels_path # 基础梅尔频谱图的路径（仅在 fine_tuning=True 时使用）

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename)
            audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        #print(audio.shape)
        if not self.fine_tuning:
            #print('-##########################')
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
            #print(audio.shape) # (1, 8192)
            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
        else:
            #print('##########################')
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
