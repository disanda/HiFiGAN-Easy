import os
import json
import argparse
from utils import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram #, get_dataset_filelist
from torch.utils.data import DataLoader
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator

parser = argparse.ArgumentParser()
a = parser.parse_args()

a.input_training_file = './LJSpeech-1.1/training.txt'
a.input_validation_file = './LJSpeech-1.1/validation.txt'
a.input_wavs_dir = './LJSpeech-1.1/wavs/'
a.config = './config/config_v1.json'
a.input_mels_dir = ''

with open(a.config) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)

def get_dataset_filelist(a): # a获取到是一个文件对象, 指定训练文件和验证文件中读取音频文件的路径列表，并返回这些路径的集合。
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]
    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files

training_filelist, validation_filelist = get_dataset_filelist(a)
#print(training_filelist)

trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False, fmax_loss=h.fmax_for_loss, device='cpu',
                          fine_tuning=False, base_mels_path=a.input_mels_dir)

mel, audio, filename, mel_loss = trainset[0] # 调用内置的__getitem__方法
print(mel.shape) # mel_num
print(audio.shape) # segment_size
print(filename)
print(mel_loss.shape)

train_loader = DataLoader(trainset, shuffle=False, batch_size=h.batch_size, drop_last=True)

# "batch_size": 16,
# "segment_size": 8192,
# "num_mels": 80,
# "num_freq": 1025,
# "n_fft": 1024,
# "hop_size": 256,
# "win_size": 1024,

generator = Generator(h)
mpd = MultiPeriodDiscriminator()
msd = MultiScaleDiscriminator()

for i, batch in enumerate(train_loader):
    x, y, _, y_mel = batch # mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze()
    print(batch[0].shape, y.shape, y_mel.shape)
    y_ = generator(x)
    print(y_.shape)
    d1 = mpd(y.unsqueeze(1),y_)  #[4, 5, 16, (102,102,105,110]
    d2 = msd(y.unsqueeze(1),y_)  #[4, 3, 16, (128,65,33)]
    print(d1[0][4][15].shape)
    print(d2[0][2][15].shape)
    break