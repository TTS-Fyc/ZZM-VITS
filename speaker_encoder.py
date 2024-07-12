import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from speechbrain.pretrained import ECAPA_TDNN
from speechbrain.pretrained import WavLM

class SpeakerEncoder(nn.Module):
    def __init__(self, wavlm_model_hub='microsoft/wavlm-base', embedding_dim=192):
        super(SpeakerEncoder, self).__init__()
        wavlm_model_hub = "microsoft/wavlm-large"
        save_path = "./wavlmsavedir"
        self.wavlm = WavLM(wavlm_model_hub, save_path)
        self.ecapa_tdnn = ECAPA_TDNN(80, lin_neurons=192)

    def forward(self, audio):
        features = self.wavlm(audio).last_hidden_state
        features = features.permute(0, 2, 1)
        embeddings = self.ecapa_tdnn(features)
        return embeddings

# 数据预处理函数
def preprocess_audio(audio_path, target_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
    return waveform


# 额外的辅助函数和类
class WavLMDataset(Dataset):
    def __init__(self, audio_files, target_sample_rate=16000):
        self.audio_files = audio_files
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_file)
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)(waveform)
        return waveform

class WavLMPreprocessor:
    def __init__(self, target_sample_rate=16000):
        self.target_sample_rate = target_sample_rate

    def preprocess(self, audio_file):
        waveform, sample_rate = torchaudio.load(audio_file)
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)(waveform)
        return waveform

# 更多的模型定义
class ECAPA_TDNN_Extended(nn.Module):
    def __init__(self, input_dim=512, lin_neurons=192):
        super(ECAPA_TDNN_Extended, self).__init__()
        self.tdnn1 = nn.Conv1d(input_dim, 512, kernel_size=5, padding=2)
        self.tdnn2 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.tdnn3 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.tdnn4 = nn.Conv1d(512, 512, kernel_size=1)
        self.tdnn5 = nn.Conv1d(512, 1536, kernel_size=1)
        self.fc = nn.Linear(1536, lin_neurons)

    def forward(self, x):
        x = torch.relu(self.tdnn1(x))
        x = torch.relu(self.tdnn2(x))
        x = torch.relu(self.tdnn3(x))
        x = torch.relu(self.tdnn4(x))
        x = torch.relu(self.tdnn5(x))
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return x

class SpeakerEncoderExtended(nn.Module):
    def __init__(self, wavlm_model_name='microsoft/wavlm-base', embedding_dim=192):
        super(SpeakerEncoderExtended, self).__init__()
        wavlm_model_hub = "microsoft/wavlm-large"
        save_path = "./wavlmsavedir"
        self.wavlm = WavLM(wavlm_model_hub, save_path)
        self.ecapa_tdnn = ECAPA_TDNN_Extended(input_dim=self.wavlm.config.hidden_size, lin_neurons=embedding_dim)

    def forward(self, audio):
        features = self.wavlm(audio).last_hidden_state
        features = features.permute(0, 2, 1)
        embeddings = self.ecapa_tdnn(features)
        return embeddings
