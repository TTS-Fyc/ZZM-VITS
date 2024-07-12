import torch 
from torch.nn import functional as F
import torch.nn as nn

import commons
import numpy as np
import librosa
from speaker_encoder import WavLMModel as wavlm_features
from speaker_encoder import SpeakerEncoder as speaker_embedding
from speechbrain.pretrained import DiscreteHuBERT

# 初始化 DiscreteHuBERT 模型
model = DiscreteHuBERT.from_hparams(source="speechbrain/hubert-discrete", savedir="tmpdir")

# 加载音频文件
audio_file = "path/to/your/audio/file.wav"
waveform, sample_rate = torchaudio.load(audio_file)

# 对音频进行处理，获取离散的表示
discrete_units = model.encode_batch(waveform)

print(discrete_units)


def speaker_embedding(x)

def feature_loss(fmap_r, fmap_g):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2 


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = torch.mean((1-dr)**2)
    g_loss = torch.mean(dg**2)
    loss += (r_loss + g_loss)
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


def generator_loss(disc_outputs):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1-dg)**2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()

  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l




def cosine_similarity(x1, x2):
    return F.cosine_similarity(x1, x2)

def L_WSL(g, h, alpha):
    n = g.size(0)  # 样本数量

    phi_g = wavlm_features(g)
    phi_h = wavlm_features(h)
    varphi_g = speaker_embedding(g)
    varphi_h = speaker_embedding(h)

    cos_sim_wavlm = cosine_similarity(phi_g, phi_h)
    cos_sim_speaker = cosine_similarity(varphi_g, varphi_h)

    loss = -alpha / n * torch.sum((cos_sim_wavlm + cos_sim_speaker) / 2)
    return loss




def mcd_loss(tar_path, ref_path, sr=22050, n_mfcc=13):
    y1, sr1 = librosa.load(tar_path, sr=sr)
    y2, sr2 = librosa.load(ref_path, sr=sr)

    if sr1 != sr2:
        raise ValueError("两个音频文件的采样率必须相同")

    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=n_mfcc)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=n_mfcc)

    if mfcc1.shape != mfcc2.shape:
        raise ValueError("两个 MFCC 特征序列的形状必须相同")

    diff = mfcc1 - mfcc2
    dist = np.sqrt((diff ** 2).sum(axis=0)).mean()
    l_mcd = (10 / np.log(10)) * np.sqrt(2) * dist
    return l_mcd

def huber_loss(tar_path, ref_path, sr=22050)
    y1, sr1 = librosa.load(tar_path, sr=sr)
    y2, sr2 = librosa.load(ref_path, sr=sr)

    if sr1 != sr2:
        raise ValueError("两个音频文件的采样率必须相同")

    model_hub = "./huber-base-ls960"
    save_path = "./hubertsavedir"
    ssl_layer_num = -1
    kmeasns_repo_id = "speechbrain/SSL_Quantization"
    kmeans_filename = "LibriSpeech_hubert_k128_L7.pt"
    kmeans_cache_dir="hubertsavedir"
    model = DiscreteHuBERT(model_hub, save_path,freeze = True,ssl_layer_num=ssl_layer_num,kmeans_repo_id=kmeans_repo_id, kmeans_filename=kmeans_filename, kmeans_cache_dir=kmeans_cache_dir)
    tar_embs, tar_tokens = model(y1)
    ref_embs, ref_tokens = model(y2)

    criterion = nn.MSELoss()

    loss = criterion(tar_embs, ref_embs)
    return loss
