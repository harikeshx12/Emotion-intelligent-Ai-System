import numpy as np
import librosa
import torch


class AudioPipeline:
def __init__(self, sr=16000, model_path=None, device='cpu'):
self.sr = sr
self.device = device
self.model = None
if model_path:
self.model = torch.load(model_path, map_location=device)
self.model.eval()


def extract_features(self, audio_np):
if audio_np is None:
return None
if audio_np.ndim>1:
audio_np = audio_np.mean(axis=1)
audio_resampled = librosa.resample(audio_np.astype(float), orig_sr=self.sr, target_sr=self.sr)
# mfcc
mfcc = librosa.feature.mfcc(audio_resampled, sr=self.sr, n_mfcc=13)
# pitch via librosa.yin (approx)
try:
f0 = librosa.yin(audio_resampled, fmin=50, fmax=500)
pitch = np.nanmean(f0)
except Exception:
pitch = 0.0
feat = np.concatenate([mfcc.mean(axis=1), [np.nan_to_num(pitch)]])
return feat


def get_embedding(self, audio_np):
feat = self.extract_features(audio_np)
if feat is None:
return None
if self.model is None:
return torch.tensor(feat).float()
with torch.no_grad():
inp = torch.tensor(feat).float().unsqueeze(0).to(self.device)
emb = self.model(inp)
return emb.squeeze(0).cpu()src/pipelines/audio_pipeline.py
