import numpy as np
import torch


class Fusion:
def __init__(self, method='late', device='cpu'):
self.method = method
self.device = device
# small MLP for mid-level fusion
if method == 'mid':
self.mlp = torch.nn.Sequential(
torch.nn.Linear(256, 128),
torch.nn.ReLU(),
torch.nn.Linear(128, 64),
).to(device)


def fuse(self, video_emb, audio_emb, text_feat):
# Normalize and concat
features = []
if video_emb is not None:
v = torch.tensor(video_emb).float() if not isinstance(video_emb, torch.Tensor) else video_emb
features.append(v.flatten())
if audio_emb is not None:
a = torch.tensor(audio_emb).float() if not isinstance(audio_emb, torch.Tensor) else audio_emb
features.append(a.flatten())
if text_feat is not None:
# text_feat is a dict from sentiment pipeline
t = torch.tensor([1.0 if text_feat.get('label','')=='POSITIVE' else 0.0, text_feat.get('score',0.0)])
features.append(t)
if not features:
return None
x = torch.cat([f if isinstance(f, torch.Tensor) else torch.tensor(f).float() for f in features]).float()
if self.method == 'mid':
# project or pad to 256
if x.numel() < 256:
x = torch.nn.functional.pad(x, (0, 256 - x.numel()))
