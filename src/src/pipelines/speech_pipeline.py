import torch
from transformers import pipeline


class SpeechPipeline:
def __init__(self, asr_model_name='openai/whisper-small', device='cpu'):
self.device = device
# using transformers pipeline for ASR if available; if not, user can plug Whisper offline
try:
self.asr = pipeline("automatic-speech-recognition", model=asr_model_name, device=0 if device!='cpu' else -1)
except Exception as e:
print('ASR pipeline init failed:', e)
self.asr = None
# sentiment model (text)
try:
self.sentiment = pipeline("sentiment-analysis", device=0 if device!='cpu' else -1)
except Exception as e:
print('Sentiment pipeline init failed:', e)
self.sentiment = None


def transcribe(self, audio_path_or_array):
if self.asr is None:
return ""
try:
out = self.asr(audio_path_or_array)
return out.get('text','')
except Exception as e:
print('ASR error:', e)
return ""


def analyze_text(self, text):
if not text or self.sentiment is None:
return {'label':'NEUTRAL', 'score': 0.0}
out = self.sentiment(text[:512])
if isinstance(out, list) and out:
return out[0]
return {'label':'NEUTRAL', 'score': 0.0}
