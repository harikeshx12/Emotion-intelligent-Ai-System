# Emotion-intelligent-Ai-System

AI-based multimodal emotion detection system for EdTech applications.

## Project overview

This repository implements a real-time multimodal emotion detection system designed for educational technology (EdTech). The system fuses video (facial expressions), audio (vocal tone), and speech (transcript/sentiment) to infer learners’ emotional states such as engagement, confusion, frustration, and satisfaction. The goal is to enable emotion-aware learning platforms that adapt content delivery, provide analytics to educators, and improve learner outcomes.

Key capabilities:
- Real-time facial expression analysis from webcam or recorded video.
- Audio signal processing and prosody analysis for vocal emotion.
- Speech-to-text + sentiment analysis for semantic cues.
- Multimodal fusion model that outputs an overall emotion state and confidence.
- Tools for visualization, logging, and educator-facing analytics.

## Features

- Video: face detection, facial landmark extraction, expression features.
- Audio: loudness, pitch, spectral features, voice activity detection (VAD).
- Speech: speech recognition (ASR) + text-based sentiment features.
- Fusion: late- and mid-level fusion options; modular model interface.
- Demo: local demo app (notebook / web UI) to run live or on recorded sessions.
- Privacy-first design guidance and configurable logging levels.

## Architecture

High-level components:
1. Capture layer — webcam / microphone input, preprocessing, buffering.
2. Modality pipelines:
   - Video pipeline: face detection -> landmark + CNN features -> per-frame embedding
   - Audio pipeline: VAD -> frame-level features (MFCCs, pitch) -> temporal embedding
   - Speech pipeline: ASR -> token embeddings -> sentiment scores
3. Fusion layer — combines modality embeddings (concatenation, attention, or ensemble).
4. Classifier / inference model — outputs emotion class + confidence and metadata (timestamps).
5. Backend analytics & dashboard — aggregates session-level metrics for instructors.

Models and libraries used:
- CV: OpenCV, Dlib/MediaPipe, or a pretrained facial-expression CNN
- Audio: librosa, pyAudioAnalysis, torchaudio
- ASR & NLP: Whisper / Vosk / wav2vec + transformer-based sentiment models
- ML stack: PyTorch / TensorFlow, scikit-learn for baseline fusion

## Quick start (local)

Requirements
- Python 3.8+
- GPU recommended for model inference/training
- Example packages: opencv-python, dlib or mediapipe, librosa, torch, transformers

Install (example)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run demo (example)
```bash
# Start a live demo (if demo script exists)
python demo/live_demo.py --camera 0
# Or run on a sample video
python demo/run_on_video.py --input samples/session.mp4 --output outputs/annotated.mp4
```

Notes:
- Replace the demo commands with the actual script paths in this repo.
- For ASR, download the chosen model weights or configure API keys as outlined in the configuration section.

## Configuration

- config/default.yaml — contains paths to models, sampling rates, thresholds.
- Environment variables for keys (if using cloud ASR) and for toggling data logging.
- Logging levels: OFF, METADATA_ONLY, FULL (use METADATA_ONLY for privacy-preserving deployment).

## Data & Privacy

- Prefer on-device inference for sensitive deployments to avoid sending raw audio/video off-premise.
- When storing data, anonymize and downsample; store only extracted features where possible.
- Provide the user/learner with opt-in consent and clear retention policies.
- Include dataset provenance, and remove or mask PII in any exported reports.

## Model training & evaluation

- Modular training scripts:
  - train/video_train.py
  - train/audio_train.py
  - train/fusion_train.py
- Suggested evaluation metrics:
  - Accuracy, precision/recall per emotion class
  - F1-score, confusion matrix
  - Session-level metrics (engagement rate, confusion episodes per hour)
- Cross-validation across participants and sessions to avoid subject bias.

## Deployment

Options:
- On-device (edge) using optimized models (TorchScript, TensorRT) for privacy and low-latency.
- Containerized cloud deployment (Docker + Kubernetes) with autoscaling for many concurrent sessions.
- Web demo with WebRTC for capturing video/audio in browser and forwarding to backend (careful with privacy).

Example Dockerfile (not included): build a minimal image with only the inference artifacts.

## Limitations & Ethics

- Models can be biased by training data (age, gender, ethnicity). Evaluate fairness and test across subgroups.
- Emotion inference is probabilistic; avoid deterministic claims or decisions that could harm learners.
- Use as an assistive analytics tool for educators, not as an absolute measure of student worth or ability.
- Document limitations and provide human-in-the-loop safeguards.

## Contributing

- Please open issues for bugs or feature requests.
- Follow the repository's testing guidelines and add unit/integration tests.
- Create feature branches and open pull requests with clear descriptions and reproducible steps.

Suggested contribution workflow:
1. Fork -> feature branch -> tests -> PR -> review -> merge.

## Roadmap / Suggested next steps

- Add a reproducible demo (notebook and web UI).
- Provide pre-trained model checkpoints and a model card describing dataset and metrics.
- Add CI to run basic unit tests on PRs and a GitHub Action to build the Docker image.
- Create an automated evaluation suite with example datasets.

## Resources & references

- Papers and libraries for facial expression recognition, multimodal fusion, and ethical guidelines will be listed

## Contact

Maintainer: @harikeshx12
