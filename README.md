# Offline WhisperX

Run different pipelines of WhisperX - _Transcription, Diarization, VAD, Alignment_ in **OFFLINE** mode.

### Pre-Trained Models

- [Systran/faster-whisper](https://huggingface.co/Systran)
- [pyannote diairzation 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- Wav2Vec2 - [PyTorch link](https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth)
- [VAD(Voice Activity Detection) Model](https://en.wikipedia.org/wiki/Voice_activity_detection) - download the model weights from [here](https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin).

### Usage

First download all the pre-trained models required for your task. Next, specify the paths to those pre-trained models in the `TranscriptionConfig` class.
