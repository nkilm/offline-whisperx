# Offline WhisperX

Run different pipelines of WhisperX - _Transcription, Diarization, VAD, Alignment_ in **OFFLINE** mode.

### Pre-Trained Model(s) Download Links

- [Systran/faster-whisper](https://huggingface.co/Systran)
- [pyannote diairzation 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- Wav2Vec2 - [PyTorch link](https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth)
- [VAD(Voice Activity Detection) Model](https://en.wikipedia.org/wiki/Voice_activity_detection) - download the model weights from [here](https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin).

### NOTE

The script requires `whisperx==3.1.3` to work. Older versions of whisperx have import-related issues in some pieces of code. So before you run the script, please do install the correct packages along with version from [requirements.txt](./requirements.txt) file.

### Usage

First download all the pre-trained models required for your task. Next, specify the paths to those pre-trained models in the `TranscriptionConfig` class.

```python
from transcriber import Transcriber, TranscriptionConfig

# all required models are available in the models directory
MODELS_DIR = r"./models"

FASTER_WHISPER_PATH = f"{MODELS_DIR}/faster_whisper_base"
ALIGN_MODEL_DIR = f"{MODELS_DIR}/wav2vec2_base"
VAD_MODEL_FP = f"{MODELS_DIR}/voice_activity_detection/whisperx_vad_segmentation.bin"

PYANNOTE_CONFIG_PATH = "./pyannote_config.yaml"

# change the config as per your requirements
config = TranscriptionConfig(
    whisper_model_name=FASTER_WHISPER_PATH,  # provide whisper model name or path
    align_model_dir=ALIGN_MODEL_DIR,
    vad_model_fp=VAD_MODEL_FP,
    pyannote_config_path=PYANNOTE_CONFIG_PATH,
    compute_type="float8" if torch.cuda.is_available() else "int8",
    diarize=True,
    output_dir="./output",
    output_format="json",
)

audio_file = "./data/sample.wav"

transcriber = Transcriber(config)
transcriptions = transcriber.transcribe(audio_path=audio_file)
transcriptions = transcriber.align_transcriptions(transcriptions)
transcriptions = transcriber.diarize_transcriptions(transcriptions)
transcriber.write_transcriptions(transcriptions=transcriptions)
```
