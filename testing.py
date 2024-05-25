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
    compute_type="int8",
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
