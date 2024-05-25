"""
Wrapper built around the WhisperX library to transcribe audio files in OFFLINE mode.
The transcriber can be used to transcribe audio files, align the transcriptions, diarize the transcriptions.

pre-trained models required:
- faster-whisper
- pyannote diairzation 3.1
- wav2vec2-base: for alignment
    - Download link - https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth
- VAD(Voice Activity Detection) model
    
whisperx version: 3.1.3
"""


import gc
import os

import numpy as np
import torch
from whisperx.alignment import align, load_align_model
from whisperx.asr import load_model
from whisperx.audio import load_audio
from whisperx.diarize import assign_word_speakers
from whisperx.utils import get_writer

from pyannote_diarization import DiarizationPipeline


class TranscriptionConfig:
    def __init__(
        self,
        whisper_model_name="base",
        whisper_download_root=None,
        device=None,
        device_index=0,
        batch_size=8,
        compute_type="float16",
        output_dir=".",
        output_format="all",
        verbose=True,
        task="transcribe",
        language=None,
        align_model=None,
        interpolate_method="nearest",
        no_align=False,
        return_char_alignments=False,
        vad_onset=0.500,
        vad_offset=0.363,
        chunk_size=30,
        diarize=False,
        min_speakers=None,
        max_speakers=None,
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=1.0,
        length_penalty=1.0,
        suppress_tokens="-1",
        suppress_numerals=False,
        initial_prompt=None,
        condition_on_previous_text=False,
        fp16=True,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        max_line_width=None,
        max_line_count=None,
        highlight_words=False,
        segment_resolution="sentence",
        threads=0,
        hf_token=None,
        print_progress=False,
        align_model_dir=None,
        vad_model_fp=None,
        pyannote_config_path=None,
    ):
        self.whisper_model_name = whisper_model_name
        self.whisper_download_root = whisper_download_root
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_index = device_index
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.output_dir = output_dir
        self.output_format = output_format
        self.verbose = verbose
        self.task = task
        self.language = language
        self.align_model = align_model
        self.interpolate_method = interpolate_method
        self.no_align = no_align
        self.return_char_alignments = return_char_alignments
        self.vad_onset = vad_onset
        self.vad_offset = vad_offset
        self.chunk_size = chunk_size
        self.diarize = diarize
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.temperature = temperature
        self.best_of = best_of
        self.beam_size = beam_size
        self.patience = patience
        self.length_penalty = length_penalty
        self.suppress_tokens = suppress_tokens
        self.suppress_numerals = suppress_numerals
        self.initial_prompt = initial_prompt
        self.condition_on_previous_text = condition_on_previous_text
        self.fp16 = fp16
        self.temperature_increment_on_fallback = temperature_increment_on_fallback
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.max_line_width = max_line_width
        self.max_line_count = max_line_count
        self.highlight_words = highlight_words
        self.segment_resolution = segment_resolution
        self.threads = threads
        self.hf_token = hf_token
        self.print_progress = print_progress
        self.align_model_dir = align_model_dir
        self.vad_model_fp = vad_model_fp
        self.pyannote_config_path = pyannote_config_path


class Transcriber:
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.device = self.config.device
        self.faster_whisper_threads = 4
        if self.config.threads > 0:
            torch.set_num_threads(self.config.threads)
            self.faster_whisper_threads = self.config.threads

    def load_asr_model(self):
        return load_model(
            whisper_arch=self.config.whisper_model_name,
            device=self.config.device,
            device_index=self.config.device_index,
            download_root=self.config.whisper_download_root,
            compute_type=self.config.compute_type,
            language=self.config.language,
            asr_options=self.get_asr_options(),
            vad_options={
                "vad_onset": self.config.vad_onset,
                "vad_offset": self.config.vad_offset,
            },
            task=self.config.task,
            threads=self.faster_whisper_threads,
            vad_model_fp=self.config.vad_model_fp,
        )

    def get_asr_options(self):
        temperature = self.config.temperature
        if (increment := self.config.temperature_increment_on_fallback) is not None:
            temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
        else:
            temperature = [temperature]

        return {
            "beam_size": self.config.beam_size,
            "patience": self.config.patience,
            "length_penalty": self.config.length_penalty,
            "temperatures": temperature,
            "compression_ratio_threshold": self.config.compression_ratio_threshold,
            "log_prob_threshold": self.config.logprob_threshold,
            "no_speech_threshold": self.config.no_speech_threshold,
            "condition_on_previous_text": self.config.condition_on_previous_text,
            "initial_prompt": self.config.initial_prompt,
            "suppress_tokens": [int(x) for x in self.config.suppress_tokens.split(",")],
            "suppress_numerals": self.config.suppress_numerals,
        }

    def transcribe(self, audio_path):
        model = self.load_asr_model()
        results = []

        audio = load_audio(audio_path)
        result = model.transcribe(
            audio,
            batch_size=self.config.batch_size,
            chunk_size=self.config.chunk_size,
            print_progress=self.config.print_progress,
        )
        results.append((result, audio_path))

        del model
        gc.collect()
        torch.cuda.empty_cache()

        return results

    def align_transcriptions(self, transcriptions):
        if self.config.no_align:
            return transcriptions

        align_language = self.config.language or "en"
        align_model, align_metadata = load_align_model(
            align_language,
            self.device,
            model_name=self.config.align_model,
            model_dir=self.config.align_model_dir,
        )

        aligned_results = []

        for result, audio_path in transcriptions:
            if len(transcriptions) > 1:
                input_audio = audio_path
            else:
                input_audio = load_audio(audio_path)

            if align_model is not None and len(result["segments"]) > 0:
                if result.get("language", "en") != align_metadata["language"]:
                    align_model, align_metadata = load_align_model(
                        result["language"], self.device
                    )
                result = align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    input_audio,
                    self.device,
                    interpolate_method=self.config.interpolate_method,
                    return_char_alignments=self.config.return_char_alignments,
                    print_progress=self.config.print_progress,
                )

            aligned_results.append((result, audio_path))

        del align_model
        gc.collect()
        torch.cuda.empty_cache()

        return aligned_results

    def diarize_transcriptions(self, transcriptions):
        if not self.config.diarize:
            return transcriptions

        diarize_model = DiarizationPipeline(
            config_path=self.config.pyannote_config_path
        )
        diarized_results = []

        for result, audio_path in transcriptions:
            diarize_segments = diarize_model(
                audio_path,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers,
            )
            result = assign_word_speakers(diarize_segments, result)
            diarized_results.append((result, audio_path))

        return diarized_results

    def write_transcriptions(self, transcriptions):
        writer = get_writer(self.config.output_format, self.config.output_dir)
        writer_args = {
            "highlight_words": self.config.highlight_words,
            "max_line_count": self.config.max_line_count,
            "max_line_width": self.config.max_line_width,
        }

        for result, audio_path in transcriptions:
            result["language"] = self.config.language or "en"
            writer(result, audio_path, writer_args)
