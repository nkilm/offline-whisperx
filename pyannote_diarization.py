from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from pyannote.audio import Pipeline
from whisperx.audio import SAMPLE_RATE, load_audio

""" 
Offline pyannote diarization- 
https://stackoverflow.com/questions/76769776/way-to-offline-speaker-diarization-with-hugging-face
"""


class DiarizationPipeline:
    def __init__(
        self,
        config_path=None,  # path to .yaml config file
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)

        # load model from config file
        self.model = Pipeline.from_pretrained(config_path).to(device)

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    ):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            "waveform": torch.from_numpy(audio[None, :]),
            "sample_rate": SAMPLE_RATE,  # 16KHz
        }
        segments = self.model(
            audio_data,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        diarize_df = pd.DataFrame(
            segments.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
        diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)
        return diarize_df
