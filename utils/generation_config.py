from dataclasses import dataclass, fields
from typing import Optional, Union

@dataclass
class GenerationConfig:
    num_step: int = 32
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0
    pad_duration: float = 0.1
    fade_duration: float = 0.1

    @classmethod
    def from_dict(cls, kwargs_dict):
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs_dict.items() if k in valid_keys}
        return cls(**filtered)

@dataclass
class GenerationParameter:
    language: Union[str, list[str], None] = None
    instruct: Union[str, list[str], None] = None
    duration: Union[float, list[Optional[float]], None] = None
    speed: Union[float, list[Optional[float]], None] = None
    normalize_text: bool = False

    @classmethod
    def from_dict(cls, kwargs_dict: dict) -> "GenerationParameter":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs_dict.items() if k in valid_keys}
        return cls(**filtered)

