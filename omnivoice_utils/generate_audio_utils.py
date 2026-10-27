import numpy as np

from consts import DEBUG
from models.voice_clone_prompt import VoiceClonePrompt
from omnivoice_utils.data_models import GenerationConfig, GenerationParameter

def generate_voice(
    voice_clone_prompt: VoiceClonePrompt,
    generation_config: GenerationConfig,
    generation_parameter: GenerationParameter,
) -> np.ndarray:
    if DEBUG:
        return np.random.rand(1, 100_000).astype(np.float32)
    else:
        return np.random.rand(1, 100_000).astype(np.float32)