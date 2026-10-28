import logging
import numpy as np
from consts import DEBUG
import time

logger = logging.getLogger(__name__)


def create_voice_clone_prompt(
    voice_audio,
    ref_wav_path,
    ref_audio_tokens_path,
):
    if DEBUG:
        ref_audio_tokens_path = f"{ref_audio_tokens_path}.npy"
        return _create_voice_clone_prompt_debug(
            voice_audio,
            ref_wav_path,
            ref_audio_tokens_path,
        )
        
    ref_audio_tokens_path = f"{ref_audio_tokens_path}.pt"
    return _create_voice_clone_prompt(
        voice_audio,
        ref_wav_path,
        ref_audio_tokens_path,
    )


def _create_voice_clone_prompt_debug(
    voice_audio,
    ref_wav_path,
    ref_audio_tokens_path,
):
    ref_wav = np.random.rand(1, 100_000).astype(np.float32)

    ref_rms = float(np.sqrt(np.mean(ref_wav**2)))

    if 0 < ref_rms < 0.1:
        ref_wav = ref_wav * 0.1 / ref_rms

    audio_length = int(ref_wav.shape[-1] / 24_000)

    chunk_size = 10
    clip_size = ref_wav.shape[-1] % chunk_size

    if clip_size > 0:
        ref_wav = ref_wav[:, :-clip_size]

    ref_audio_tokens = np.random.rand(8, 100).astype(np.float32)

    np.save(ref_wav_path, ref_wav)
    np.save(ref_audio_tokens_path, ref_audio_tokens)

    return audio_length, ref_rms, ref_audio_tokens, ref_audio_tokens_path


def _create_voice_clone_prompt(
    voice_audio,
    ref_wav_path,
    ref_audio_tokens_path,
):
    import torch
    with torch.inference_mode():
        from omnivoice.utils.audio import load_audio
        import state
    
        ref_wav = load_audio(voice_audio.path, state.model.sampling_rate)
        logger.info(f"Load Audio: {ref_wav.shape}")
    
        ref_rms = float(np.sqrt(np.mean(ref_wav**2)))
    
        if 0 < ref_rms < 0.1:
            ref_wav = ref_wav * 0.1 / ref_rms
    
        audio_length = int(ref_wav.shape[-1] / state.model.sampling_rate)
    
        chunk_size = state.model.audio_tokenizer.config.hop_length
        clip_size = ref_wav.shape[-1] % chunk_size
    
        if clip_size > 0:
            ref_wav = ref_wav[:, :-clip_size]
    
        logger.info(f"After Clip: {ref_wav.shape}")
    
        ref_wav_tensor = torch.from_numpy(ref_wav).to(
            state.model.audio_tokenizer.device
        )
        logger.info(f"Ref Wav Tensor: {ref_wav_tensor.shape}")
    
        ref_audio_tokens = state.model.audio_tokenizer.encode(
            ref_wav_tensor.unsqueeze(0),
        ).audio_codes.squeeze(0)
    
        logger.info(f"Audio Tokens: {ref_audio_tokens.shape}")
    
        np.save(ref_wav_path, ref_wav)
        torch.save(ref_audio_tokens, ref_audio_tokens_path)
    
        return audio_length, ref_rms, ref_audio_tokens, ref_audio_tokens_path