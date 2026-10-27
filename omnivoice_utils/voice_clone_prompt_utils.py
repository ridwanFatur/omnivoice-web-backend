import numpy as np
import logging
from consts import DEBUG

logger = logging.getLogger(__name__)


def load_audio(path: str) -> np.ndarray:
    if DEBUG:
        return np.random.rand(1, 100_000).astype(np.float32)
    else:
        from omnivoice.utils.audio import load_audio as _load_audio
        from omnivoice_utils.prepare_omnivoice import model
        
        ref_wav = _load_audio(path, model.sampling_rate)
        logger.info(f"Load Audio: {ref_wav.shape}")
        return ref_wav
        

def audio_tokenizer_encode(ref_wav):
    if DEBUG:
        chunk_size = 10
    else:
        from omnivoice_utils.prepare_omnivoice import model
        chunk_size = model.audio_tokenizer.config.hop_length
    
    clip_size = int(ref_wav.shape[-1] % chunk_size)
    ref_wav = ref_wav[:, :-clip_size] if clip_size > 0 else ref_wav
    logger.info(f"After Clip: {ref_wav.shape}")
    
    if DEBUG:
        ref_audio_tokens =  np.random.rand(
			8, 100
		).astype(np.float32)
    else:
        import torch
        from omnivoice_utils.prepare_omnivoice import model
        ref_wav_tensor = torch.from_numpy(ref_wav).to(model.audio_tokenizer.device)
        logger.info(f"Ref Wav Tensor: {ref_wav_tensor.shape}")
        
        ref_audio_tokens = model.audio_tokenizer.encode(
			ref_wav_tensor.unsqueeze(0),
		).audio_codes.squeeze(0)  # (C, T)
        logger.info(f"Audio Tokens: {ref_audio_tokens.shape}")
    
    return ref_wav, ref_audio_tokens

def create_voice_clone_prompt(voice_audio, ref_wav_path, ref_audio_tokens_path):
    ref_wav = load_audio(
        voice_audio.path,
    )

    ref_rms = float(np.sqrt(np.mean(ref_wav**2)))

    if 0 < ref_rms < 0.1:
        ref_wav = ref_wav * 0.1 / ref_rms
        
    if DEBUG:
        audio_length = int(ref_wav.shape[-1] / 24_000)
    else:
        from omnivoice_utils.prepare_omnivoice import model
        audio_length = int(ref_wav.shape[-1] / model.sampling_rate)

    ref_wav, ref_audio_tokens = audio_tokenizer_encode(ref_wav)

    np.save(ref_wav_path, ref_wav)
    
    if DEBUG:
        np.save(ref_audio_tokens_path, ref_audio_tokens)
    else:
        import torch
        torch.save(ref_audio_tokens, ref_audio_tokens_path)
    
    return audio_length, ref_rms, ref_audio_tokens