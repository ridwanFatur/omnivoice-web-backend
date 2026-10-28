import numpy as np

from consts import DEBUG
from models.voice_clone_prompt import VoiceClonePrompt
from omnivoice_utils.data_models import GenerationConfig, GenerationParameter
from dataclasses import asdict
from omnivoice_utils.prepare_omnivoice import model

def generate_voice(
    text: str,
    voice_clone_prompt: VoiceClonePrompt,
    generation_config: GenerationConfig,
    generation_parameter: GenerationParameter,
) -> np.ndarray:
    if DEBUG:
        return np.random.rand(1, 100_000).astype(np.float32)
    else:
        from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
        from omnivoice.models.omnivoice import VoiceClonePrompt as _VoiceClonePrompt
        import torch
        
        ref_audio_tokens = torch.load(
			voice_clone_prompt.ref_audio_tokens_path,
			map_location="cpu",
			weights_only=True,
		)
        
        _voice_clone_prompt = _VoiceClonePrompt(
            ref_audio_tokens=ref_audio_tokens,
            ref_text=voice_clone_prompt.ref_text,
            ref_rms=voice_clone_prompt.ref_rms,
        )
        
        gen_config = OmniVoiceGenerationConfig.from_dict(asdict(generation_config))       
        full_task = model._preprocess_all(
            text=text,
            language=generation_parameter.language,
            voice_clone_prompt=_voice_clone_prompt,
            instruct=generation_parameter.instruct,
            preprocess_prompt=gen_config.preprocess_prompt,
            speed=generation_parameter.speed,
            duration=generation_parameter.duration,
            normalize_text=generation_parameter.normalize_text,
		)
        
        short_idx, long_idx = full_task.get_indices(
            gen_config, model.audio_tokenizer.config.frame_rate
        )

        results = [None] * full_task.batch_size

        if short_idx:
            short_task = full_task.slice_task(short_idx)
            short_results = model._generate_iterative(short_task, gen_config)
            for idx, res in zip(short_idx, short_results):
                results[idx] = res

        if long_idx:
            long_task = full_task.slice_task(long_idx)
            long_results = model._generate_chunked(long_task, gen_config)
            for idx, res in zip(long_idx, long_results):
                results[idx] = res

        generated_audios = []
        for i in range(full_task.batch_size):
            assert results[i] is not None, f"Result {i} was not generated"
            generated_audios.append(
                model._decode_and_post_process(
                    results[i],
                    full_task.ref_rms[i],
                    gen_config,  # type: ignore[arg-type]
                )
            )
        return generated_audios[0]