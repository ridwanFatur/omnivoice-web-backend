import logging
import time
from dataclasses import asdict

import numpy as np

from consts import DEBUG
from models.voice_clone_prompt import VoiceClonePrompt
import state
from utils.generation_config import GenerationConfig, GenerationParameter

logger = logging.getLogger(__name__)


def generate_voice(
    text: str,
    voice_clone_prompt: VoiceClonePrompt,
    generation_config: GenerationConfig,
    generation_parameter: GenerationParameter,
) -> tuple[list, int]:
    if DEBUG:
        sample_rate = 24_000
        return [np.random.rand(1, 100_000).astype(np.float32), sample_rate]

    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
    from omnivoice.models.omnivoice import VoiceClonePrompt as _VoiceClonePrompt
    import torch
    
    with torch.inference_mode():
        total_start = time.perf_counter()
    
        logger.info("Start voice generation")
    
        # Load reference audio tokens
        stage_start = time.perf_counter()
        logger.info("Loading reference audio tokens")
    
        ref_audio_tokens = torch.load(
            voice_clone_prompt.ref_audio_tokens_path,
            map_location="cpu",
            weights_only=True,
        )
    
        logger.info(
            "Reference audio tokens loaded: shape=%s, duration=%.2fs",
            tuple(ref_audio_tokens.shape),
            time.perf_counter() - stage_start,
        )
    
        # Prepare voice clone prompt
        stage_start = time.perf_counter()
        logger.info("Preparing voice clone prompt")
    
        _voice_clone_prompt = _VoiceClonePrompt(
            ref_audio_tokens=ref_audio_tokens,
            ref_text=voice_clone_prompt.ref_text,
            ref_rms=voice_clone_prompt.ref_rms,
        )
    
        logger.info(
            "Voice clone prompt prepared in %.2fs",
            time.perf_counter() - stage_start,
        )
    
        # Prepare generation config
        stage_start = time.perf_counter()
        logger.info("Preparing generation config")
    
        gen_config = OmniVoiceGenerationConfig.from_dict(
            asdict(generation_config)
        )
    
        logger.info(
            "Generation config prepared in %.2fs",
            time.perf_counter() - stage_start,
        )
    
        # Preprocess generation task
        stage_start = time.perf_counter()
        logger.info("Preprocessing text and generation task")
    
        full_task = state.model._preprocess_all(
            text=text,
            language=generation_parameter.language,
            voice_clone_prompt=_voice_clone_prompt,
            instruct=generation_parameter.instruct,
            preprocess_prompt=gen_config.preprocess_prompt,
            speed=generation_parameter.speed,
            duration=generation_parameter.duration,
            normalize_text=generation_parameter.normalize_text,
        )
    
        logger.info(
            "Generation task preprocessed: batch_size=%s, duration=%.2fs",
            full_task.batch_size,
            time.perf_counter() - stage_start,
        )
    
        # Split short and long tasks
        stage_start = time.perf_counter()
        logger.info("Classifying tasks by length")
    
        short_idx, long_idx = full_task.get_indices(
            gen_config,
            state.model.audio_tokenizer.config.frame_rate,
        )
    
        logger.info(
            "Task classification completed: short=%d, long=%d, duration=%.2fs",
            len(short_idx),
            len(long_idx),
            time.perf_counter() - stage_start,
        )
    
        results = [None] * full_task.batch_size
    
        # Generate short tasks
        if short_idx:
            stage_start = time.perf_counter()
            logger.info("Generating short tasks: count=%d", len(short_idx))
    
            short_task = full_task.slice_task(short_idx)
            short_results = state.model._generate_iterative(
                short_task,
                gen_config,
            )
    
            for idx, res in zip(short_idx, short_results):
                results[idx] = res
    
            logger.info(
                "Short task generation completed: count=%d, duration=%.2fs",
                len(short_idx),
                time.perf_counter() - stage_start,
            )
    
        # Generate long tasks
        if long_idx:
            stage_start = time.perf_counter()
            logger.info("Generating long tasks: count=%d", len(long_idx))
    
            long_task = full_task.slice_task(long_idx)
            long_results = state.model._generate_chunked(
                long_task,
                gen_config,
            )
    
            for idx, res in zip(long_idx, long_results):
                results[idx] = res
    
            logger.info(
                "Long task generation completed: count=%d, duration=%.2fs",
                len(long_idx),
                time.perf_counter() - stage_start,
            )
    
        # Decode and post-process generated audio
        stage_start = time.perf_counter()
        logger.info("Decoding and post-processing generated audio")
    
        generated_audios = []
    
        for i in range(full_task.batch_size):
            assert results[i] is not None, f"Result {i} was not generated"
    
            audio = state.model._decode_and_post_process(
                results[i],
                full_task.ref_rms[i],
                gen_config,  # type: ignore[arg-type]
            )
    
            generated_audios.append(audio)
    
            logger.info(
                "Audio decoded: index=%d/%d, shape=%s",
                i + 1,
                full_task.batch_size,
                getattr(audio, "shape", None),
            )
    
        logger.info(
            "Audio decoding and post-processing completed in %.2fs",
            time.perf_counter() - stage_start,
        )
    
        sample_rate = state.model.sampling_rate
    
        logger.info(
			"Voice generation completed: sample_rate=%d, num_audios=%d, audio_shapes=%s, total_duration=%.2fs",
			sample_rate,
			len(generated_audios),
			[getattr(audio, "shape", None) for audio in generated_audios],
			time.perf_counter() - total_start,
		)
    
        return generated_audios, sample_rate
