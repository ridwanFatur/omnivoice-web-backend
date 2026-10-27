import logging
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from consts import SAMPLE_RATE
from database import get_db
from models.voice_clone_prompt import VoiceClonePrompt
from models.voice_generation import VoiceGeneration
from omnivoice_utils.data_models import GenerationConfig, GenerationParameter
from omnivoice_utils.generate_audio_utils import generate_voice

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/generate-audio",
    tags=["Generate Audio"],
)

UPLOAD_DIR = (
    Path(__file__).resolve().parent.parent
    / "audio-files"
    / "generated-audio"
)

DbSession = Annotated[Session, Depends(get_db)]


class VoiceGenerationCreate(BaseModel):
    name: str
    voice_clone_prompt_id: str | None = None
    generation_config: dict[str, Any] = Field(default_factory=dict)
    generation_parameter: dict[str, Any] = Field(default_factory=dict)


class VoiceGenerationUpdate(BaseModel):
    name: str = Field(min_length=1)


def serialize_generation(generation: VoiceGeneration) -> dict:
    return {
        "id": generation.id,
        "created_at": generation.created_at,
        "name": generation.name,
        "path": generation.path,
        "audio_length": generation.audio_length,
        "voice_clone_prompt_id": generation.voice_clone_prompt_id,
        "generation_config": generation.generation_config,
    }


@router.post("/")
def create_voice_generation(
    payload: VoiceGenerationCreate,
    db: DbSession,
):
    voice_clone_prompt = None

    if payload.voice_clone_prompt_id:
        voice_clone_prompt = (
            db.query(VoiceClonePrompt)
            .filter(
                VoiceClonePrompt.id == payload.voice_clone_prompt_id
            )
            .first()
        )

        if voice_clone_prompt is None:
            raise HTTPException(
                status_code=404,
                detail="Voice clone prompt not found",
            )

    generation_parameter = GenerationParameter.from_dict(
        payload.generation_parameter
    )

    generation_config = GenerationConfig.from_dict(
        payload.generation_config
    )

    combined_config = {
        "generation_config": asdict(generation_config),
        "generation_parameter": asdict(generation_parameter),
    }

    generation_id = str(uuid.uuid4())
    audio_path = UPLOAD_DIR / f"gen-audio-{generation_id}.wav"

    generated_audio = generate_voice(
        voice_clone_prompt=voice_clone_prompt,
        generation_config=payload.generation_config,
        generation_parameter=generation_parameter,
    )

    generated_audio = np.asarray(
        generated_audio,
        dtype=np.float32,
    ).squeeze()

    if generated_audio.ndim != 1:
        raise HTTPException(
            status_code=500,
            detail="Generated audio must be a mono waveform",
        )

    audio_length = round(len(generated_audio) / SAMPLE_RATE)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    try:
        sf.write(
            str(audio_path),
            generated_audio,
            SAMPLE_RATE,
            format="WAV",
        )

        generation = VoiceGeneration(
            id=generation_id,
            created_at=int(time.time() * 1000),
            name=payload.name,
            path=str(audio_path),
            audio_length=audio_length,
            voice_clone_prompt_id=(
                voice_clone_prompt.id
                if voice_clone_prompt is not None
                else None
            ),
            generation_config=combined_config,
        )

        db.add(generation)
        db.commit()
        db.refresh(generation)

        return serialize_generation(generation)

    except Exception:
        db.rollback()
        audio_path.unlink(missing_ok=True)
        logger.exception("Failed to create voice generation")
        raise


@router.get("/")
def list_voice_generations(
    db: DbSession,
    search: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
):
    query = db.query(VoiceGeneration)

    if search and search.strip():
        query = query.filter(
            VoiceGeneration.name.ilike(f"%{search.strip()}%")
        )

    total = query.with_entities(
        func.count(VoiceGeneration.id)
    ).scalar() or 0

    items = (
        query.order_by(VoiceGeneration.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "items": [serialize_generation(item) for item in items],
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size,
        },
    }


@router.patch("/{generation_id}")
def update_voice_generation(
    generation_id: str,
    payload: VoiceGenerationUpdate,
    db: DbSession,
):
    generation = (
        db.query(VoiceGeneration)
        .filter(VoiceGeneration.id == generation_id)
        .first()
    )

    if generation is None:
        raise HTTPException(
            status_code=404,
            detail="Voice generation not found",
        )

    generation.name = payload.name.strip()

    if not generation.name:
        raise HTTPException(
            status_code=422,
            detail="Name cannot be empty",
        )

    db.commit()
    db.refresh(generation)

    return serialize_generation(generation)


@router.delete("/{generation_id}", status_code=204)
def delete_voice_generation(
    generation_id: str,
    db: DbSession,
):
    generation = (
        db.query(VoiceGeneration)
        .filter(VoiceGeneration.id == generation_id)
        .first()
    )

    if generation is None:
        raise HTTPException(
            status_code=404,
            detail="Voice generation not found",
        )

    audio_path = Path(generation.path)

    try:
        db.delete(generation)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to delete voice generation")
        raise

    try:
        audio_path.unlink(missing_ok=True)
    except OSError:
        logger.exception(
            "Record deleted, but audio file could not be removed: %s",
            audio_path,
        )

    return Response(status_code=204)