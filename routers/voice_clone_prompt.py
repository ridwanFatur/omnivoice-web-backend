import uuid
from pathlib import Path
from typing import Annotated
import shutil
from fastapi import APIRouter, Depends, Form, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from database import get_db
from models.voice_audio import VoiceAudio
from models.voice_clone_prompt import VoiceClonePrompt
from omnivoice_utils.voice_clone_prompt_utils import create_voice_clone_prompt
from consts import DEBUG

router = APIRouter(
    prefix="/api/voice-clone-prompt",
    tags=["Voice Clone Prompt"],
)

UPLOAD_DIR = (
    Path(__file__).resolve().parent.parent
    / "audio-files"
    / "voice-clone-prompt"
)


DbSession = Annotated[Session, Depends(get_db)]

@router.post("/")
def generate_voice_clone_prompt(
    db: DbSession,
    name: Annotated[str, Form()],
    voice_audio_id: Annotated[str, Form()],
    tags: Annotated[str, Form()] = "",
    ref_text: Annotated[str | None, Form()] = None,
):
    voice_audio = db.get(VoiceAudio, voice_audio_id)

    if voice_audio is None:
        raise HTTPException(
            status_code=404,
            detail="Voice audio not found",
        )

    prompt_id = str(uuid.uuid4())
    prompt_dir = UPLOAD_DIR / prompt_id
    prompt_dir.mkdir(parents=True, exist_ok=True)

    ref_wav_path = prompt_dir / f"ref_wav-{prompt_id}.npy"
    
    if DEBUG:
        ref_audio_tokens_path = (
			prompt_dir / f"ref_audio_tokens-{prompt_id}.npy"
		)
    else:
        ref_audio_tokens_path = (
			prompt_dir / f"ref_audio_tokens-{prompt_id}.pt"
		)       

    try:
        audio_length, ref_rms, ref_audio_tokens = create_voice_clone_prompt(voice_audio, ref_wav_path, ref_audio_tokens_path)
        prompt = VoiceClonePrompt(
            id=prompt_id,
            name=name,
            tags=tags,
            ref_audio_tokens_path=str(ref_audio_tokens_path),
            ref_wav_path=str(ref_wav_path),
            ref_text=ref_text,
            ref_rms=ref_rms,
            voice_audio_id=voice_audio.id,
            audio_length=audio_length,
            ref_audio_tokens_shape=str(
                list(ref_audio_tokens.shape)
            ),
        )

        db.add(prompt)
        db.commit()
        db.refresh(prompt)

        return {
            "id": prompt.id,
            "created_at": prompt.created_at,
            "name": prompt.name,
            "tags": prompt.tags,
            "ref_audio_tokens_path": prompt.ref_audio_tokens_path,
            "ref_wav_path": prompt.ref_wav_path,
            "ref_text": prompt.ref_text,
            "ref_rms": prompt.ref_rms,
            "voice_audio_id": prompt.voice_audio_id,
            "audio_length": prompt.audio_length,
            "ref_audio_tokens_shape": prompt.ref_audio_tokens_shape,
        }

    except Exception:
        db.rollback()

        if prompt_dir.exists():
            import shutil
            shutil.rmtree(prompt_dir)

        raise
    
@router.get("/")
def list_voice_clone_prompts(
    db: DbSession,
    search: Annotated[str | None, Query()] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
):
    query = db.query(VoiceClonePrompt)

    if search and search.strip():
        keyword = f"%{search.strip()}%"
        query = query.filter(
            or_(
                VoiceClonePrompt.name.ilike(keyword),
                VoiceClonePrompt.tags.ilike(keyword),
            )
        )

    total = query.with_entities(
        func.count(VoiceClonePrompt.id)
    ).scalar() or 0

    prompts = (
        query
        .order_by(VoiceClonePrompt.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    return {
        "items": [
            {
                "id": prompt.id,
                "created_at": prompt.created_at,
                "name": prompt.name,
                "tags": prompt.tags,
                "ref_audio_tokens_path": prompt.ref_audio_tokens_path,
                "ref_wav_path": prompt.ref_wav_path,
                "ref_text": prompt.ref_text,
                "ref_rms": prompt.ref_rms,
                "voice_audio_id": prompt.voice_audio_id,
                "audio_length": prompt.audio_length,
                "ref_audio_tokens_shape": prompt.ref_audio_tokens_shape,
            }
            for prompt in prompts
        ],
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit,
    }


@router.patch("/{prompt_id}")
def update_voice_clone_prompt(
    prompt_id: str,
    db: DbSession,
    name: Annotated[str | None, Form()] = None,
    tags: Annotated[str | None, Form()] = None,
):
    prompt = db.get(VoiceClonePrompt, prompt_id)

    if prompt is None:
        raise HTTPException(
            status_code=404,
            detail="Voice clone prompt not found",
        )

    if name is None and tags is None:
        raise HTTPException(
            status_code=400,
            detail="At least one field (name or tags) must be provided",
        )

    if name is not None:
        prompt.name = name

    if tags is not None:
        prompt.tags = tags

    db.commit()
    db.refresh(prompt)

    return {
        "id": prompt.id,
        "created_at": prompt.created_at,
        "name": prompt.name,
        "tags": prompt.tags,
        "ref_audio_tokens_path": prompt.ref_audio_tokens_path,
        "ref_wav_path": prompt.ref_wav_path,
        "ref_text": prompt.ref_text,
        "ref_rms": prompt.ref_rms,
        "voice_audio_id": prompt.voice_audio_id,
        "audio_length": prompt.audio_length,
        "ref_audio_tokens_shape": prompt.ref_audio_tokens_shape,
    }


@router.delete("/{prompt_id}")
def delete_voice_clone_prompt(
    prompt_id: str,
    db: DbSession,
):
    prompt = db.get(VoiceClonePrompt, prompt_id)

    if prompt is None:
        raise HTTPException(
            status_code=404,
            detail="Voice clone prompt not found",
        )

    prompt_dir = UPLOAD_DIR / prompt.id

    try:
        db.delete(prompt)
        db.commit()

        if prompt_dir.exists():
            shutil.rmtree(prompt_dir)

        return {
            "message": "Voice clone prompt deleted successfully",
            "id": prompt_id,
        }

    except Exception:
        db.rollback()
        raise