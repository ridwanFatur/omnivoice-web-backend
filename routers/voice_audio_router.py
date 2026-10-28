
import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy import func, select
from consts import ALLOWED_EXTENSIONS
from database import get_db
from models.voice_audio import VoiceAudio
import soundfile as sf
from sqlalchemy import func, or_, select

router = APIRouter(
    prefix="/api/voice-audio",
    tags=["Voice Audio"],
)

UPLOAD_DIR = (
    Path(__file__).resolve().parent.parent
    / "audio-files"
    / "uploaded-audio"
)

DbSession = Annotated[Session, Depends(get_db)]


class VoiceAudioUpdate(BaseModel):
    name: str | None = None
    tags: str | None = None


class VoiceAudioResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: int
    name: str
    tags: str
    path: str
    audio_length: int
    sample_rate: int


class VoiceAudioListResponse(BaseModel):
    items: list[VoiceAudioResponse]
    total: int
    page: int
    page_size: int

@router.post("/", response_model=VoiceAudioResponse, status_code=201)
def create_voice_audio(
    db: DbSession,
    file: Annotated[UploadFile, File()],
    name: Annotated[str, Form()],
    tags: Annotated[str, Form()] = "",
):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio extension: {extension or 'none'}",
        )

    audio_id = str(uuid.uuid4())
    filename = f"{audio_id}{extension}"

    relative_path = (
        Path("audio-files") / "uploaded-audio" / filename
    )
    destination = UPLOAD_DIR / filename

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with destination.open("wb") as output:
            shutil.copyfileobj(file.file, output)

        try:
            info = sf.info(destination)
        except (RuntimeError, ValueError, OSError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Unable to read audio file. Unsupported or invalid format.",
            ) from exc

        audio_length = round(info.duration * 1000)
        sample_rate = info.samplerate

        if audio_length <= 0 or sample_rate <= 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid audio duration or sample rate.",
            )

        audio = VoiceAudio(
            id=audio_id,
            name=name,
            tags=tags,
            path=relative_path.as_posix(),
            audio_length=audio_length,
            sample_rate=sample_rate,
        )

        db.add(audio)
        db.commit()
        db.refresh(audio)

        return audio

    except Exception:
        db.rollback()
        destination.unlink(missing_ok=True)
        raise

    finally:
        file.file.close()

@router.patch("/{audio_id}", response_model=VoiceAudioResponse)
def update_voice_audio(
    audio_id: str,
    payload: VoiceAudioUpdate,
    db: DbSession,
):
    audio = db.get(VoiceAudio, audio_id)

    if audio is None:
        raise HTTPException(
            status_code=404,
            detail="Voice audio not found",
        )

    updates = payload.model_dump(exclude_unset=True)

    if not updates:
        raise HTTPException(
            status_code=400,
            detail="At least one field (name or tags) must be provided",
        )

    for field, value in updates.items():
        if value is None:
            raise HTTPException(
                status_code=422,
                detail=f"{field} cannot be null",
            )
        setattr(audio, field, value)

    db.commit()
    db.refresh(audio)

    return audio


@router.delete("/{audio_id}", status_code=204)
def delete_voice_audio(
    audio_id: str,
    db: DbSession,
):
    audio = db.get(VoiceAudio, audio_id)

    if audio is None:
        raise HTTPException(
            status_code=404,
            detail="Voice audio not found",
        )

    file_path = Path(audio.path)
    if not file_path.is_absolute():
        file_path = Path(__file__).resolve().parent.parent / file_path

    try:
        db.delete(audio)
        db.commit()
    except Exception:
        db.rollback()
        raise

    file_path.unlink(missing_ok=True)


@router.get("/", response_model=VoiceAudioListResponse)
def list_voice_audio(
    db: DbSession,
    search: Annotated[str | None, Query()] = None,
    audio_length_gt: Annotated[int | None, Query(ge=0)] = None,
    audio_length_gte: Annotated[int | None, Query(ge=0)] = None,
    audio_length_lt: Annotated[int | None, Query(ge=0)] = None,
    audio_length_lte: Annotated[int | None, Query(ge=0)] = None,
    sample_rate_gt: Annotated[int | None, Query(ge=0)] = None,
    sample_rate_gte: Annotated[int | None, Query(ge=0)] = None,
    sample_rate_lt: Annotated[int | None, Query(ge=0)] = None,
    sample_rate_lte: Annotated[int | None, Query(ge=0)] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
):
    statement = select(VoiceAudio)
        
    if search and search.strip():
        keyword = f"%{search.strip()}%"
        statement = statement.where(
            or_(
                VoiceAudio.name.ilike(keyword),
                VoiceAudio.tags.ilike(keyword),
            )
        )

    if audio_length_gt is not None:
        statement = statement.where(
            VoiceAudio.audio_length > audio_length_gt
        )
    if audio_length_gte is not None:
        statement = statement.where(
            VoiceAudio.audio_length >= audio_length_gte
        )
    if audio_length_lt is not None:
        statement = statement.where(
            VoiceAudio.audio_length < audio_length_lt
        )
    if audio_length_lte is not None:
        statement = statement.where(
            VoiceAudio.audio_length <= audio_length_lte
        )

    if sample_rate_gt is not None:
        statement = statement.where(
            VoiceAudio.sample_rate > sample_rate_gt
        )
    if sample_rate_gte is not None:
        statement = statement.where(
            VoiceAudio.sample_rate >= sample_rate_gte
        )
    if sample_rate_lt is not None:
        statement = statement.where(
            VoiceAudio.sample_rate < sample_rate_lt
        )
    if sample_rate_lte is not None:
        statement = statement.where(
            VoiceAudio.sample_rate <= sample_rate_lte
        )

    count_statement = select(func.count()).select_from(
		statement.order_by(None).subquery()
	)

    total = db.scalar(count_statement) or 0

    statement = (
        statement
        .order_by(VoiceAudio.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    items = db.scalars(statement).all()

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }
