import time
import uuid

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from database import Base

class VoiceClonePrompt(Base):
    __tablename__ = "voice_clone_prompt"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    created_at: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=lambda: int(time.time() * 1000),
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    tags: Mapped[str] = mapped_column(String, nullable=False, default="")
    ref_audio_tokens_path: Mapped[str] = mapped_column(String, nullable=False)
    ref_wav_path: Mapped[str] = mapped_column(String, nullable=False)
    ref_text: Mapped[str | None] = mapped_column(String, nullable=True)
    ref_rms: Mapped[float] = mapped_column(Float, nullable=False)
    voice_audio_id: Mapped[str | None] = mapped_column(
        String(36),
		ForeignKey("voice_audio.id", ondelete="SET NULL"),
		nullable=True,
    )
    audio_length: Mapped[int] = mapped_column(Integer, nullable=False)
    ref_audio_tokens_shape: Mapped[str] = mapped_column(String, nullable=False)