import time
import uuid

from sqlalchemy import BigInteger, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class VoiceGeneration(Base):
    __tablename__ = "voice_generation"

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
    text: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    audio_length: Mapped[int] = mapped_column(Integer, nullable=False)
    voice_clone_prompt_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("voice_clone_prompt.id"),
        nullable=True,
    )
    generation_config: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )