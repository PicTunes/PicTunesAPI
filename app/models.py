from sqlalchemy import String, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base

class ImageAsset(Base):
    __tablename__ = "image_assets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_path: Mapped[str | None] = mapped_column(String(512), nullable=True)