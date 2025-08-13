# app/db.py
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, Text
from app.config import settings

Base = declarative_base()
SessionLocal = None
engine = None

if settings.DB_URL:
    engine = create_engine(settings.DB_URL, pool_pre_ping=True, future=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

class ImageAsset(Base):
    __tablename__ = "image_assets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(512), nullable=False)         # public URL to the image
    label = Column(String(128), nullable=True)
    embedding_path = Column(String(512), nullable=True)  # .npy path for vector

def init_db():
    if engine is not None:
        Base.metadata.create_all(bind=engine)
