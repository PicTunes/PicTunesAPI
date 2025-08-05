from sqlalchemy import Column, Integer, String, DateTime, Text, LargeBinary
from sqlalchemy.sql import func
from app.core.database import Base

class ProcessedImage(Base):
    __tablename__ = "processed_images"
    
    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String(255), nullable=False)
    processed_filename = Column(String(255), nullable=False)
    user_id = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class MediaFile(Base):
    __tablename__ = "media_files"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # 'image', 'audio', 'video'
    file_size = Column(Integer, nullable=False)
    user_id = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
