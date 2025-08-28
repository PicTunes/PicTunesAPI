from pydantic import BaseModel, AnyHttpUrl, Field
from typing import List, Optional

class Music(BaseModel):
    title: str
    composer: str
    start: int
    end: int
    link: str

class SimilarItem(BaseModel):
    imageUrl: AnyHttpUrl
    score: float = Field(ge=0.0, le=1.0, description="Similarity in [0,1]")
    label: Optional[str] = None

class UploadResponse(BaseModel):
    label: str
    music: List[Music]
    similar: List[SimilarItem] = []
    videoUrl: Optional[AnyHttpUrl] = None

class MergeResponse(BaseModel):
    videoUrl: AnyHttpUrl
    duration: float