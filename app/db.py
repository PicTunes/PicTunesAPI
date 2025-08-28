from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
from app.config import get_settings

settings = get_settings()

class Base(DeclarativeBase):
    pass

engine = None
SessionLocal: Optional[async_sessionmaker[AsyncSession]] = None

if settings.DATABASE_URL_ASYNC:
    engine = create_async_engine(
        settings.DATABASE_URL_ASYNC,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=5,
        max_overflow=10,
        future=True,
    )
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if SessionLocal is None:
        raise RuntimeError("DATABASE_URL_ASYNC is not configured.")
    async with SessionLocal() as session:
        yield session

async def check_db() -> bool:
    if engine is None:
        return False
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))
    return True