from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi import Depends

DATABASE_URL = "postgresql://pdfdb_x4xc_user:ecJPPDfIe6H9r3aToayzr1WuzC0CqJIf@dpg-d26460u3jp1c73cgkeo0-a.oregon-postgres.render.com/pdfdb_x4xc"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
