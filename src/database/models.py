# src/database/models.py
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, Boolean, Float, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, DeclarativeBase
from typing import Optional
import os
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
class Base(DeclarativeBase):
    pass

class Request(Base):
    __tablename__ = "requests"
    
    id = Column(Integer, primary_key=True)
    file_path = Column(String, nullable=False)
    file_name = Column(String, nullable=False)
    file_size = Column(Integer)
    file_type = Column(String)
    meta_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc) )
    status = Column(String, default="pending")

    # Relationships
    processes = relationship("Process", back_populates="request")

class Process(Base):
    __tablename__ = "processes"
    
    id = Column(Integer, primary_key=True)
    request_id = Column(Integer, ForeignKey("requests.id"))
    model_id = Column(Integer, ForeignKey("models.id"))
    process_type = Column(String, nullable=False)  # 'ocr' or 'llm'
    started_at = Column(DateTime, default=datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    status = Column(String, default="pending")
    error = Column(Text, nullable=True)

    # Relationships
    request = relationship("Request", back_populates="processes")
    model = relationship("Model")
    responses = relationship("Response", back_populates="process")

class Response(Base):
    __tablename__ = "responses"
    
    id = Column(Integer, primary_key=True)
    process_id = Column(Integer, ForeignKey("processes.id"))
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    meta_data = Column(JSON, nullable=True)

    # Relationships
    process = relationship("Process", back_populates="responses")

class Model(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # 'ocr' or 'llm'
    # version = Column(String)
    parameters = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)

# Database connection utility
class DatabaseManager:
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        
    def init_db(self, test_mode: bool = False):
        # Get database URL from environment variables
        if test_mode:
            pass
        else:
            db_url = os.getenv("DATABASE_URL")
  
        
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create all tables
        Base.metadata.create_all(bind=self.engine)
        # logger.info("create all table")
    
    def get_session(self):
        if not self.SessionLocal:
            raise Exception("Database not initialized. Call init_db() first.")
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

# Create global database manager instance
db_manager = DatabaseManager()

#test connection
def test_database_connection():
    """Test basic database connectivity"""
    try:
        # Initialize database
        db_manager.init_db()
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False
    
test_database_connection()