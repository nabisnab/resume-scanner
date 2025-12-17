"""
Application Configuration Settings
Provides configuration classes for different environments and application settings.
"""

import os
from datetime import timedelta


class Config:
    """Base configuration class with default settings."""
    
    # Application settings
    APP_NAME = "Resume Scanner"
    APP_VERSION = "1.0.0"
    DEBUG = False
    TESTING = False
    
    # Flask settings
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    JSON_SORT_KEYS = False
    
    # Database settings
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_RECORD_QUERIES = False
    
    # Session settings
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    
    # File upload settings
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    ALLOWED_EXTENSIONS = {"pdf", "doc", "docx", "txt"}
    
    # API settings
    API_VERSION = "v1"
    API_TITLE = "Resume Scanner API"
    API_DOCS_ENABLED = True
    
    # CORS settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    CORS_ALLOW_HEADERS = ["Content-Type", "Authorization"]
    CORS_ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Resume parsing settings
    MIN_RESUME_LENGTH = 100  # Minimum characters in a resume
    MAX_RESUME_LENGTH = 100000  # Maximum characters in a resume
    
    # AI/ML settings
    MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased")
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    
    # Pagination settings
    ITEMS_PER_PAGE = 20
    MAX_ITEMS_PER_PAGE = 100
    
    # Cache settings
    CACHE_TYPE = "simple"
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Timeout settings (in seconds)
    REQUEST_TIMEOUT = 30
    FILE_PROCESSING_TIMEOUT = 60


class DevelopmentConfig(Config):
    """Development environment configuration."""
    
    DEBUG = True
    TESTING = False
    SQLALCHEMY_ECHO = True
    SQLALCHEMY_RECORD_QUERIES = True
    SESSION_COOKIE_SECURE = False
    LOG_LEVEL = "DEBUG"
    API_DOCS_ENABLED = True


class TestingConfig(Config):
    """Testing environment configuration."""
    
    TESTING = True
    DEBUG = True
    SQLALCHEMY_ECHO = False
    SESSION_COOKIE_SECURE = False
    UPLOAD_FOLDER = "test_uploads"
    LOG_LEVEL = "DEBUG"
    
    # Use in-memory SQLite for testing
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


class ProductionConfig(Config):
    """Production environment configuration."""
    
    DEBUG = False
    TESTING = False
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_RECORD_QUERIES = False
    SESSION_COOKIE_SECURE = True
    LOG_LEVEL = "WARNING"
    API_DOCS_ENABLED = False
    
    # Production database URI should be set via environment variable
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "postgresql://user:password@localhost/resume_scanner"
    )


class StagingConfig(ProductionConfig):
    """Staging environment configuration."""
    
    DEBUG = False
    TESTING = False
    LOG_LEVEL = "INFO"
    API_DOCS_ENABLED = True


# Configuration factory
def get_config(env: str = None) -> Config:
    """
    Get configuration class based on environment.
    
    Args:
        env: Environment name ('development', 'testing', 'production', 'staging')
             If None, uses FLASK_ENV environment variable or defaults to 'development'
    
    Returns:
        Configuration class instance
    """
    if env is None:
        env = os.getenv("FLASK_ENV", "development").lower()
    
    config_map = {
        "development": DevelopmentConfig,
        "testing": TestingConfig,
        "production": ProductionConfig,
        "staging": StagingConfig,
    }
    
    return config_map.get(env, DevelopmentConfig)()
