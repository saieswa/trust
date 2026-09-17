import os
from dotenv import load_dotenv

# Locate the .env file in the backend directory (parent of app/)
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(backend_dir, ".env")
load_dotenv(dotenv_path)

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com")
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    UPSTASH_REDIS_REST_URL: str = os.getenv("UPSTASH_REDIS_REST_URL", "")
    UPSTASH_REDIS_REST_TOKEN: str = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
    REDIS_HOST: str = os.getenv("REDIS_HOST", "")
    REDIS_PORT: str = os.getenv("REDIS_PORT", "6379")
    REDIS_URL: str = os.getenv("REDIS_URL", "")

settings = Settings()
