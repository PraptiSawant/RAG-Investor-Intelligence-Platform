import os
from urllib.parse import quote

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def get_engine(database: str | None = None):
    """
    Create PostgreSQL connection.
    
    Args:
        database: Database name. Defaults to POSTGRES_DATABASE from .env.
    """
    
    connection_string = os.getenv("DATABASE_URL")

    return create_engine(connection_string)
