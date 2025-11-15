import sqlite3
from contextlib import contextmanager
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

DB_PATH = "conversations.db"


def init_db():
    """Initialize the SQLite database with conversations table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """    
        CREATE TABLE IF NOT EXISTS conversations (    
            id INTEGER PRIMARY KEY AUTOINCREMENT,  
            role TEXT NOT NULL,  
            content TEXT NOT NULL,  
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  
        )    
        """
    )
    conn.commit()
    conn.close()


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# Initialize database on startup
init_db()


class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


@app.post("/conversation")
async def save_message(message: ConversationMessage):
    """Save a conversation message."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """    
            INSERT INTO conversations (role, content)    
            VALUES (?, ?)    
            """,
            (message.role, message.content),
        )
        conn.commit()
        message_id = cursor.lastrowid

    return {"status": "saved", "message_id": message_id}


@app.get("/conversation")
async def get_conversation():
    """Get all conversation messages."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """  
            SELECT role, content, created_at   
            FROM conversations   
            ORDER BY created_at ASC  
            """
        )
        rows = cursor.fetchall()

        messages = [
            {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    return {"messages": messages, "count": len(messages)}


@app.get("/conversation/stats")
async def conversation_stats():
    """Get conversation statistics."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM conversations")
        total_messages = cursor.fetchone()["count"]

    return {"total_messages": total_messages}


@app.delete("/conversation")
async def clear_conversation():
    """Clear all conversation messages."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversations")
        conn.commit()
        deleted_count = cursor.rowcount

    return {"status": "cleared", "deleted_messages": deleted_count}
