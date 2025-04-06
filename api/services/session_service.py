import asyncio
import json
from config import supabase

async def save_session_to_supabase(session_id, messages):
    """
    Save the entire conversation to Supabase (table: session_history).
    """
    data = {
        "session_id": str(session_id),
        "messages": json.dumps(messages, indent=4)
    }
    await asyncio.to_thread(lambda: supabase.table("session_history").upsert(data).execute())

