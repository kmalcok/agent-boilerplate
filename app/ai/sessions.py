"""
TR: Oturum (session) fabrikasi. Konusma gecmisini SDK otomatik yonetsin diye
    yapilandirmaya gore uygun Session nesnesini uretir.
EN: Session factory. Produces the right Session object based on configuration so
    the SDK manages conversation history automatically.
"""

from __future__ import annotations

from agents.memory import Session, SQLiteSession

from app.config import SessionBackend, Settings


def build_session(session_id: str, settings: Settings) -> Session:
    """
    TR: `session_id` icin bir Session olusturur (kullanici/konusma basina bir ID).
    EN: Builds a Session for `session_id` (one ID per user/conversation).
    """
    backend = settings.session_backend

    if backend is SessionBackend.MEMORY:
        # TR: Process bitince kaybolur; demo/test icin idealdir. / EN: Lost when the process ends; ideal for demo/test.
        return SQLiteSession(session_id)

    if backend is SessionBackend.SQLITE:
        # TR: Dosya tabanli kalici SQLite. / EN: File-based persistent SQLite.
        return SQLiteSession(session_id, settings.session_db_path)

    if backend is SessionBackend.REDIS:
        # TR: Uretimde, worker'lar arasinda paylasimli bellek icin. / EN: For shared memory across workers in prod.
        #     `pip install openai-agents[redis]`
        from agents.extensions.memory import RedisSession

        return RedisSession.from_url(session_id, url=settings.redis_url)

    raise ValueError(f"Unsupported session backend: {backend}")
