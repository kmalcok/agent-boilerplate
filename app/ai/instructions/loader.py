"""
TR: Agent talimatlari (system prompt) kod icine gomulmez; bu klasordeki `.md`
    dosyalarinda tutulur. Boylece prompt'lar versiyonlanabilir, gozden gecirilebilir
    ve kod degismeden guncellenebilir.
EN: Agent instructions (system prompts) are not hard-coded; they live in `.md`
    files in this folder. This makes prompts versionable, reviewable, and editable
    without touching code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# TR: Bu dosyanin bulundugu klasor = instructions koku.
# EN: The folder containing this file = the instructions root.
_INSTRUCTIONS_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_instructions(name: str) -> str:
    """
    TR: `<name>.md` dosyasini okur ve metnini dondurur. Sonuc onbeklenir.
    EN: Reads the `<name>.md` file and returns its text. The result is cached.

    Args:
        name: TR: Uzantisiz dosya adi (or. "triage"). / EN: File name without extension (e.g. "triage").
    """
    path = _INSTRUCTIONS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Instruction file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
