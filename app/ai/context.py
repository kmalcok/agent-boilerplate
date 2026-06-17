"""
TR: Calisma boyunca tasinan yerel context (dependency injection).
    Bu nesne LLM'e GONDERILMEZ; sadece tool/hook/guardrail kodunuz okur.
    Buraya kullanici kimligi, izinler, logger, DB baglantisi gibi seyler konur.
EN: The local context carried throughout a run (dependency injection).
    This object is NOT sent to the LLM; only your tool/hook/guardrail code reads it.
    Put things like user identity, permissions, logger, DB handles here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AppContext:
    """TR: Istek basina uygulama context'i. / EN: Per-request application context."""

    # TR: Hangi kullanici? Yetki ve loglama icin kullanilir. / EN: Which user? Used for authz and logging.
    user_id: str = "anonymous"
    # TR: Kullanici premium mu? Ornek: bazi toollari kosullu acmak icin.
    # EN: Is the user premium? Example: to conditionally enable some tools.
    is_pro_user: bool = False
    # TR: Ornek: cok kiracili (multi-tenant) sistemlerde tenant izolasyonu.
    # EN: Example: tenant isolation in multi-tenant systems.
    tenant_id: str | None = None
    # TR: Calisma boyunca biriktirilen serbest metadata. / EN: Free-form metadata accumulated during the run.
    metadata: dict[str, str] = field(default_factory=dict)
