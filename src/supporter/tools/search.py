from ..config import config
from ..logger import logger
from .base import ToolError

_SEARCH_SYSTEM_INSTRUCTION = (
    "You are a search expert. Use Google Search to produce detailed, highly "
    "accurate answers grounded in real sources. Include relevant facts, "
    "figures, technical details, and the source URLs you used. Format the "
    "output to be consumed by another LLM."
)


async def google_search(query: str) -> str:
    """Degraded best-effort web lookup -- a FALLBACK, not the primary search tool.

    Prefer `browse` for all web search and research: it reaches real pages with
    far greater depth and reliability. Use google_search ONLY when browse is
    unavailable. The answer is synthesized by a separate Live LLM sub-provider,
    not raw search results, so it may paraphrase facts and omit or fabricate
    source URLs. Treat its output as unverified.

    Args:
        query: The search query string.

    Returns:
        An LLM-synthesized answer (best effort, unverified). Raises ToolError
        if the lookup fails.
    """
    logger.info(f"Tool: google_search — query='{query}'")

    from ..pool import get_provider

    provider = get_provider(
        live=True,
        model_name=config.gemini_live_fallback_model,
        system_instruction=_SEARCH_SYSTEM_INSTRUCTION,
    )

    try:
        result = await provider.generate(prompt=query)
        logger.info(f"Tool: google_search succeeded — text_len={len(result.text)}")
        return result.text

    except Exception as e:
        raise ToolError(f"Search failed for '{query}': {e}") from e
