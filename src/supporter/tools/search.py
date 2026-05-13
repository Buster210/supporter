from ..config import config
from ..logger import logger
from .base import ToolError


async def google_search(query: str) -> str:
    """Performs a Google Search to retrieve accurate, up-to-date internet data.
    Args:
        query: The search query string.
    Returns:
        Detailed answer compiled from search results, including source URLs.
    """
    logger.info(f"Tool: google_search — query='{query}'")

    from ..pool import get_provider

    provider = get_provider(live=True, model_name=config.gemini_live_fallback_model)

    try:
        result = await provider.generate(
            prompt=query,
            options={
                "use_search": True,
                "system_instruction": (
                    "You are a search expert. Provide a detailed, highly accurate "
                    "answer based on the search results. Include all relevant "
                    "facts, figures, and technical details. Format the output "
                    "to be consumed by another LLM."
                ),
            },
        )

        candidates = getattr(result.raw, "candidates", None)
        if not candidates:
            return result.text

        meta = getattr(candidates[0], "grounding_metadata", None)
        if not meta:
            return result.text

        sources = [
            f"- {getattr(chunk.web, 'title', 'Search Result')}: {chunk.web.uri}"
            for chunk in getattr(meta, "grounding_chunks", []) or []
            if getattr(chunk, "web", None) and getattr(chunk.web, "uri", None)
        ]

        if not sources:
            return result.text

        full_response = f"{result.text}\n\n\nSOURCES FOUND:\n" + "\n".join(sources)
        logger.info(
            f"Tool: google_search succeeded — text_len={len(result.text)}, "
            f"sources={len(sources)}"
        )
        logger.debug(f"Tool: google_search full sources: {sources!r}")
        return full_response

    except Exception as e:
        raise ToolError(f"Search failed for '{query}': {e}") from e
