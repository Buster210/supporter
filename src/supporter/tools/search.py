from ..config import config
from ..logger import logger


async def google_search(query: str) -> str:
    logger.info(f"Tool Execute: google_search(query='{query}')")

    from ..index import get_provider

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

        sources = []
        grounding_chunks = getattr(meta, "grounding_chunks", []) or []
        for chunk in grounding_chunks:
            web = getattr(chunk, "web", None)
            if not web:
                continue

            url = getattr(web, "uri", "")
            if url:
                title = getattr(web, "title", "Search Result")
                sources.append(f"- {title}: {url}")

        if not sources:
            return result.text

        full_response = f"{result.text}\n\n\nSOURCES FOUND:\n" + "\n".join(sources)
        logger.debug(f"Tool Success: google_search returned {len(full_response)} chars")
        return full_response

    except Exception as e:
        logger.error(f"Tool Failure: google_search failed: {e}")
        return f"Error performing search: {e!s}"
