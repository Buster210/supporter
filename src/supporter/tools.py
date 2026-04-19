from .config import config
from .logger import logger


async def google_search(query: str) -> str:
    logger.info(f"Tool Execute: google_search(query='{query}')")

    target_model = config.gemini_model
    from .providers import GeminiProvider

    provider = GeminiProvider(
        api_key=config.gemini_api_keys[0], model_name=target_model
    )

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

        response_parts = [result.text]

        raw_response = result.raw
        if hasattr(raw_response, "candidates") and raw_response.candidates:
            candidate = raw_response.candidates[0]
            if (
                hasattr(candidate, "grounding_metadata")
                and candidate.grounding_metadata
            ):
                meta = candidate.grounding_metadata

                sources = []
                if hasattr(meta, "grounding_chunks") and meta.grounding_chunks:
                    for chunk in meta.grounding_chunks:
                        if hasattr(chunk, "web") and chunk.web:
                            title = getattr(chunk.web, "title", "Search Result")
                            url = getattr(chunk.web, "uri", "")
                            if url:
                                sources.append(f"- {title}: {url}")

                if sources:
                    response_parts.append("\n\nSOURCES FOUND:\n" + "\n".join(sources))

        full_response = "\n".join(response_parts)
        logger.debug(f"Tool Success: google_search returned {len(full_response)} chars")
        return full_response

    except Exception as e:
        logger.error(f"Tool Failure: google_search failed: {e}")
        return f"Error performing search: {e!s}"
