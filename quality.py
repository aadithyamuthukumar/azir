"""Historical quality estimates for `azir-auto` routing.

Estimates come only from persisted LLM-as-a-judge scores
(`response_evaluations`, see judge.py) -- nothing is judged during routing.
A score persisted for one request can influence later routing decisions.
"""

import asyncio
import logging

logger = logging.getLogger("azir.quality")

# Fewer evaluations than this aren't trusted: one or two judge scores say
# little about a model.
MIN_QUALITY_SAMPLES = 5

# Quality assumed without enough history (and when the lookup fails):
# neutral, so an untried model is neither punished nor rewarded.
DEFAULT_QUALITY_SCORE = 0.5

# Upper bound on the quality lookup, which runs before every `quality` or
# `balanced` `azir-auto` request: the most a slow or unreachable database
# can add to routing.
QUALITY_LOOKUP_TIMEOUT_SECONDS = 0.5


def quality_from_history(row: dict | None, task: str | None) -> float:
    """One model's estimate from its aggregated history:

    1. its average for `task`, with >= MIN_QUALITY_SAMPLES evaluations for that task
    2. else its overall average, with >= MIN_QUALITY_SAMPLES evaluations in total
    3. else DEFAULT_QUALITY_SCORE
    """
    if row is None:
        return DEFAULT_QUALITY_SCORE

    if task is not None and row["task_count"] >= MIN_QUALITY_SAMPLES:
        return row["task_average"]

    if row["overall_count"] >= MIN_QUALITY_SAMPLES:
        return row["overall_average"]

    return DEFAULT_QUALITY_SCORE


async def load_quality_estimates(store, models: list[str], task: str | None) -> dict[str, float]:
    """Quality estimate per model name, from one grouped query.

    Never raises: without a store, or if the lookup fails or exceeds
    QUALITY_LOOKUP_TIMEOUT_SECONDS, every model gets DEFAULT_QUALITY_SCORE
    so routing carries on.
    """
    neutral = {name: DEFAULT_QUALITY_SCORE for name in models}

    if store is None or not models:
        return neutral

    try:
        history = await asyncio.wait_for(
            store.fetch_quality_history(models, task), timeout=QUALITY_LOOKUP_TIMEOUT_SECONDS
        )
    except Exception:
        logger.warning(
            "Quality history lookup failed; routing with neutral quality %.1f.",
            DEFAULT_QUALITY_SCORE,
            exc_info=True,
        )
        return neutral

    return {name: quality_from_history(history.get(name), task) for name in models}
