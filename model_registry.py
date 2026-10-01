from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    provider: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    capabilities: set[str]
    enabled: bool = True


# The single source of truth for which concrete models Azir can route to.
# Insertion order is significant: it is the deterministic preference order
# used when `azir-auto` or fallback has to choose among several models.
#
# Costs are rough, static USD-per-1K-token rates used only for telemetry
# estimates -- not a substitute for each provider's current pricing.
MODEL_REGISTRY = {
    "claude-sonnet-4-6": ModelConfig(
        name="claude-sonnet-4-6",
        provider="anthropic",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        capabilities={"chat", "coding", "reasoning"},
    ),
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        provider="openai",
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
        capabilities={"chat", "classification"},
    ),
}


def get_model(name: str) -> ModelConfig | None:
    return MODEL_REGISTRY.get(name)


def get_enabled_models() -> list[ModelConfig]:
    return [model for model in MODEL_REGISTRY.values() if model.enabled]


def find_models(
    *,
    capability: str | None = None,
    provider: str | None = None,
) -> list[ModelConfig]:
    """Enabled models matching every given filter, in registry order."""
    return [
        model
        for model in get_enabled_models()
        if (capability is None or capability in model.capabilities)
        and (provider is None or model.provider == provider)
    ]
