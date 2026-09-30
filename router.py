from fastapi import FastAPI, HTTPException


def select_provider(app: FastAPI, model: str):
    if model.startswith("claude"):
        return app.state.anthropic_provider

    if model.startswith("gpt"):
        return app.state.openai_provider

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported model: {model}",
    )

