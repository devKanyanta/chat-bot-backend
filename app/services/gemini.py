import google.generativeai as genai
from fastapi import HTTPException, status

from ..config import get_settings
from ..schemas import ChatRequest


class GeminiChatService:
    def __init__(self) -> None:
        self.settings = get_settings()

        if not self.settings.gemini_api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GEMINI_API_KEY is not configured.",
            )

        genai.configure(api_key=self.settings.gemini_api_key)

    def generate_reply(self, payload: ChatRequest) -> tuple[str, str]:
        generation_config = genai.types.GenerationConfig(
            temperature=payload.temperature,
            max_output_tokens=payload.max_output_tokens,
        )

        model = genai.GenerativeModel(
            model_name=self.settings.gemini_model,
            system_instruction=payload.system_prompt,
            generation_config=generation_config,
        )

        contents = [
            {
                "role": "model" if item.role == "assistant" else "user",
                "parts": [item.content],
            }
            for item in payload.history
        ]
        contents.append({"role": "user", "parts": [payload.message]})

        try:
            response = model.generate_content(contents)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini request failed: {exc}",
            ) from exc

        reply = getattr(response, "text", "").strip()
        if not reply:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini returned an empty response.",
            )

        return reply, self.settings.gemini_model
