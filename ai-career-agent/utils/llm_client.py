"""Provider-neutral text and JSON generation backed by the Google GenAI SDK."""
import json

from google.genai import types


class GeminiLLM:
    """Small application-facing wrapper around Google's Gemini SDK client."""
    def __init__(self, sdk_client, model: str, fallback_model: str):
        self.sdk_client = sdk_client
        self.model = model
        self.fallback_model = fallback_model

    def generate_response(self, prompt: str, temperature: float = 0.3, json_mode: bool = False) -> str:
        """Generate text; retry once with the fallback only if the model is unavailable."""
        try:
            return self._generate(self.model, prompt, temperature, json_mode)
        except Exception as error:
            if self.model == self.fallback_model or not _model_unavailable(error):
                raise
            self.model = self.fallback_model
            return self._generate(self.model, prompt, temperature, json_mode)

    def generate_response_stream(
        self,
        prompt: str,
        temperature: float = 0.35,
        system_instruction: str | None = None,
    ):
        """Yield one Gemini request as text arrives; never issue a duplicate chat call."""
        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="text/plain",
            system_instruction=system_instruction,
        )
        emitted = False
        for response in self.sdk_client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=config,
        ):
            text = getattr(response, "text", None)
            if isinstance(text, str) and text:
                emitted = True
                yield text
        if not emitted:
            raise RuntimeError("Gemini returned an empty response.")

    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> dict:
        """Request valid JSON from Gemini and parse it into a Python structure."""
        prompt = f"{system_prompt}\n\n{user_prompt}"
        content = self.generate_response(prompt, temperature=temperature, json_mode=True)
        try:
            return json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError("Gemini returned invalid JSON.") from error

    def _generate(self, model: str, prompt: str, temperature: float, json_mode: bool) -> str:
        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        response = self.sdk_client.models.generate_content(model=model, contents=prompt, config=config)
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Gemini returned an empty response.")
        return text.strip()


def _model_unavailable(error: Exception) -> bool:
    """Recognize unavailable model errors without treating quota/auth as fallback."""
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    message = str(error).lower()
    return status == 404 or "model not found" in message or "model is unavailable" in message or "not supported for generatecontent" in message
