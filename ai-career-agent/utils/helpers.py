"""Application settings, Gemini client creation, and safe UI error messages."""
import os

from dotenv import load_dotenv


DEFAULT_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-3.5-flash-lite"


def load_settings() -> dict:
    """Read local configuration without logging or displaying secrets."""
    load_dotenv()
    return {
        "api_key": os.getenv("GEMINI_API_KEY", "").strip(),
        "model": os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    }


def create_llm_client(settings: dict):
    """Create the official Google Gemini client, or None when not configured."""
    if not settings.get("api_key"):
        return None
    from google import genai
    from utils.llm_client import GeminiLLM

    from google.genai import types

    sdk_client = genai.Client(
        api_key=settings["api_key"],
        http_options=types.HttpOptions(timeout=45_000),
    )
    return GeminiLLM(
        sdk_client=sdk_client,
        model=settings["model"],
        fallback_model=settings.get("fallback_model", FALLBACK_MODEL),
    )


def safe_error_message(error: Exception) -> str:
    """Translate provider errors to safe messages; never show raw exception text."""
    name = type(error).__name__.lower()
    message = str(error).lower()
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if isinstance(error, (TimeoutError,)) or "timeout" in name or "timed out" in message:
        return "The request timed out. Gemini calls are limited to 45 seconds; wait briefly and retry. Local role retrieval uses only the cached embedding model."
    if status == 429 or "resource_exhausted" in message or "quota" in message or "rate limit" in message:
        return "Gemini quota or rate limit reached. Wait a little and try again, or check the usage limits for your Gemini API key."
    if "api key not valid" in message or "api_key_invalid" in message or "invalid api key" in message:
        return "Gemini rejected the API key. Check GEMINI_API_KEY in your .env file."
    if status == 403 or "permission_denied" in message or "permission denied" in message:
        return "This API key does not have permission to use the Gemini API or selected model. Check the key's Google AI Studio project and model access."
    if status in {500, 502, 503, 504} or "service unavailable" in message or "server_error" in message:
        return "Gemini is temporarily unavailable. Check your connection and try again shortly."
    if (
        "connection" in name
        or "connecterror" in name
        or "timeout" in name
        or "transport" in name
        or "connection" in message
        or "winerror 10013" in message
    ):
        if "winerror 10013" in message:
            return "This app process is blocked from opening an outbound network connection, so it cannot reach Gemini. Restart the app in an environment that allows outbound HTTPS connections."
        return "Could not connect to Gemini. Check that outbound HTTPS access is available, then try again."
    if "empty llm" in message or "empty response" in message:
        return "Gemini returned an empty response. Please try again."
    if "json" in message and ("decode" in message or "invalid" in message):
        return "Gemini returned a response that could not be read as structured analysis. A local skill comparison can be used instead."
    if "local_files_only" in message or "offline mode" in message or "not a local folder" in message:
        return "The local sentence-embedding model is not available. Restore all-MiniLM-L6-v2 in the Hugging Face model cache, then retry role search."
    if any(term in message for term in ("no job-role requirements", "rag returned no requirements", "role knowledge directory", "role search index")):
        return "RAG could not retrieve the selected role's requirements. Check the matching data/job_roles document and local vector_db index."
    if isinstance(error, FileNotFoundError) and ("role" in message or "job_roles" in message or "faiss" in message):
        return "The local role documents or vector index are missing. Check data/job_roles and vector_db, then retry role search."
    if isinstance(error, ValueError) and ("very little text" in message or "empty" in message or "selectable text" in message):
        return "The uploaded resume did not contain enough readable text. Try a text-based PDF, DOCX, or TXT file."
    if isinstance(error, ValueError) and "unsupported file type" in message:
        return "This resume format is not supported. Upload a PDF, DOCX, or TXT file."
    if "faiss" in name or "vector" in message or "index" in message:
        return "The role knowledge index could not be prepared. Check the data/job_roles files and vector_db permissions."
    if "sentence" in message or "huggingface" in message:
        return "The embedding model could not be loaded. Check your network access and embedding model setting."
    if status == 400 or "invalid_argument" in message:
        return "Gemini rejected the request format. Check the selected model and request settings, then try again."
    if status == 401 or "unauthenticated" in message or "authentication" in message:
        return "Gemini authentication failed. Check that GEMINI_API_KEY is valid and enabled for the Gemini API."
    if status == 404 or "model not found" in message or "not supported for generatecontent" in message:
        return "The configured Gemini model is unavailable for this API key. Set GEMINI_MODEL to a model enabled in Google AI Studio."
    return "Gemini returned an unexpected provider error. Check the server log for the redacted error category, then retry."
