import time

from google import genai


def _get_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "code", None)
    if callable(status_code):
        try:
            status_code = status_code()
        except Exception:
            status_code = None

    if status_code is not None:
        try:
            return int(status_code)
        except (TypeError, ValueError):
            return None

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if response_status is not None:
        try:
            return int(response_status)
        except (TypeError, ValueError):
            return None

    return None


class QAEngine:
    def __init__(self, api_key: str):
        # Preferred path according to requested API-key initialization style.
        if hasattr(genai, "configure"):
            genai.configure(api_key=api_key)
            self.client = genai.Client()
        else:
            self.client = genai.Client(api_key=api_key)

    def answer_question(
        self,
        question: str,
        context_chunks: list[str],
        model_name: str = "gemini-flash-latest",
    ) -> str:
        return answer_question(
            model=self.client,
            question=question,
            context_chunks=context_chunks,
            model_name=model_name,
        )


def setup_gemini(api_key: str, model_name: str = "gemini-flash-latest") -> genai.Client:
    # model_name is kept for backward compatibility with existing callers.
    _ = model_name
    return QAEngine(api_key=api_key).client


def answer_question(
    model: genai.Client,
    question: str,
    context_chunks: list[str],
    model_name: str = "gemini-flash-latest",
) -> str:
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""
Du bist ein präziser Assistent für Dokumentfragen.

Beantworte die Frage ausschließlich auf Basis des folgenden Kontexts.
Wenn die Antwort nicht im Kontext steht, antworte genau:
"Diese Information ist im Dokument nicht vorhanden."

KONTEXT:
{context}

FRAGE:
{question}

ANTWORT:
""".strip()

    for attempt in range(3):
        try:
            response = model.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            break
        except Exception as exc:
            if _get_status_code(exc) == 503:
                if attempt < 2:
                    time.sleep(2)
                    continue
                return "⚠️ Die KI ist gerade überlastet. Bitte versuche es in 30 Sekunden erneut."
            raise

    if not getattr(response, "text", None):
        raise ValueError("Gemini hat keine Antwort zurückgegeben.")

    return response.text
