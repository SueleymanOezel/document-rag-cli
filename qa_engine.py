from google import genai


def setup_gemini(api_key: str, model_name: str = "gemini-flash-latest") -> genai.Client:
    # model_name is kept for backward compatibility with existing callers.
    _ = model_name
    return genai.Client(api_key=api_key)


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

    response = model.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    if not getattr(response, "text", None):
        raise ValueError("Gemini hat keine Antwort zurückgegeben.")

    return response.text
