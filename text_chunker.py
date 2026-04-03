from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    if not text.strip():
        raise ValueError("Der Eingabetext ist leer.")

    if chunk_size <= 0:
        raise ValueError("chunk_size muss größer als 0 sein.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap darf nicht negativ sein.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap muss kleiner als chunk_size sein.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n## ", "\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_text(text)