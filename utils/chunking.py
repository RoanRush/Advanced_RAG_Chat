import re
from langchain.text_splitter import RecursiveCharacterTextSplitter


def semantic_chunk(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """
    Split text at natural semantic boundaries (paragraph breaks, markdown headers,
    horizontal rules) rather than fixed character counts.
    Large sections are further split with RecursiveCharacterTextSplitter.
    Tiny trailing fragments are merged into the previous chunk.
    """
    boundary_pattern = re.compile(
        r"(?:\n\s*\n)"
        r"|(?:^#{1,6}\s.+$)"
        r"|(?:\n[-─═]{3,}\n)",
        re.MULTILINE,
    )

    raw_sections = [s.strip() for s in boundary_pattern.split(text) if s.strip()]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )

    chunks: list[str] = []
    for section in raw_sections:
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            chunks.extend(splitter.split_text(section))

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < 80:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    return merged
