import fitz  # PyMuPDF
from typing import List, Dict

def extract_text_from_pdf(file_path: str) -> List[Dict]:
    doc = fitz.open(file_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "source": f"{file_path}#page={page_num + 1}",
                "text": text.strip(),
                "page": page_num + 1
            })
    return pages