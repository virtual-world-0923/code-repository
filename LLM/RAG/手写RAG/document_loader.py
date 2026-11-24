import os
import pymupdf


def load_txt(file_path):
    """读取 TXT 文件"""

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return [{
        "text": text,
        "source": file_path,
        "page": None
    }]


def load_pdf(file_path):
    """读取 PDF 文件"""

    documents = []

    doc = pymupdf.open(file_path)

    for page_num, page in enumerate(doc):

        text = page.get_text()

        if text.strip():

            documents.append({
                "text": text,
                "source": file_path,
                "page": page_num + 1
            })

    doc.close()

    return documents


def load_documents(data_dir):
    """批量读取目录中的 PDF 和 TXT"""

    documents = []

    for filename in os.listdir(data_dir):

        file_path = os.path.join(data_dir, filename)

        if filename.lower().endswith(".txt"):

            documents.extend(load_txt(file_path))

        elif filename.lower().endswith(".pdf"):

            documents.extend(load_pdf(file_path))

    return documents