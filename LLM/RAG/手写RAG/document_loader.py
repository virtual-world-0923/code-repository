import os
from pathlib import Path

import pymupdf


def load_txt(file_path: Path) -> list:
    """读取一个txt文件

    Args:
        file_path: 文件路径

    Returns:
    ```
        [
            {"text": text, "source": file_path, "page": None}
        ]
    ```
    """

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return [{"text": text, "source": file_path, "page": None}]


def load_pdf(file_path: Path) -> list:
    """读取一个 PDF 文件

    Args:
        file_path: 文件路径

    Returns:
        包含每页文本信息的字典列表，格式如下：
        ```
        [
            {"text": text, "source": file_path, "page": 1},
            {"text": text, "source": file_path, "page": 2},
            ...
        ]
        ```
    """

    documents = []

    doc = pymupdf.open(file_path)

    for page_num, page in enumerate(doc):
        text = page.get_text()

        if text.strip():
            documents.append({"text": text, "source": file_path, "page": page_num + 1})

    doc.close()

    return documents


def load_documents(data_dir: Path) -> list:
    """批量读取目录中的 PDF 和 TXT

    Args:
        data_dir: 存放文件目录

    Returns:
    ```
        txt：一个文件一条记录，page = None
        PDF：一页一条记录，page = 页码
        
        [
            {"text": text, "source": file_path, "page": None},
            {"text": text, "source": file_path, "page": 1},
            {"text": text, "source": file_path, "page": 2},
            ...
        ]
    ```
    """

    documents = []

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)

        if filename.lower().endswith(".txt"):
            documents.extend(load_txt(file_path))

        elif filename.lower().endswith(".pdf"):
            documents.extend(load_pdf(file_path))

    return documents
