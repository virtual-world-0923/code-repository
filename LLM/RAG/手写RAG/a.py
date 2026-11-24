from pathlib import Path
import faiss
from LLM.RAG.手写RAG.config import VECTOR_DB_DIR
import os

a = Path(VECTOR_DB_DIR,"faiss.index")

index = faiss.IndexFlatIP(50)
p = str(Path(VECTOR_DB_DIR.parent.parent , "faiss.index"))
folder = os.path.dirname(p) or "."
print(folder)
# faiss.write_index(index, p)