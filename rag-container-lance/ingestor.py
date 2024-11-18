import os

from langchain_core.documents import Document
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Ingestor:
    def __init__(self):
        self.embeddings = FastEmbedEmbeddings(model_name=os.getenv('EMBEDDINGS'))
        self.semantic_splitter = SemanticChunker(
            self.embeddings, breakpoint_threshold_type="interquartile"
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=128,
            add_start_index=True,
        )
    def chunk(self, doc_paths: List[Path]) -> list[Document]:
        documents = []
        for doc_path in doc_paths:
            loaded_documents = PyPDFium2Loader(doc_path).load()
            document_text = "\n".join([doc.page_content for doc in loaded_documents])
            documents.extend(
                self.recursive_splitter.split_documents(
                    self.semantic_splitter.create_documents([document_text])
                )
            )
            for doc in documents:
                print(doc.page_content)
        return documents


if __name__ == "__main__":
    docs: List[Path] = [Path(r"D:\test_rag_doc\b.pdf")]
    chunked_docs = Ingestor().chunk(docs)
    print(chunked_docs)
