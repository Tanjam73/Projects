import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class KnowledgeAssistant:

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.documents = []
        self.index = None

    def load_documents(
        self,
        folder_path
    ):

        texts = []

        for file in os.listdir(folder_path):

            if file.endswith(".txt"):

                path = os.path.join(
                    folder_path,
                    file
                )

                with open(
                    path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    texts.append(
                        f.read()
                    )

        return texts

    def chunk_documents(
        self,
        docs,
        chunk_size=300
    ):

        chunks = []

        for doc in docs:

            words = doc.split()

            for i in range(
                0,
                len(words),
                chunk_size
            ):

                chunk = " ".join(
                    words[i:i+chunk_size]
                )

                chunks.append(
                    chunk
                )

        return chunks

    def build_knowledge_base(
        self,
        folder_path
    ):

        docs = self.load_documents(
            folder_path
        )

        self.documents = self.chunk_documents(
            docs
        )

        embeddings = self.model.encode(
            self.documents,
            convert_to_numpy=True
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(
            embeddings
        )

        print(
            f"Indexed {len(self.documents)} chunks"
        )

    def retrieve(
        self,
        query,
        top_k=3
    ):

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for idx in indices[0]:

            results.append(
                self.documents[idx]
            )

        return results

    def answer(
        self,
        query
    ):

        retrieved_docs = self.retrieve(
            query
        )

        print("\nRelevant Context:\n")

        for i, doc in enumerate(
            retrieved_docs,
            start=1
        ):

            print(
                f"\nDocument {i}:\n"
            )

            print(doc)

        print(
            "\nSuggested Answer:\n"
        )

        print(
            "Based on the retrieved documents, "
            + retrieved_docs[0]
        )

assistant = KnowledgeAssistant()

assistant.build_knowledge_base(
    "knowledge_base"
)

while True:

    query = input(
        "\nAsk Question: "
    )

    if query.lower() == "exit":
        break

    assistant.answer(
        query
    )
