import os
from openai import OpenAI

class RAGGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")

    def answer(self, question, contexts):
        blocks = []
        for i, c in enumerate(contexts, 1):
            meta = c.get("metadata", {})
            blocks.append(
                f"[{i}] Source: {meta.get('source','unknown')}, page {meta.get('page','?')}\n{c['text']}"
            )
        prompt = ("You are an engineering document assistant. Answer ONLY from the supplied context.\n"
                  "Cite every factual claim with [n] using the source number. If the context is insufficient, say so.\n"
                  "Do not invent standards, clauses, numerical values, or page numbers.\n\n"
                  "QUESTION:\n" + question + "\n\nCONTEXT:\n" + "\n\n".join(blocks))
        response = self.client.responses.create(model=self.model, input=prompt)
        return response.output_text
