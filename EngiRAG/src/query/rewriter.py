import os
from openai import OpenAI

class QueryRewriter:
    def __init__(self):
        self.enabled = bool(os.getenv('OPENAI_API_KEY'))
        self.model = os.getenv('OPENAI_MODEL', 'gpt-5-mini')
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY']) if self.enabled else None

    def rewrite(self, question: str):
        if not self.enabled:
            return [question]
        prompt = f"Rewrite the engineering question into 3 concise retrieval queries. Preserve technical terms, units, standard names, and constraints. Return one query per line, no numbering.\nQuestion: {question}"
        r = self.client.responses.create(model=self.model, input=prompt)
        queries = [x.strip() for x in r.output_text.splitlines() if x.strip()]
        return [question] + queries[:3]
