import re


def compare_texts(text_a, text_b):
    """Simple deterministic document comparison useful before LLM summarization."""
    a=set(re.findall(r'\b\w[\w.-]*\b', text_a.lower()))
    b=set(re.findall(r'\b\w[\w.-]*\b', text_b.lower()))
    return {
        'unique_to_a': sorted(a-b),
        'unique_to_b': sorted(b-a),
        'shared_terms': sorted(a&b),
        'jaccard_similarity': len(a&b)/max(1, len(a|b))
    }
