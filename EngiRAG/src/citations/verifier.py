import re

CITATION_PATTERN = re.compile(r"\[(\d+)\]")

def extract_citations(answer: str):
    return [int(x) for x in CITATION_PATTERN.findall(answer)]

def verify_citations(answer: str, contexts):
    cited = extract_citations(answer)
    valid = [n for n in cited if 1 <= n <= len(contexts)]
    return {
        "citations_found": cited,
        "valid_citations": valid,
        "invalid_citations": [n for n in cited if n < 1 or n > len(contexts)],
        "citation_coverage": len(set(valid)) / max(1, len(set(cited))),
    }
