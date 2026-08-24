def recall_at_k(retrieved, relevant, k):
    relevant=set(relevant)
    if not relevant: return 0.0
    return len(set(retrieved[:k]) & relevant)/len(relevant)

def precision_at_k(retrieved, relevant, k):
    relevant=set(relevant)
    if k <= 0: return 0.0
    return len(set(retrieved[:k]) & relevant)/k

def reciprocal_rank(retrieved, relevant):
    relevant=set(relevant)
    for i, item in enumerate(retrieved,1):
        if item in relevant: return 1/i
    return 0.0
