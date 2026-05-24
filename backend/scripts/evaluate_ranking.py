import numpy as np
from typing import List

def precision_at_k(actual: List[int], predicted: List[int], k: int) -> float:
    """
    Computes Precision at K.
    actual: List of relevant candidate IDs
    predicted: Ranked list of candidate IDs predicted by the model
    """
    if not actual:
        return 0.0
    predicted_k = predicted[:k]
    relevant_in_k = len(set(predicted_k).intersection(set(actual)))
    return relevant_in_k / k

def recall_at_k(actual: List[int], predicted: List[int], k: int) -> float:
    """
    Computes Recall at K.
    actual: List of relevant candidate IDs
    predicted: Ranked list of candidate IDs predicted by the model
    """
    if not actual:
        return 0.0
    predicted_k = predicted[:k]
    relevant_in_k = len(set(predicted_k).intersection(set(actual)))
    return relevant_in_k / len(actual)

def average_precision(actual: List[int], predicted: List[int]) -> float:
    """
    Computes Average Precision (AP) for a single query.
    """
    if not actual:
        return 0.0
    score = 0.0
    num_hits = 0.0
    
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            
    return score / min(len(actual), len(predicted))

def mean_average_precision(actual_lists: List[List[int]], predicted_lists: List[List[int]]) -> float:
    """
    Computes Mean Average Precision (MAP) across multiple queries.
    """
    return np.mean([average_precision(a, p) for a, p in zip(actual_lists, predicted_lists)])

def mean_reciprocal_rank(actual_lists: List[List[int]], predicted_lists: List[List[int]]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR).
    """
    rs = []
    for actual, predicted in zip(actual_lists, predicted_lists):
        for i, p in enumerate(predicted):
            if p in actual:
                rs.append(1.0 / (i + 1.0))
                break
        else:
            rs.append(0.0)
    return np.mean(rs)

def dcg_at_k(relevances: List[float], k: int) -> float:
    """
    Computes Discounted Cumulative Gain (DCG) at K.
    relevances: List of true relevance scores ordered by the model's prediction rank.
    """
    relevances = np.asfarray(relevances)[:k]
    if relevances.size:
        return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
    return 0.0

def ndcg_at_k(relevances: List[float], k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (NDCG) at K.
    """
    dcg = dcg_at_k(relevances, k)
    # Ideal DCG is achieved when sorted in descending order
    idcg = dcg_at_k(sorted(relevances, reverse=True), k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

# ==========================================
# SIMULATION / EVALUATION EXAMPLE
# ==========================================
if __name__ == "__main__":
    print("--- Evaluating Models ---")
    
    # Let's say we have 1 job and a pool of 10 candidates.
    # The "ground truth" (actual relevant candidates for the job) are candidates: 1, 3, 5, 7
    ground_truth = [1, 3, 5, 7]
    
    # Let's say these are the rankings produced by the three models (List of candidate IDs ordered by rank)
    tfidf_ranking = [2, 4, 1, 8, 9, 3, 10, 5, 6, 7]      # TF-IDF struggled, pushed relevant ones down
    bert_ranking = [1, 5, 2, 8, 3, 4, 7, 9, 10, 6]       # BERT did better, but false positive (2) is high
    hybrid_ranking = [3, 1, 5, 7, 2, 4, 8, 9, 10, 6]     # Hybrid put all relevant ones in Top 4!
    
    k = 5
    print(f"\n--- Metrics @ K={k} ---")
    print(f"TF-IDF  - Precision@{k}: {precision_at_k(ground_truth, tfidf_ranking, k):.2f} | Recall@{k}: {recall_at_k(ground_truth, tfidf_ranking, k):.2f}")
    print(f"BERT    - Precision@{k}: {precision_at_k(ground_truth, bert_ranking, k):.2f} | Recall@{k}: {recall_at_k(ground_truth, bert_ranking, k):.2f}")
    print(f"Hybrid  - Precision@{k}: {precision_at_k(ground_truth, hybrid_ranking, k):.2f} | Recall@{k}: {recall_at_k(ground_truth, hybrid_ranking, k):.2f}")

    print("\n--- MRR (Mean Reciprocal Rank) ---")
    print(f"TF-IDF MRR: {mean_reciprocal_rank([ground_truth], [tfidf_ranking]):.2f}")
    print(f"BERT MRR:   {mean_reciprocal_rank([ground_truth], [bert_ranking]):.2f}")
    print(f"Hybrid MRR: {mean_reciprocal_rank([ground_truth], [hybrid_ranking]):.2f}")

    print("\n--- NDCG (Using Graded Relevance) ---")
    # For NDCG, we need graded relevance scores (0 to 3 scale: 0=Bad, 3=Perfect)
    # Let's say candidates [1, 3, 5, 7] have true relevances of [3.0, 2.0, 2.5, 1.0]
    # And everyone else has 0.0
    true_relevance_map = {1: 3.0, 3: 2.0, 5: 2.5, 7: 1.0}
    
    # Convert rankings to ordered relevance lists
    tfidf_rels = [true_relevance_map.get(c, 0.0) for c in tfidf_ranking]
    bert_rels = [true_relevance_map.get(c, 0.0) for c in bert_ranking]
    hybrid_rels = [true_relevance_map.get(c, 0.0) for c in hybrid_ranking]
    
    print(f"TF-IDF  - NDCG@{k}: {ndcg_at_k(tfidf_rels, k):.2f}")
    print(f"BERT    - NDCG@{k}: {ndcg_at_k(bert_rels, k):.2f}")
    print(f"Hybrid  - NDCG@{k}: {ndcg_at_k(hybrid_rels, k):.2f}")
