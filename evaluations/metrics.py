def precision_at_k(relevant_list, k):
    precisions = []
    for rels in relevant_list:
        top_k = rels[:k]
        if len(top_k) == 0:
            precisions.append(0.0)
        else:
            precisions.append(sum(top_k) / len(top_k))
    return sum(precisions) / len(precisions)


def recall_at_k(relevant_list, k):
    recalls = []
    for rels in relevant_list:
        num_relevant = sum(rels)
        if num_relevant == 0:
            recalls.append(0.0)  # avoid division by zero
        else:
            recalls.append(sum(rels[:k]) / num_relevant)
    return sum(recalls) / len(recalls)
