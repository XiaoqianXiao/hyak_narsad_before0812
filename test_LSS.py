def compute_roi_pair(i, j, ts1, ts2, similarity, pair_num, total_pairs):
    try:
        if similarity == 'pearson':
            sim = pearsonr(ts1[:, i], ts2[:, j])[0]
        elif similarity == 'cosine':
            sim = cosine_similarity(ts1[:, i].reshape(1, -1), ts2[:, j].reshape(1, -1))[0, 0]
        else:
            raise ValueError("similarity must be 'pearson' or 'cosine'")
        logger.debug(f"ROI pair {pair_num}/{total_pairs} ({i} vs {j}) similarity: {sim:.4f}")
        return sim
    except Exception as e:
        logger.error(f"Error computing ROI pair {i} vs {j} ({pair_num}/{total_pairs}): {e}")
        return np.nan