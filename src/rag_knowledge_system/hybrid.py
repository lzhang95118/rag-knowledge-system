def keyword_overlap_score(
    query: str,
    text: str,
) -> float:

    query_words  = query.lower().split()
    text_words  = text.lower().split()

    if not query_words:
        return 0.0

    matches = 0

    for word in query_words:
        if word in text_words:
            matches = matches + 1

    return matches/len(query_words)
