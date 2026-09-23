"""Context chunk re-ordering strategies including Lost-in-the-Middle mitigation."""


def reorder_standard[T](items: list[T]) -> list[T]:
    """Retain standard monotonically descending retrieval rank order [d_1, d_2, ..., d_K]."""
    return list(items)


def reorder_lost_in_the_middle[T](items: list[T]) -> list[T]:
    """Alternating placement placing highest-scoring passages at start and end of context.

    Mitigates the 'Lost-in-the-Middle' phenomenon where LLM attention degrades in mid-context.
    Formula: [d_1, d_3, d_5, ..., d_6, d_4, d_2]
    """
    if len(items) <= 2:
        return list(items)

    front_list: list[T] = []
    back_list: list[T] = []
    for idx, item in enumerate(items):
        if idx % 2 == 0:
            front_list.append(item)
        else:
            back_list.append(item)

    # Combine: front items ascending, back items descending
    return front_list + list(reversed(back_list))
