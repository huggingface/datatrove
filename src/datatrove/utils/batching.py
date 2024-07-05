import itertools


def batched(iterable, n):
    """In python 3.12+ we could use itertools.batched instead

    One difference with itertools.batched: we return a list instead of a tuple

    Args:
      iterable:
      n:

    Returns:

    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch
