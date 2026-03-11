import unittest

from datatrove.utils.batching import batched


class TestBatched(unittest.TestCase):
    def test_uneven_split_produces_short_last_batch(self):
        result = list(batched("ABCDEFG", 3))
        assert result == [["A", "B", "C"], ["D", "E", "F"], ["G"]]

    def test_returns_lists_not_tuples(self):
        """This differs from itertools.batched which returns tuples."""
        result = list(batched("AB", 2))
        assert isinstance(result[0], list)

    def test_invalid_batch_size_raises(self):
        with self.assertRaises(ValueError):
            list(batched([1, 2], 0))
        with self.assertRaises(ValueError):
            list(batched([1, 2], -1))

    def test_generator_input_is_consumed_lazily(self):
        gen = (x for x in range(7))
        result = list(batched(gen, 3))
        assert result == [[0, 1, 2], [3, 4, 5], [6]]


if __name__ == "__main__":
    unittest.main()
