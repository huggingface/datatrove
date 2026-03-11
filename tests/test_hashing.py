import unittest

from datatrove.utils.hashing import HashConfig, create_hash_func

from .utils import require_xxhash


class TestCreateHashFunc(unittest.TestCase):
    def test_sha1_str_and_bytes_produce_different_funcs(self):
        """sha1 dispatches correctly based on input_type."""
        config = HashConfig(precision=64, hash_fc="sha1")
        hf_str = create_hash_func(config, str)
        hf_bytes = create_hash_func(config, bytes)
        assert hf_str("hello") == hf_bytes(b"hello")

    def test_sha1_32_vs_64_differ(self):
        config32 = HashConfig(precision=32, hash_fc="sha1")
        config64 = HashConfig(precision=64, hash_fc="sha1")
        h32 = create_hash_func(config32)("hello world")
        h64 = create_hash_func(config64)("hello world")
        assert h32 != h64

    @require_xxhash
    def test_xxhash_deterministic(self):
        config = HashConfig(precision=64, hash_fc="xxhash")
        hf = create_hash_func(config)
        assert hf("test_string") == hf("test_string")

    @require_xxhash
    def test_xxhash_different_inputs_different_hashes(self):
        config = HashConfig(precision=64, hash_fc="xxhash")
        hf = create_hash_func(config)
        assert hf("hello") != hf("world")

    @require_xxhash
    def test_xxhash_32_vs_64_differ(self):
        h32 = create_hash_func(HashConfig(precision=32, hash_fc="xxhash"))("hello")
        h64 = create_hash_func(HashConfig(precision=64, hash_fc="xxhash"))("hello")
        assert h32 != h64

    def test_unknown_hash_fc_raises(self):
        config = HashConfig.__new__(HashConfig)
        object.__setattr__(config, "precision", 64)
        object.__setattr__(config, "hash_fc", "unknown_algo")
        with self.assertRaises(ValueError):
            create_hash_func(config)


if __name__ == "__main__":
    unittest.main()
