"""Tests for hash functions that work with bytes"""

import unittest

from datatrove.utils.hashes.sha1 import (
    sha1_hash32,
    sha1_hash32_bytes,
    sha1_hash64,
    sha1_hash64_bytes,
)
from datatrove.utils.hashing import HashConfig, create_hash_func
from tests.utils import require_xxhash


class TestHashBytes(unittest.TestCase):
    """Test hash functions for bytes input"""

    def test_sha1_hash32_bytes(self):
        """Test SHA1 32-bit hash for bytes"""
        test_data = b"Hello, World!"
        hash_result = sha1_hash32_bytes(test_data)

        # Should return an integer
        self.assertIsInstance(hash_result, int)

        # Should be deterministic
        self.assertEqual(hash_result, sha1_hash32_bytes(test_data))

        # Different data should produce different hash (with high probability)
        different_hash = sha1_hash32_bytes(b"Different data")
        self.assertNotEqual(hash_result, different_hash)

    def test_sha1_hash64_bytes(self):
        """Test SHA1 64-bit hash for bytes"""
        test_data = b"Hello, World!"
        hash_result = sha1_hash64_bytes(test_data)

        # Should return an integer
        self.assertIsInstance(hash_result, int)

        # Should be deterministic
        self.assertEqual(hash_result, sha1_hash64_bytes(test_data))

        # Different data should produce different hash
        different_hash = sha1_hash64_bytes(b"Different data")
        self.assertNotEqual(hash_result, different_hash)

    def test_sha1_hash_consistency_string_vs_bytes(self):
        """Test that hashing string vs bytes produces same result"""
        test_string = "Hello, World!"
        test_bytes = test_string.encode("utf-8")

        # 32-bit
        hash_from_string = sha1_hash32(test_string)
        hash_from_bytes = sha1_hash32_bytes(test_bytes)
        self.assertEqual(hash_from_string, hash_from_bytes)

        # 64-bit
        hash_from_string = sha1_hash64(test_string)
        hash_from_bytes = sha1_hash64_bytes(test_bytes)
        self.assertEqual(hash_from_string, hash_from_bytes)

    def test_sha1_hash_binary_data(self):
        """Test hashing of binary data with non-UTF8 bytes"""
        # Binary data that's not valid UTF-8
        binary_data = bytes(range(256))

        # Should not raise an exception
        hash32 = sha1_hash32_bytes(binary_data)
        hash64 = sha1_hash64_bytes(binary_data)

        self.assertIsInstance(hash32, int)
        self.assertIsInstance(hash64, int)

        # Should be deterministic
        self.assertEqual(hash32, sha1_hash32_bytes(binary_data))
        self.assertEqual(hash64, sha1_hash64_bytes(binary_data))

    @require_xxhash
    def test_xxhash32_bytes(self):
        """Test xxHash 32-bit hash for bytes"""
        from datatrove.utils.hashes.xxhash import xxhash32, xxhash32_bytes

        test_data = b"Hello, World!"
        hash_result = xxhash32_bytes(test_data)

        # Should return an integer
        self.assertIsInstance(hash_result, int)

        # Should be deterministic
        self.assertEqual(hash_result, xxhash32_bytes(test_data))

        # Should match string version
        self.assertEqual(xxhash32("Hello, World!"), hash_result)

    @require_xxhash
    def test_xxhash64_bytes(self):
        """Test xxHash 64-bit hash for bytes"""
        from datatrove.utils.hashes.xxhash import xxhash64, xxhash64_bytes

        test_data = b"Hello, World!"
        hash_result = xxhash64_bytes(test_data)

        # Should return an integer
        self.assertIsInstance(hash_result, int)

        # Should be deterministic
        self.assertEqual(hash_result, xxhash64_bytes(test_data))

        # Should match string version
        self.assertEqual(xxhash64("Hello, World!"), hash_result)

    def test_create_hash_func_with_bytes_sha1_32(self):
        """Test create_hash_func with bytes input type for SHA1 32-bit"""
        config = HashConfig(precision=32, hash_fc="sha1")
        hash_func = create_hash_func(config, input_type=bytes)

        test_data = b"Test data"
        result = hash_func(test_data)

        self.assertIsInstance(result, int)
        self.assertEqual(result, sha1_hash32_bytes(test_data))

    def test_create_hash_func_with_bytes_sha1_64(self):
        """Test create_hash_func with bytes input type for SHA1 64-bit"""
        config = HashConfig(precision=64, hash_fc="sha1")
        hash_func = create_hash_func(config, input_type=bytes)

        test_data = b"Test data"
        result = hash_func(test_data)

        self.assertIsInstance(result, int)
        self.assertEqual(result, sha1_hash64_bytes(test_data))

    @require_xxhash
    def test_create_hash_func_with_bytes_xxhash_32(self):
        """Test create_hash_func with bytes input type for xxHash 32-bit"""
        from datatrove.utils.hashes.xxhash import xxhash32_bytes

        config = HashConfig(precision=32, hash_fc="xxhash")
        hash_func = create_hash_func(config, input_type=bytes)

        test_data = b"Test data"
        result = hash_func(test_data)

        self.assertIsInstance(result, int)
        self.assertEqual(result, xxhash32_bytes(test_data))

    @require_xxhash
    def test_create_hash_func_with_bytes_xxhash_64(self):
        """Test create_hash_func with bytes input type for xxHash 64-bit"""
        from datatrove.utils.hashes.xxhash import xxhash64_bytes

        config = HashConfig(precision=64, hash_fc="xxhash")
        hash_func = create_hash_func(config, input_type=bytes)

        test_data = b"Test data"
        result = hash_func(test_data)

        self.assertIsInstance(result, int)
        self.assertEqual(result, xxhash64_bytes(test_data))

    def test_create_hash_func_string_type_default(self):
        """Test that create_hash_func defaults to string input type"""
        config = HashConfig(precision=32, hash_fc="sha1")

        # Default should be str
        hash_func = create_hash_func(config)
        result = hash_func("Test data")
        self.assertEqual(result, sha1_hash32("Test data"))

        # Explicit str type
        hash_func = create_hash_func(config, input_type=str)
        result = hash_func("Test data")
        self.assertEqual(result, sha1_hash32("Test data"))

    def test_hash_empty_bytes(self):
        """Test hashing of empty bytes"""
        empty_bytes = b""

        # Should not raise exception
        hash32 = sha1_hash32_bytes(empty_bytes)
        hash64 = sha1_hash64_bytes(empty_bytes)

        self.assertIsInstance(hash32, int)
        self.assertIsInstance(hash64, int)

        # Empty bytes should produce consistent hash
        self.assertEqual(hash32, sha1_hash32_bytes(b""))
        self.assertEqual(hash64, sha1_hash64_bytes(b""))


if __name__ == "__main__":
    unittest.main()
