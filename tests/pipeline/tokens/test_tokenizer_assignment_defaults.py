import unittest

from datatrove.utils.word_tokenizers import load_tokenizer_assignments


class TestTokenizerAssignmentDefaults(unittest.TestCase):
    def test_bare_code_follows_default_script_row(self):
        """CSV stores TRUE/FALSE as strings. A truthy "FALSE" used to overwrite the bare code."""
        load_tokenizer_assignments.cache_clear()
        factories = load_tokenizer_assignments()

        self.assertEqual(factories["snd"].args, ("IndicNLPTokenizer", "ur"))
        self.assertEqual(factories["snd"].args, factories["snd_Arab"].args)
        self.assertNotEqual(factories["snd"].args, factories["snd_Latn"].args)
        self.assertEqual(factories["zho"].args, ("SpaCyTokenizer", "zh"))
        self.assertEqual(factories["hin"].args, ("IndicNLPTokenizer", "hi"))
        self.assertEqual(factories["hin"].args, factories["hin_Deva"].args)
