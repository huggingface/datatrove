# import os
# from bisect import bisect_right
# from typing import Dict, List, Tuple
#
# import kenlm
# import pandas as pd
# import sentencepiece
# from loguru import logger
#
# from datatrove.data import Document
# from datatrove.pipeline.filters.base_filter import BaseFilter
# from datatrove.pipeline.writers.disk_base import DiskWriter
#
# MODELS_BASE_URL = 'http://dl.fbaipublicfiles.com/cc_net/lm/'
#
#
# class CCNetPerplexityLabeler(BaseFilter):
#     name = "🤖 CCNet Perplexity-filter"
#
#     def __init__(
#             self,
#             models_folder: str = None,
#             languages: List[str] = None,
#             exclusion_writer: DiskWriter = None,
#     ):
#         """
#         @param languages: list of languages to compute perplexities for
#         """
#         super(CCNetPerplexityLabeler, self).__init__(exclusion_writer)
#         self.models_folder = models_folder if models_folder else get_asset_path("download/ccnet_lm")
#         if languages is None:
#             languages = ['en']
#         self.languages = languages
#         if self.rank != -1:
#             self.cutoff_csv = pd.read_csv(get_asset_path("ccnet_perplexity_cutoff.csv"), index_col=0)
#         self.models = {}
#         self.kenlm_config = kenlm.Config()
#         self.kenlm_config.load_method = 2
#
#     def _get_files(self, language):
#         return [
#             (os.path.join(self.models_folder, fn), f"{MODELS_BASE_URL}{fn}")
#             for fn in [f"{language}.arpa.bin", f"{language}.sp.model"]
#         ]
#
#     def _get_model(self, language):
#         assert language in self.languages, f"Filtering language '{language}' but it was not in the list of languages."
#         if language not in self.models:
#             files = self._get_files(language)
#             for fpath, _ in files:
#                 assert os.path.isfile(fpath), f"Could not find {fpath}."
#             kenlm_model = kenlm.Model(files[0][0], self.kenlm_config)
#             sp_model = sentencepiece.SentencePieceProcessor()
#             sp_model.load(files[1][0])
#
#             self.models[language] = (kenlm_model, sp_model)
#         return self.models[language]
#
#     def filter(self, doc: Document) -> bool | Tuple[bool, str]:
#         """
#         Compute perplexity and quality percentile for this sample in its own language.
#
#         @return: perplexity and quality percentile labels
#         """
#         language = doc.metadata["language"]
#         kenlm_model, sp_model = self._get_model(language)
#         tokenized_document = " ".join(sp_model.encode_as_pieces(doc.content))
#         perplexity = compute_perplexity(tokenized_document, kenlm_model)
#         quality_p = bisect_right(self.cutoff_csv[language], perplexity) - 1
#         self.stat_update(f"ccnet_{quality_p}")
#         return
#         self.count(f"ccnet_{quality_p}")
#         return {
#             L_PERPLEXITY: perplexity,
#             L_QUALITY_PERCENTILE: quality_p
#         }
#
#     def label(self, sample: Macrodata) -> Dict:
#         """
#         Compute perplexity and quality percentile for this sample in its own language.
#
#         @param sample:
#         @return: perplexity and quality percentile labels
#         """
#         language = sample.labels["language"]
#         kenlm_model, sp_model = self._get_model(language)
#         tokenized_document = " ".join(sp_model.encode_as_pieces(sample.content))
#         perplexity = compute_perplexity(tokenized_document, kenlm_model)
#         quality_p = bisect_right(self.cutoff_csv[language], perplexity) - 1
#         self.count(f"ccnet_{quality_p}")
#         return {
#             L_PERPLEXITY: perplexity,
#             L_QUALITY_PERCENTILE: quality_p
#         }
#
#     def pre_launch_setup(self):
#         """
#         Download the models if they don't exist
#         """
#         super(CCNetPerplexityLabeler, self).pre_launch_setup()
#         for language in self.languages:
#             for fpath, url in self._get_files(language):
#                 if not os.path.isfile(fpath):
#                     logger.info(f"{fpath} not found. Downloading...")
#                     download_file(url, fpath)
#
#
# def compute_perplexity(tokenized_document, model):
#     lines = tokenized_document.split("\n")
#
#     doc_log_score, doc_length = 0, 0
#     for line in lines:
#         log_score = model.score(line)
#         length = len(line.split()) + 1
#         doc_log_score += log_score
#         doc_length += length
#
#     perplexity = round(10 ** (-doc_log_score / doc_length), 1)
#     return perplexity
