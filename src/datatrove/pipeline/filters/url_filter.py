import os
import re
import tarfile
from typing import Iterable

from huggingface_hub import cached_assets_path

from datatrove.data import Document
from datatrove.io import safely_create_file
from datatrove.utils._import_utils import ASSETS_PATH
from datatrove.utils.logging import logger

from ..writers.disk_base import DiskWriter
from .base_filter import BaseFilter


normalizer = re.compile(r"[^a-zA-Z0-9]+")


def normalize(text, replace=""):
    return normalizer.sub(replace, text).lower()


def parse_list(line, do_normalize=True):
    return {normalize(x) if do_normalize else x.strip() for x in line if x[0] != "#"}


def get_list(abs_path: str, file_name: str, extra: set, do_normalize: bool = True):
    with open(os.path.join(abs_path, file_name)) as f:
        return parse_list(f, do_normalize).union(extra)


class URLFilter(BaseFilter):
    """
    Performs filtering based on samples urls.
    Samples are removed if:
    - their domain is present on `block_listed_domains`
    - if their subdomain is present on `block_listed_domains`
    - if the full url is present on `block_listed_url`
    - if any word from `banned_words` is in the url
    - if there are at least `soft_word_threshold` words from `soft_banned_words` in the url
    - if any word from `banned_subwords` is a substring of the url
    """

    name = "😈 Url-filter"
    _requires_dependencies = ["tldextract", "fasteners", ("ahocorasick", "pyahocorasick")]

    def __init__(
        self,
        soft_word_threshold: int = 2,
        extra_domains: Iterable = None,
        extra_urls: Iterable = None,
        banned_words: Iterable = None,
        banned_subwords: Iterable = None,
        soft_banned_words: Iterable = None,
        use_integrated_lists: bool = True,
        exclusion_writer: DiskWriter = None,
    ):
        import ahocorasick
        from tldextract import TLDExtract

        super().__init__(exclusion_writer)
        self.soft_word_threshold = soft_word_threshold
        self.block_listed_domains = parse_list(extra_domains, do_normalize=False) if extra_domains else set()
        self.block_listed_url = parse_list(extra_urls, do_normalize=False) if extra_urls else set()
        self.banned_words = parse_list(banned_words) if banned_words else set()
        self.banned_subwords = parse_list(banned_subwords) if banned_subwords else set()
        self.soft_banned_words = parse_list(soft_banned_words) if soft_banned_words else set()
        self.use_integrated_lists = use_integrated_lists
        self._downloaded = False
        self.tldextractor = TLDExtract()

        self.banned_subwords_automaton = ahocorasick.Automaton(ahocorasick.STORE_INTS)
        for word in self.banned_subwords:
            self.banned_subwords_automaton.add_word(word, len(self.banned_subwords_automaton))

        if not self.use_integrated_lists:
            self.banned_subwords_automaton.make_automaton()

    def download_data(self):
        if self._downloaded or not self.use_integrated_lists:
            return
        download_dir = cached_assets_path(library_name="datatrove", namespace="filters", subfolder="url_filter")
        file_to_lock = os.path.join(download_dir, "url_filterblacklists.tar.gz")

        def do_extract():
            logger.info("💥 Extracting url filter blacklists...")
            with tarfile.open(os.path.join(ASSETS_PATH, "url_filterblacklists.tar.gz"), "r:gz") as tar:
                tar.extractall(download_dir)
            logger.info("💥 Extracted url filter blacklists.")

        safely_create_file(file_to_lock, do_extract)

        self.block_listed_domains = get_list(
            download_dir, "adult/domains", self.block_listed_domains, do_normalize=False
        )
        self.block_listed_url = get_list(download_dir, "adult/urls", self.block_listed_url, do_normalize=False)
        self.banned_words = get_list(ASSETS_PATH, "banned_words.txt", self.banned_words)
        self.banned_subwords = get_list(ASSETS_PATH, "banned_subwords.txt", self.banned_subwords)
        self.soft_banned_words = get_list(ASSETS_PATH, "soft_banned_words.txt", self.soft_banned_words)
        for word in self.banned_subwords:
            self.banned_subwords_automaton.add_word(word, len(self.banned_subwords_automaton))
        self.banned_subwords_automaton.make_automaton()
        self._downloaded = True

    def filter(self, document: Document) -> bool | tuple[bool, str]:
        self.download_data()
        url = document.metadata.get("url")

        assert url, "Document does not have url in its metadata"
        url_info = self.tldextractor(url)

        if url_info.registered_domain in self.block_listed_domains:
            return False, "domain"

        if url_info.fqdn in self.block_listed_domains:
            return False, "subdomain"

        if url in self.block_listed_url:
            return False, "url"

        url_words = set(normalizer.split(url))
        if any(word in url_words for word in self.banned_words):
            return False, "hard_blacklisted"

        nb_soft_words = sum([word in url_words for word in self.soft_banned_words])
        if nb_soft_words >= self.soft_word_threshold:
            return False, "soft_blacklisted"

        normalized_space = normalize(url)
        if self.banned_subwords and next(self.banned_subwords_automaton.iter(normalized_space), False):
            return False, "blacklisted_subword"

        return True
