import os
import re
import tarfile
from typing import Iterable, Tuple, Union

from tldextract import TLDExtract

from datatrove.data import Document
from datatrove.utils.assets import ASSETS_PATH, DOWNLOAD_PATH

from ..writers.disk_base import DiskWriter
from .base_filter import BaseFilter


normalizer = re.compile(r"[^a-zA-Z0-9]+")


def normalize(text, replace=""):
    return normalizer.sub(replace, text).lower()


def load_list(path, do_normalize=True):
    with open(path) as f:
        return {normalize(x) if do_normalize else x.strip() for x in f if x[0] != "#"}


class URLFilter(BaseFilter):
    name = "😈 Url-filter"

    def __init__(
        self,
        soft_word_threshold: int = 2,
        extra_domains: Iterable = None,
        extra_urls: Iterable = None,
        banned_words: Iterable = None,
        banned_subwords: Iterable = None,
        soft_banned_words: Iterable = None,
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__(exclusion_writer)
        self.download_path = os.path.join(DOWNLOAD_PATH, "url_filter")
        self.soft_word_threshold = soft_word_threshold
        self.block_listed_domains = extra_domains
        self.block_listed_url = extra_urls
        self.banned_words = banned_words
        self.banned_subwords = banned_subwords
        self.soft_banned_words = soft_banned_words
        self._downloaded = False
        self.tldextractor = TLDExtract()

    def get_list(self, abs_path: str, file_name: str, extra: set = None, do_normalize: bool = True):
        return load_list(os.path.join(abs_path, file_name), do_normalize=do_normalize).union(
            set(extra) if extra else set()
        )

    def download_data(self):
        if self._downloaded:
            return
        if not os.path.isfile(f"{self.download_path}adult/domains") or not os.path.isfile(
            f"{self.download_path}adult/urls"
        ):
            with tarfile.open(os.path.join(ASSETS_PATH, "url_filterblacklists.tar.gz"), "r:gz") as tar:
                tar.extractall(self.download_path)
        self.block_listed_domains = self.get_list(
            self.download_path, "adult/domains", self.block_listed_domains, do_normalize=False
        )
        self.block_listed_url = self.get_list(
            self.download_path, "adult/urls", self.block_listed_url, do_normalize=False
        )
        self.banned_words = self.get_list(ASSETS_PATH, "banned_words.txt", self.banned_words)
        self.banned_subwords = self.get_list(ASSETS_PATH, "banned_subwords.txt", self.banned_subwords)
        self.soft_banned_words = self.get_list(ASSETS_PATH, "soft_banned_words.txt", self.soft_banned_words)
        self._downloaded = True

    def filter(self, document: Document) -> Union[bool, Tuple[bool, str]]:
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
        if any(word in normalized_space for word in self.banned_subwords):
            return False, "blacklisted_subword"

        return True
