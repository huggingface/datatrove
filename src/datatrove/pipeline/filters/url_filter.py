import os
import tarfile
import urllib

import tldextract
from loguru import logger

from datatrove.data import Document
from datatrove.utils.typeshelper import LocalPaths

from .base_filter import BaseFilter


BLOCK_LIST = "ftp://ftp.ut-capitole.fr/pub/reseau/cache/squidguard_contrib/blacklists.tar.gz"
ADULT_LIST = "ftp://ftp.ut-capitole.fr/pub/reseau/cache/squidguard_contrib/adult.tar.gz"


def load_list(path) -> set:
    with open(path) as f:
        return {x.strip() for x in f if x[0] != "#"}


class URLFilter(BaseFilter):
    name = "😈 url-filter"

    def __init__(self, model_local_path: str = os.path.join(LocalPaths.download, "url_filter/"), **kwargs):
        super().__init__(**kwargs)
        self.model_local_path = model_local_path
        self.block_listed_domains, self.block_listed_url = self.get_data()

    def get_data(self) -> tuple[set, set]:
        if not os.path.isfile(self.model_local_path + "blacklists.tar.gz"):
            os.makedirs(os.path.dirname(self.model_local_path), exist_ok=True)
            logger.info("⬇️ Downloading block-list urls language identifier model ...")
            urllib.request.urlretrieve(ADULT_LIST, self.model_local_path + "blacklists.tar.gz")

        with tarfile.open(self.model_local_path + "blacklists.tar.gz", "r:gz") as tar:
            tar.extractall(self.model_local_path)

        return load_list(f"{self.model_local_path}adult/domains"), load_list(f"{self.model_local_path}adult/urls")

    def filter(self, document: Document) -> bool | tuple[bool, str]:
        url = document.metadata.get("url")
        assert url, "Document does not have url in its metadata"
        url_info = tldextract.extract(url)

        if url_info.domain in self.block_listed_domains:
            return False, "domain"

        if ".".join([url_info.domain, url_info.subdomain]) in self.block_listed_url:
            return False, "url"

        return True
