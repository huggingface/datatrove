import json, lzma
import os.path
from json import JSONDecodeError
from typing import Callable, Literal

from loguru import logger

from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.readers.base import BaseDiskReader


class RawSwissDoxReader(BaseDiskReader):
    """Reads the Raw SwissDox dataset. This matches the meta data with the artices, and filters those
    that we are not allowed to use. This was run only on Jans Local Computer before uploading the data to the Cluster.

    Args:
        data_folder: the data folder to read from. Directory where the swissdox data is stored
        compression: the compression to use (default: "infer")
        limit: limit the number of files to read
        progress: show progress bar
        adapter: function to adapt the data dict from the source to a Document.
            Take as input: data: dict, path: str, id_in_file: int | str
            Return: a dict with at least a "text" key
        text_key: key to use for the text in the default adapter (default: "text"). Ignored if you provide your own `adapter`
        id_key: key to use for the id in the default adapter (default: "id"). Ignored if you provide your own `adapter`
        default_metadata: default metadata to add to all documents
        recursive: if True, will read files recursively in subfolders (default: True)
        glob_pattern: a glob pattern to filter files to read (default: None)
        shuffle_files: shuffle the files within the returned shard. Mostly used for data viz. purposes, do not use
            with dedup blocks
    """

    name = "🐿 SwissDox Reader"

    def __init__(
        self,
        data_folder: DataFolderLike,
        compression: Literal["infer", "gzip", "zstd"] | None = "infer",
        limit: int = -1,
        progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        self.base_folder = data_folder
        self.meta_data_folder = get_datafolder(os.path.join(data_folder, "meta"))
        self.glob_pattern = "*.xz"
        super().__init__(
            os.path.join(self.base_folder, 'langident'),
            limit,
            progress,
            None,
            text_key,
            id_key,
            default_metadata,
            recursive,
            self.glob_pattern,
            shuffle_files,
        )
        self.compression = compression


    def iterate_entries(self, f, meta_data: dict):
        tmp_odict = None
        tmp_text = ""
        ignroe_article = False
        for li, line in enumerate(f):
            lang, tag, content = line.replace("\n", "").split("\t")
            if tag == 'id':
                if tmp_odict is not None:
                    tmp_odict['text'] = tmp_text
                    tmp_text = ""
                    yield li, tmp_odict

                meta_data_sample = meta_data.get(content)
                if meta_data_sample is None:
                    ignroe_article = True
                    continue
                else:
                    ignroe_article = False

                tmp_odict = {
                    "id": content,
                    "metadata": meta_data_sample
                }
            elif tag == 'lang_dist':
                if ignroe_article:
                    continue
                tmp_odict['metadata']['lang_dist'] = content
            else:
                if ignroe_article:
                    continue
                tmp_text = f"{tmp_text}\n{content}"
    def load_meta_data(self, filepath):
        meta_data_full = {}
        with self.meta_data_folder.open(filepath, "r", encoding='utf-8', compression=self.compression) as mf:
            for line in mf:
                sline = line.replace("\n", "").split("\t")

                lid = sline[0]
                lang = sline[1]
                sdict = sline[-1].replace("'{", "{").replace("}'", "}")
                news_paper = sline[-2]
                news_paper_short = sline[-3]
                date = sline[-4]

                meta_dict = json.loads(sdict)
                meta_dict['news_paper_short'] = news_paper_short.strip()
                meta_dict['news_paper'] = news_paper.strip()
                meta_dict['pub_date'] = date.strip()
                meta_dict['lang'] = lang
                meta_data_full[lid] = meta_dict
        return meta_data_full

    def read_file(self, filepath: str):
        with self.data_folder.open(filepath, "r", encoding='utf-8', compression=self.compression) as f:
            try:
                meta_fname = f"{filepath.split('.')[0]}.{filepath.split('.')[2]}"
                meta_data_full = self.load_meta_data(meta_fname)
                for li, data_dict in self.iterate_entries(f, meta_data_full):
                    with self.track_time():
                        try:
                            document = self.get_document_from_dict(data_dict, filepath, li)
                            if not document:
                                continue
                        except (EOFError, JSONDecodeError) as e:
                            logger.warning(f"Error when reading `{filepath}`: {e}")
                            continue
                    yield document
            except UnicodeDecodeError as e:
                logger.warning(f"File `{filepath}` may be corrupted: raised UnicodeDecodeError ({e})")
