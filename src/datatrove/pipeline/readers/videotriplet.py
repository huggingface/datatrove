import json
import warnings
import os
from typing import Callable, List, Dict
from datatrove.io import DataFileLike, DataFolderLike, get_datafolder, download_file
from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.data import Document, Media, MediaType
from datatrove.data import Document, DocumentsPipeline

class VideoTripletReader(BaseDiskReader):
    """Read triplets of video, metadata, and optional caption files."""

    name = "🎥 Video Triplet Reader"

    def __init__(
        self,
        data_folder: DataFolderLike,
        paths_file: DataFileLike | None = None,
        metadata_origin: str | None = None,
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        local_cache_dir = "/tmp/local_video_cache"
    ):
        self.metadata_origin = metadata_origin
        self.local_cache_dir = local_cache_dir  
        os.makedirs(self.local_cache_dir, exist_ok=True)
        super().__init__(
            data_folder,
            paths_file,
            limit,
            skip,
            file_progress,
            doc_progress,
            adapter,
            text_key,
            id_key,
            default_metadata,
            recursive,
        )


    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Overrides the base run method to handle triplet statistics correctly."""
        triplet_count = 0
        if data:
            yield from data
        for triplet in self.find_triplets(rank, world_size):
            document = self.process_triplet(triplet)
            if document:
                self.stat_update("documents")  # Track the number of triplets processed
                self.update_doc_stats(document)
                triplet_count += 1
                yield document

    def find_triplets(self, rank: int = 0, world_size: int = 1) -> List[Dict[str, str]]:
        """Find triplets of video, metadata, and caption files in the data folder."""
        triplets = []
        video_extensions = (".mp4", ".avi", ".mkv", ".mov")
        metadata_extension = ".json"
        caption_extension = ".vtt"

        if self.paths_file:
            with self.data_folder.open(self.paths_file, "r") as f:
                paths = [line.strip() for line in f]
        else:
            paths = self.data_folder.list_files(recursive=self.recursive)

        for path in paths:
            base_name, ext = os.path.splitext(path)
            if ext in video_extensions:
                video_file = path
                metadata_file = base_name + metadata_extension
                caption_file = base_name + caption_extension

                if self.data_folder.exists(metadata_file):
                    triplet = {
                        "video": video_file,
                        "metadata": metadata_file,
                        "caption": caption_file if self.data_folder.exists(caption_file) else None,
                    }
                    triplets.append(triplet)        
        return triplets[rank::world_size]

    def read_file(self, filepath: str):
        for triplet in self.find_triplets():
            with self.track_time():
                document = self.process_triplet(triplet)
                if document:
                    yield document

    def process_triplet(self, triplet: Dict[str, str]) -> Document | None:
        video_path = triplet["video"]
        metadata_path = triplet["metadata"]
        caption_path = triplet["caption"]
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        # Resolve the correct URL and local paths
        video_url = self.data_folder.resolve_paths(video_path)
        video_local_path = self.ensure_local_copy(video_url)

        # Load metadata, video, and caption data
        metadata = self.load_json(metadata_path)
        video_media = Media(type=MediaType.VIDEO, url=video_url, local_path=video_local_path)
        caption_text = self.load_caption(caption_path) if caption_path else ""

        document = Document(
            text=caption_text,
            id=video_id,
            media=[video_media],
            metadata=metadata,
        )

        return document

    def ensure_local_copy(self, video_url: str) -> str:
        """Ensure that the video is available locally. If not, download it."""
        if self.data_folder.is_local():
            return video_url

        local_path = os.path.join(self.local_cache_dir, os.path.basename(video_url))
        if not os.path.exists(local_path):
            download_file(video_url, local_path)
        return local_path

    def load_json(self, filepath: str) -> dict:
        with self.data_folder.open(filepath, "r") as f:
            data = json.load(f)

        if self.metadata_origin == "youtube":
            return self.process_youtube_metadata(data)
        elif self.metadata_origin is None:
            warnings.warn("metadata_origin is not specified. Loading full JSON without processing.")
            return data
        else:
            return data

    def load_caption(self, filepath: str) -> str:
        with self.data_folder.open(filepath, "r") as f:
            return f.read()

    def process_youtube_metadata(self, data: dict) -> dict:
        processed_metadata = {
            "video_codec": data.get("vcodec"),
            "audio_codec": data.get("acodec"),
            "video_resolution": data.get("resolution"),
            "duration": data.get("duration_string"),
            "title": data.get("title"),
            "description": data.get("description"),
            "categories": data.get("categories"),
            "tags": data.get("tags"),
            "channel": data.get("channel"),
            "view_count": data.get("view_count"),
            "comment_count": data.get("comment_count"),
            "like_count": data.get("like_count"),
            "channel_follower_count": data.get("channel_follower_count"),
            "upload_date": data.get("upload_date"),
            "language": data.get("language"),
            "age_limit": data.get("age_limit"),
        }
        return processed_metadata
