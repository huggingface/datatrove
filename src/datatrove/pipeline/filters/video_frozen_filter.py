import shutil
from typing import Tuple

from loguru import logger

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter


class VideoFrozenFilter(BaseFilter):
    """Filter that uses ffmpeg to detect if a video is static (frozen)."""

    name = "🧊 Video-Frozen-filter"
    _requires_dependencies = ["ffmpeg"]

    def __init__(self, exclusion_writer=None, batch_size: int = 1, freeze_threshold: float = 0.005, freeze_duration: int = 60):
        """
        Args:
            exclusion_writer: optionally pass in a writer that will save the dropped documents.
            batch_size: the number of documents to process in a batch.
            freeze_threshold: the noise threshold for detecting a frozen frame (default is 0.005).
            freeze_duration: the minimum duration (in seconds) that frames must be frozen to trigger detection (default is 60 seconds).
        """
        super().__init__(exclusion_writer, batch_size)
        self.ffmpeg = None
        self.freeze_threshold = freeze_threshold
        self.freeze_duration = freeze_duration

        # Check if ffmpeg is installed
        if shutil.which("ffmpeg") is None:
            raise EnvironmentError("ffmpeg is not installed. Please install it to use the VideoFrozenFilter. More details: https://www.ffmpeg.org/download.html")

    def filter(self, doc: Document) -> bool | Tuple[bool, str]:
        video_path = doc.media[0].local_path if doc.media else None
        import os
        if not os.path.exists(video_path):
            logger.warning(f"Video path does not exist: {video_path}")
        if video_path and self.is_video_frozen(video_path):
            return False, "frozen_video"
        return True

    def is_video_frozen(self, video_path: str) -> bool:
        """Dynamically determines intervals and checks if the video is frozen during those intervals."""

        if self.ffmpeg is None:
            import ffmpeg
            self.ffmpeg = ffmpeg

        video_duration = self.get_video_duration(video_path)

        # Adjusted video duration to account for 10-second padding
        effective_duration = video_duration - 20  # Remove 10 seconds from start and end

        if effective_duration <= 0:
            # If the effective duration is less than or equal to 0, return False as we can't analyze anything
            return False

        intervals = []

        # If the effective duration is very short, analyze the whole effective video
        if effective_duration < 300:
            intervals = [("10", str(effective_duration))]
        else:
            # Create intervals every 5 minutes (300 seconds), analyzing 1-minute chunks
            intervals = [(str(10 + i * 300), "60") for i in range(int(effective_duration // 300))]

            # Handle the remaining part of the video, if it exists
            remainder = effective_duration % 300
            if remainder > 0:
                intervals.append((str(video_duration - remainder - 10), str(remainder)))

        for start_time, duration in intervals:
            if self.check_freeze(video_path, start_time, duration):
                print(f"{video_path} at {start_time} seen as frozen")
                return True
        return False


    def get_video_duration(self, video_path: str) -> float:
        """Get the duration of the video in seconds using ffmpeg."""
        try:
            probe = self.ffmpeg.probe(video_path)
            return float(probe['format']['duration'])
        except self.ffmpeg.Error as e:
            logger.info(f"ffprobe {video_path}:")
            logger.error(e.stderr.decode('utf-8'))
            raise e

    def check_freeze(self, video_path: str, start_time: str, duration: str) -> bool:
        """Check for frozen frames in a specific interval using ffmpeg's freezedetect filter."""
        try:
            out, err = (
                self.ffmpeg
                .input(video_path, ss=start_time, t=duration)
                .filter('freezedetect', n=self.freeze_threshold, d=self.freeze_duration)
                .output('null', f='null')
                .run(capture_stdout=True, capture_stderr=True)
            )
            err = err.decode('utf-8')
            return 'freeze_start' in err and 'freeze_end' not in err
        except self.ffmpeg.Error as e:
            print(f"Error processing video {video_path}: {e}")
            return False
