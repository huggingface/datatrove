import asyncio

from datatrove.pipeline.inference.utils.s3_processing import get_s3_bytes_with_backoff
from warcio.archiveiterator import ArchiveIterator

class S3FileLikeObject:
    """
    A file-like object that reads data from an S3 object using range requests.
    This allows lazy fetching of bytes without downloading the entire object.
    """
    def __init__(self, s3_client, s3_path: str):
        """
        Initializes the S3FileLikeObject.

        Args:
            s3_client: The S3 client instance (e.g., from boto3).
            s3_path: The full S3 path to the object (e.g., "s3://bucket/key").
        """
        self._s3_client = s3_client
        self._s3_path = s3_path
        self._position = 0

    def read(self, k: int):
        """
        Reads up to k bytes from the S3 object starting from the current position.

        Args:
            k: The maximum number of bytes to read. If k is 0, returns empty bytes.
               If k is negative, raises ValueError (standard file object behavior).

        Returns:
            A bytes object containing the data read. Returns empty bytes if the
            current position is at or beyond the end of the object.
        """
        if k < 0:
            raise ValueError("negative seek value")
        if k == 0:
            return b""

        # Calculate the byte range for the S3 request. S3 ranges are inclusive.
        start_byte = self._position
        end_byte = self._position + k - 1

        # Fetch bytes from S3 using the utility function
        # get_s3_bytes_with_backoff(s3_client, key, start_byte, end_byte)
        # The utility handles retries and potential end-of-file cases
        bytes_read = get_s3_bytes_with_backoff(self._s3_client, self._s3_path, start_byte, end_byte)

        # Update the current position by the number of bytes actually read
        self._position += len(bytes_read)

        return bytes_read

    def tell(self) -> int:
        """
        Returns the current position within the S3 object.
        """
        return self._position

    def seek(self, offset: int, whence: int = 0) -> int:
        """
        Changes the current position.

        Args:
            offset: The offset to move the position by.
            whence: The reference point for the offset.
                    0: start of the object (SEEK_SET)
                    1: current position (SEEK_CUR)
                    2: end of the object (SEEK_END) - Not supported as total size is not readily available.

        Returns:
            The new absolute position.

        Raises:
            ValueError: If whence is invalid or the resulting position is negative.
            NotImplementedError: If whence is SEEK_END (2).
        """
        if whence == 0:  # SEEK_SET
            if offset < 0:
                raise ValueError("negative seek value")
            self._position = offset
        elif whence == 1:  # SEEK_CUR
            if self._position + offset < 0:
                 raise ValueError("negative seek value")
            self._position += offset
        elif whence == 2:  # SEEK_END
             # Requires knowing the total size of the S3 object, which isn't stored
             # in this simple class. Could be added by making a HEAD request in __init__.
             raise NotImplementedError("SEEK_END not supported without knowing total size")
        else:
            raise ValueError("Invalid whence value")

        return self._position

    def close(self):
        """
        Closes the object. For this S3 implementation, this is a no-op
        as there are no persistent connections or resources to release within the object itself.
        """
        pass # No resources to close


def read_warc_bytes(s3_client, warc_file, warc_record_offset):
    s3_file = S3FileLikeObject(s3_client, warc_file)
    s3_file.seek(warc_record_offset)
    ait = ArchiveIterator(s3_file, block_size=64*1024)
    warc_record = next(ait)
    content = warc_record.content_stream().read()
    return content, 0



def read_zstd_bytes(s3_client, media_path, offset):
    length_bytes = get_s3_bytes_with_backoff(s3_client, f"{media_path}", offset, offset + 3)
    # Ensure we got exactly 4 bytes 
    assert len(length_bytes) == 4, f"Expected 4 bytes for length, got {len(length_bytes)}"
    length = int.from_bytes(length_bytes, "big")
    zstd_data = get_s3_bytes_with_backoff(s3_client, f"{media_path}", offset + 4, offset + 3 + length)
    assert len(zstd_data) == length, f"Expected {length} bytes for zstd data, got {len(zstd_data)}"
    return zstd_data, length
