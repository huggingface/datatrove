import random
import socket
import ssl
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import local
from time import sleep

import requests
from loguru import logger
from urllib3.exceptions import InsecureRequestWarning

from datatrove.data import Document, DocumentsPipeline, Media
from datatrove.pipeline.base import PipelineStep
from datatrove.utils._import_utils import is_dnspython_available


class HTTPFetchReader(PipelineStep):
    type = "📖 - READER"
    name = "🌐 HTTP Fetch Reader"

    def __init__(
        self,
        retry_codes: list[int] = [403, 408, 429, 500, 502, 503, 504],
        timeout: tuple[int, int] = (60, 600),
        workers: int = 10,
        retry_delay: int = 2,
        max_retries: int = 3,
        download_timeout: int = 10,
        max_size: int = 1024 * 1024 * 1024,
        dns_port: int | None = None,
        pool_size: int = 5,
        pool_connections: int = 5,
        custom_agent: str = "HF-Research/1.0",
    ):
        self._retry_delay = retry_delay
        self._max_retries = max_retries
        self._retry_codes = retry_codes
        self.timeout = timeout
        self.workers = workers
        self._scrapers = None
        self._thread_local = None
        self.last_log_time = 0.0
        self.start_time = 0.0
        # To prevent division by zero
        self.processed_documents = 0
        self.processed_media = 0
        self.download_timeout = download_timeout
        self.max_size = max_size
        self.dns_port = dns_port
        self.pool_size = pool_size
        self.pool_connections = pool_connections
        self.custom_agent = custom_agent

        max_timeout = self._retry_delay * (2**self._max_retries - 1)
        logger.debug(f"Initializing HTTPFetchReader with {self.workers} workers, max timeout: {max_timeout} seconds")
        super().__init__()

    @property
    def scraper(self):
        if self._thread_local is None:
            self._thread_local = local()

        if not hasattr(self._thread_local, "scrapers"):
            logger.debug(f"Initializing scrapers for worker {self._thread_local}")
            # disable ssl
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)  # supports TLSv1.0 and above
            ssl_context.options &= ~ssl.OP_NO_TLSv1
            ssl_context.options &= ~ssl.OP_NO_TLSv1_1
            try:
                ssl_context.options |= ssl.OP_LEGACY_SERVER_CONNECT  # Allows unsafe renegotiation
            except AttributeError:
                pass
            ssl_context.set_ciphers("ALL:@SECLEVEL=0")  # allows all cipher suites including weak ones

            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            import requests.adapters

            class CustomHTTPAdapter(requests.adapters.HTTPAdapter):
                def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
                    pool_kwargs["ssl_context"] = ssl_context
                    logger.debug(
                        f"init_poolmanager: {pool_kwargs} with connections: {connections} and maxsize: {maxsize}"
                    )
                    super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)

            self._thread_local.scrapers = [requests.Session()]
            for scraper in self._thread_local.scrapers:
                scraper.mount(
                    "https://",
                    CustomHTTPAdapter(
                        pool_maxsize=self.pool_size, pool_connections=self.pool_connections, pool_block=True
                    ),
                )
                scraper.mount(
                    "http://",
                    CustomHTTPAdapter(
                        pool_maxsize=self.pool_size, pool_connections=self.pool_connections, pool_block=True
                    ),
                )

        return random.choice(self._thread_local.scrapers)

    def check_robots_txt(self, url: str) -> bool:
        """Check if the URL is allowed by robots.txt. Returns True if allowed, False if disallowed."""
        try:
            from urllib.parse import urlparse
            from urllib.robotparser import RobotFileParser

            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            if not rp.can_fetch(self.custom_agent, url):
                logger.debug(f"Robots.txt disallows fetching {url}")
                return False
        except Exception:
            logger.error(f"Error checking robots.txt for {url}, skipping")

        return True

    def fetch_media(self, media: Media) -> tuple[bytes | None, dict | None]:
        url = media.url
        last_status_code = None
        last_reason = None
        response_content = None

        # Check robots.txt
        if not self.check_robots_txt(url):
            return None, {"reason": "robots_txt_disallowed", "status_code": None, "url": url}

        scraper = self.scraper

        for attempt in range(self._max_retries):
            response_content = None
            try:
                custom_headers = {"User-Agent": self.custom_agent}

                with scraper.get(
                    url, timeout=self.timeout, verify=False, headers=custom_headers, stream=True
                ) as response:
                    last_status_code = response.status_code
                    last_reason = response.reason
                    if response.status_code == 200:
                        metadata = dict(response.headers)
                        metadata["fetched_url"] = url
                        response_content = b""
                        download_start_time = time.time()
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if time.time() - download_start_time > self.download_timeout:
                                raise TimeoutError(f"Timeout fetching media from {url}")
                            response_content += chunk

                            # If we get anything over 100MB, we report every 10MB that we are still downloading
                            if (
                                len(response_content) >= 100 * 1024 * 1024
                                and len(response_content) % (10 * 1024 * 1024) == 0
                            ):
                                logger.warning(f"Downloading {len(response_content)} bytes from {url}")

                            if len(response_content) >= self.max_size:
                                response_content = response_content[: self.max_size]
                                metadata["reason"] = "length"
                                break
                        return response_content, metadata
                    # For first attempt, retry on all status codes so that we can rotate user agents
                    elif attempt >= 1 and response.status_code not in self._retry_codes:
                        break
            except requests.exceptions.SSLError as e:
                logger.warning(f"SSL error fetching media from {url}, switching to http: {e}")
                last_reason = "ssl_error"
                if url.startswith("https://"):
                    url = url.replace("https://", "http://", 1)
                else:
                    break

            except requests.exceptions.Timeout as e:
                # No point in retrying, we will never get a response
                logger.warning(f"Timeout fetching media from {url}, error: {e}")
                last_reason = "request_timeout"
                break
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error fetching media from {url}, error: {e}")
                last_reason = "connection_error"
                break

            except TimeoutError as e:
                logger.warning(f"Timeout fetching media from {url}, error: {e}")
                last_reason = "download_timeout"
                break

            except Exception as e:
                logger.warning(f"Error fetching media from {url}, error: {e}")
                last_reason = str(e)
                break

            sleep_time = self._retry_delay * (2**attempt) + random.uniform(0, 1)
            sleep(sleep_time)

        return response_content, {"status_code": last_status_code, "reason": last_reason, "url": url}

    def process_record_result(self, record: Document, media_results: list[tuple[Media, bytes | None, dict | None]]):
        updated_media = []
        for media, response, metadata in media_results:
            if response is not None:
                media.media_bytes = response
                if metadata:
                    media_metadata = dict(media.metadata or {})
                    media_metadata.update(metadata)
                    media.metadata = media_metadata
                if metadata and metadata.get("reason") == "download_timeout":
                    self.stat_update("media_fetch_timeout", value=1, unit="media")
                elif metadata and metadata.get("reason") == "length":
                    self.stat_update("media_fetch_truncated", value=1, unit="media")
                else:
                    self.stat_update("media_fetch_success", value=1, unit="media")
                self.update_media_stats(media)
            else:
                failure_metadata = metadata or {}
                media_metadata = dict(media.metadata or {})
                media_metadata["fetch_response"] = failure_metadata
                media.metadata = media_metadata
                if failure_metadata.get("reason") == "robots_txt_disallowed":
                    self.stat_update("media_fetch_robots_disallowed", value=1, unit="media")
                    logger.error(f"Robots disallowed: {media.url}, {record.id}")
                else:
                    self.stat_update("media_fetch_failed", value=1, unit="media")
                    logger.error(f"Failed to fetch media from {media.url}, {record.id}")
            self.processed_media += 1
            updated_media.append(media)

        record.media = updated_media
        return record

    def log_info(self, queue_size: int):
        # Log every 30 seconds
        if time.time() - self.last_log_time > 30:
            elapsed = max(time.time() - self.start_time, 1e-9)
            media_processed = max(self.processed_media, 1)
            throughput = self.processed_documents / elapsed
            success_rate = self.stats["media_fetch_success"].n / media_processed
            failed_rate = self.stats["media_fetch_failed"].n / media_processed
            truncated_rate = self.stats["media_fetch_truncated"].n / media_processed
            timeout_rate = self.stats["media_fetch_timeout"].n / media_processed
            robots_disallowed_rate = self.stats["media_fetch_robots_disallowed"].n / media_processed

            logger.info(
                f"Throughput: {throughput:.2f} docs/s | "
                f"Queue: {queue_size} | "
                f"Success: {success_rate:.1%} | "
                f"Failed: {failed_rate:.1%} | "
                f"Truncated: {truncated_rate:.1%} | "
                f"Timeout: {timeout_rate:.1%} | "
                f"Robots disallowed: {robots_disallowed_rate:.1%}"
            )
            self.last_log_time = time.time()

    def setup_dns_resolution(self):
        import dns.resolver
        from dns.rdatatype import RdataType

        resolver = dns.resolver.Resolver()
        resolver.nameservers = ["127.0.0.1"]
        resolver.port = self.dns_port or 53
        original_getaddrinfo = socket.getaddrinfo

        def getaddrinfo_patched(host, port, family=0, type=0, proto=0, flags=0):
            addresses = []
            try:
                addresses.extend(
                    [
                        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (rdata.address, port))
                        for rdata in resolver.resolve(host, rdtype=RdataType.A)
                    ]
                )
            except dns.resolver.LifetimeTimeout:
                return original_getaddrinfo(host, port, family, type, proto, flags)
            except (dns.resolver.NoAnswer, dns.resolver.LifetimeTimeout):
                pass

            try:
                addresses.extend(
                    [
                        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (rdata.address, port))
                        for rdata in resolver.resolve(host, rdtype=RdataType.AAAA)
                    ]
                )
            except dns.resolver.LifetimeTimeout:
                return original_getaddrinfo(host, port, family, type, proto, flags)
            except (dns.resolver.NoAnswer, dns.resolver.LifetimeTimeout):
                pass

            if len(addresses) == 0:
                return original_getaddrinfo(host, port, family, type, proto, flags)

            socket.getaddrinfo = getaddrinfo_patched
            logger.debug(f"Custom DNS resolution enabled on port {self.dns_port}")

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        self.start_time = time.time()

        if data is None:
            return

        if self.dns_port is not None:
            if not is_dnspython_available():
                raise ValueError("dnspython not installed, custom DNS resolution disabled.")
            self.setup_dns_resolution()

        # disable warnings
        import warnings

        warnings.filterwarnings("ignore", category=InsecureRequestWarning)

        def fetch_document_media(record: Document):
            media_results = []
            for media in record.media:
                response, headers = self.fetch_media(media)
                media_results.append((media, response, headers))
            return record, media_results

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = set()
            for next_record in data:
                while len(futures) >= self.workers:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED, timeout=1)
                    self.log_info(queue_size=len(futures))
                    with self.track_time():
                        pass

                    for future in done:
                        record, media_results = future.result()
                        self.processed_documents += 1
                        if media_results:
                            yield self.process_record_result(record, media_results)
                        else:
                            yield record

                new_feature = executor.submit(fetch_document_media, next_record)
                futures.add(new_feature)

            # Waiting for all futures to complete
            logger.debug("Waiting for all futures to complete")
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED, timeout=1)
                self.log_info(queue_size=len(futures))
                for future in done:
                    record, media_results = future.result()
                    self.processed_documents += 1
                    if media_results:
                        yield self.process_record_result(record, media_results)
                    else:
                        yield record
