from collections import OrderedDict
from time import sleep
from concurrent.futures import CancelledError
import time
from loguru import logger
import dns.rdatatype
from datatrove.data import Document, DocumentsPipeline, Media, MediaType
from dns.rdatatype import RdataType
import socket
from datatrove.pipeline.base import PipelineStep
from threading import local
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import requests
import ssl
import dns.resolver
import random
import cloudscraper
from urllib3.exceptions import InsecureRequestWarning
import urllib.parse
import urllib.robotparser as robotparser

from datatrove.utils.stats import MetricStats


class HTTPFetchReader(PipelineStep):
    type = "📖 - READER"
    name = "🌐 HTTP Fetch Reader"
    def __init__(self, retry_codes: list[int] = [403, 408, 429, 500, 502, 503, 504], timeout: tuple[int, int] = (60,600), workers: int = 10, retry_delay: int = 2, max_retries: int = 3, download_timeout: int = 30, use_cloudscraper: bool = True, max_size: int = 1024 * 1024 * 1024, dns_port: int = None):
        self._retry_delay = retry_delay
        self._max_retries = max_retries
        self._retry_codes = retry_codes
        self.timeout = timeout
        self.workers = workers
        self._scrapers = None
        self._thread_local = None
        self.last_log_time = 0
        self.start_time = 0
        # To prevent division by zero
        self.processed_documents = 1
        self.download_timeout = download_timeout
        self.use_cloudscraper = use_cloudscraper
        self.max_size = max_size
        self.dns_port = dns_port
        self.custom_agents = [
            # "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0",
            # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
            # "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
            "HF-Research/1.0"
        ]

        max_timeout = sum(self._retry_delay ** attempt for attempt in range(self._max_retries))
        logger.info(f"Initializing HTTPFetchReader with {self.workers} workers, max timeout: {max_timeout} seconds")
        super().__init__()

    @property
    def scraper(self):
        if self._thread_local is None:
            self._thread_local = local()

        if not hasattr(self._thread_local, "scrapers"):
            logger.info(f"Initializing scrapers for worker {self._thread_local}")
            # disable ssl
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)  # supports TLSv1.0 and above
            ssl_context.options &= ~ssl.OP_NO_TLSv1
            ssl_context.options &= ~ssl.OP_NO_TLSv1_1
            try:
                ssl_context.options |= ssl.OP_LEGACY_SERVER_CONNECT  # Allows unsafe renegotiation
            except AttributeError:
                pass
            ssl_context.set_ciphers('ALL:@SECLEVEL=0')  # allows all cipher suites including weak ones

            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            if self.use_cloudscraper:
                self._thread_local.scrapers = [cloudscraper.create_scraper(ssl_context=ssl_context, browser={
                    "platform": random.choice(['linux', 'windows', 'darwin']),
                    "desktop": True,
                    "mobile": False,
                    "browser": "chrome",
                }) for _ in range(10)]
            else:
                import requests.adapters
                class CustomHTTPAdapter(requests.adapters.HTTPAdapter):
                    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
                        pool_kwargs["ssl_context"] = ssl_context
                        super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)

                self._thread_local.scrapers = [requests.Session()]
                for scraper in self._thread_local.scrapers:
                    scraper.mount("https://", CustomHTTPAdapter(pool_maxsize=1, pool_connections=1, pool_block=True))
                    scraper.mount("http://", CustomHTTPAdapter(pool_maxsize=1, pool_connections=1, pool_block=True))

        return random.choice(self._thread_local.scrapers)

    def fetch_from_url(self, record: Document) -> tuple[bytes | None, dict | None]:
        url = record.metadata["url"]
        last_status_code = None
        last_reason = None
        response_content = None
        # Check robots.txt first
        scraper = self.scraper

        for attempt in range(self._max_retries):
            response_content = None
            try:
                custom_headers = {}
                # For some reason some websites don't like the old user agents that cloudscraper uses by default
                # so we try the new ones
                if attempt < len(self.custom_agents):
                    custom_headers["User-Agent"] = self.custom_agents[attempt]

                with scraper.get(url, timeout=self.timeout, verify=False, headers=custom_headers, stream=True) as response:
                    last_status_code = response.status_code
                    last_reason = response.reason
                    if response.status_code == 200:
                        metadata = dict(response.headers)
                        response_content = b""
                        download_start_time = time.time()
                        for chunk in response.iter_content(chunk_size=1024*1024):
                            if time.time() - download_start_time > self.download_timeout:
                                raise TimeoutError(f"Timeout fetching media from {url}")
                            response_content += chunk

                            # If we get anything over 100MB, we report every 10MB that we are still downloading
                            if len(response_content) >= 100*1024*1024 and len(response_content) % (10*1024*1024) == 0:
                                logger.warning(f"Downloading {len(response_content)} bytes from {url}")

                            if len(response_content) >= self.max_size:
                                response_content = response_content[:self.max_size]
                                metadata["reason"] = "length"
                                break
                        return response_content, metadata
                    # For first attempt, retry on all status codes so that we can rotate user agents
                    elif attempt >= 1 and response.status_code not in self._retry_codes:
                        break
            except requests.exceptions.SSLError as e:
                logger.warning(f"SSL error fetching media from {url}, switching to http: {e}")
                last_reason = "ssl_error"
                url = url.replace("https://", "http://")

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

            sleep_time = self._retry_delay * (2 ** attempt) + random.uniform(0, 1)
            sleep(sleep_time)

        return response_content, {"status_code": last_status_code, "reason": last_reason}
        
        
    def process_record_result(self, response, metadata, record):
        if response is not None:
            content_bytes = response
            record.media.append(Media(
                id=record.id,
                type=MediaType.DOCUMENT,
                media_bytes=content_bytes,
                url=record.metadata["url"],
                metadata=metadata,
            ))
            if metadata.get("reason") == "download_timeout":
                self.stat_update("media_fetch_timeout", value=1, unit="documents")
            elif metadata.get("reason") == "length":
                self.stat_update("media_fetch_truncated", value=1, unit="documents")
            else:
                self.stat_update("media_fetch_success", value=1, unit="documents")
            self.update_media_stats(record.media[-1])
        else:
            record.metadata["fetch_response"] = metadata
            # if metadata.get("reason") == "robots_disallowed":
            #     self.stat_update("media_fetch_robots_disallowed", value=1, unit="documents")
                # logger.error(f"Robots disallowed: {record.metadata['url']}, {record.id}")
            # else:
            self.stat_update("media_fetch_failed", value=1, unit="documents")
            # logger.error(f"Failed to fetch media from {record.metadata['url']}, {record.id}")
        return record

    def log_info(self, queue_size: int):
        # Log each second
        if time.time() - self.last_log_time > 1:
            throughput = self.processed_documents / (time.time() - self.start_time)
            success_rate = self.stats["media_fetch_success"].n/ self.processed_documents
            failed_rate = self.stats["media_fetch_failed"].n/ self.processed_documents
            truncated_rate = self.stats["media_fetch_truncated"].n/ self.processed_documents
            timeout_rate = self.stats["media_fetch_timeout"].n/ self.processed_documents
            # robots_disallowed_rate = self.stats["media_fetch_robots_disallowed"].n/ self.processed_documents

            logger.info(
                f"Throughput: {throughput:.2f} docs/s | "
                f"Queue: {queue_size} | "
                f"Success: {success_rate:.1%} | "
                f"Failed: {failed_rate:.1%} | "
                f"Truncated: {truncated_rate:.1%} | "
                f"Timeout: {timeout_rate:.1%}"
                # f"Robots disallowed: {robots_disallowed_rate:.1%}"
            )
            self.last_log_time = time.time()

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        self.start_time = time.time()

        if data is None:
            return


        if self.dns_port is not None:
            from dns.resolver import Resolver
            resolver = Resolver()
            resolver.nameservers = ['127.0.0.1']
            resolver.port = self.dns_port
            original_getaddrinfo = socket.getaddrinfo
            def getaddrinfo_patched(host, port, family=0, type=0, proto=0, flags=0):
                addresses = []
                try:
                    addresses.extend([(socket.AF_INET, socket.SOCK_STREAM, 6, '', (rdata.address, port)) for rdata in resolver.resolve(host, rdtype=RdataType.A)])
                except dns.resolver.LifetimeTimeout:
                    return original_getaddrinfo(host, port, family, type, proto, flags)
                except (dns.resolver.NoAnswer, dns.resolver.LifetimeTimeout):
                    pass

                try:
                    addresses.extend([(socket.AF_INET6, socket.SOCK_STREAM, 6, '', (rdata.address, port)) for rdata in resolver.resolve(host, rdtype=RdataType.AAAA)])
                except dns.resolver.LifetimeTimeout:
                    return original_getaddrinfo(host, port, family, type, proto, flags)
                except (dns.resolver.NoAnswer, dns.resolver.LifetimeTimeout):
                    pass


                if len(addresses) == 0:
                    return original_getaddrinfo(host, port, family, type, proto, flags)

                return addresses

            socket.getaddrinfo = getaddrinfo_patched

        # disable warnings
        import warnings
        warnings.filterwarnings("ignore", category=InsecureRequestWarning)

        def fetch_from_url_wrapper(record: Document):
            response, headers = self.fetch_from_url(record)
            return response, headers, record

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = set()
            for next_record in data:
                while len(futures) >= self.workers:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED, timeout=1)
                    self.log_info(queue_size=len(futures))
                    with self.track_time():
                        pass

                    for future in done:
                        response, headers, record = future.result()
                        self.processed_documents += 1
                        yield self.process_record_result(response, headers, record)

                new_feature = executor.submit(fetch_from_url_wrapper, next_record)
                futures.add(new_feature)

            # todo use self.track_time()
            # Waiting for all futures to complete
            logger.info("Waiting for all futures to complete")
            while len(futures) > 0:
                done, futures = wait(futures, return_when=FIRST_COMPLETED, timeout=1)
                self.log_info(queue_size=len(futures))
                for future in done:
                    response, headers, record = future.result()
                    self.processed_documents += 1
                    yield self.process_record_result(response, headers, record)


