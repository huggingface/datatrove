from typing import TYPE_CHECKING

from datatrove.pipeline.inference.servers.remote_base import RemoteInferenceServer


if TYPE_CHECKING:
    from datatrove.pipeline.inference.run_inference import InferenceConfig


class VLLMRemoteServer(RemoteInferenceServer):
    """
    Remote vLLM server connector.

    Connects to an existing external vLLM server endpoint instead of spawning
    a local server process. Useful for:
    - Connecting to centrally managed vLLM instances
    - Using vLLM servers on different machines
    - Sharing vLLM resources across multiple pipeline runs
    - Avoiding the overhead of starting/stopping servers

    Example:
        config = InferenceConfig(
            server_type="vllm-remote",
            model_name_or_path="meta-llama/Llama-3-8B",
            external_endpoint="http://my-vllm-server.com:8000"
        )
        server = VLLMRemoteServer(config)
    """

    def __init__(self, config: "InferenceConfig"):
        """
        Initialize remote vLLM server connector.

        Args:
            config: InferenceConfig with external_endpoint specified

        Raises:
            ValueError: If external_endpoint is not provided
        """
        if not config.external_endpoint:
            raise ValueError(
                "external_endpoint is required for vllm-remote server type. "
                "Please provide the URL of your remote vLLM server, e.g., "
                "'http://my-server.com:8000'"
            )

        super().__init__(config, endpoint=config.external_endpoint)
