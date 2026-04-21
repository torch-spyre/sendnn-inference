import json
import os
from pathlib import Path
from typing import TYPE_CHECKING
import importlib.metadata

# Third Party
from vllm.logger import init_logger

# Local
import sendnn_inference.envs as envs_spyre

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
else:
    ModelConfig = None
    VllmConfig = None

logger = init_logger(__name__)

PRE_COMPILE_MODEL_CONFIG_FILENAME = "model_compile.log.json"
PRE_COMPILE_MODEL_CATALOG_FILENAME = "pre_compiled_cache_catalog.json"
DISABLE_COMPILATION_ENV_VAR = "DISABLE_COMPILATION"


def handle_disable_compilation(vllm_config: VllmConfig, is_decoder: bool):
    """
    The `DISABLE_COMPILATION` environment variable disallows torch_sendnn from
    compiling new graphs forcing it to load from the cache. Enabling
    `SENDNN_INFERENCE_REQUIRE_PRECOMPILED_DECODERS` will force DISABLE_COMPILATION
    for decoder models and require pre-compiled models.

    In order to do this, we must load up some config from the torch_sendnn
    cache and check to make sure that the current vllm config matches,
    otherwise the cached artifacts cannot be used.
    """

    req_precompiled_decoder_env_var = "SENDNN_INFERENCE_REQUIRE_PRECOMPILED_DECODERS"

    if not envs_spyre.SENDNN_INFERENCE_REQUIRE_PRECOMPILED_DECODERS:
        return

    if not is_decoder:
        return

    # If this is a decoder model, disable compilation
    logger.info(
        "[PRECOMPILED_WARN] Setting %s because %s is a decoder model",
        DISABLE_COMPILATION_ENV_VAR,
        vllm_config.model_config.model,
    )
    os.environ[DISABLE_COMPILATION_ENV_VAR] = "true"

    # If the user has set req_precompiled_decoder_env_var,
    # then we need to enforce that they setup their cache
    torch_cache_dir = os.getenv("TORCH_SENDNN_CACHE_DIR", None)
    torch_cache_enabled = bool(int(os.getenv("TORCH_SENDNN_CACHE_ENABLE", "0")))

    if not torch_cache_dir or not torch_cache_enabled or not os.path.isdir(torch_cache_dir):
        raise ValueError(
            f"{req_precompiled_decoder_env_var}=1 requires setting"
            " TORCH_SENDNN_CACHE_DIR to a valid path and setting "
            "TORCH_SENDNN_CACHE_ENABLE=1"
        )

    compilation_config_path = Path(torch_cache_dir) / PRE_COMPILE_MODEL_CONFIG_FILENAME
    compilation_catalog_path = Path(torch_cache_dir) / PRE_COMPILE_MODEL_CATALOG_FILENAME

    if not compilation_catalog_path.exists() and not compilation_config_path.exists():
        raise ValueError(
            f"{req_precompiled_decoder_env_var}=1 was set, but no "
            f"pre-compiled model config was found in the "
            f"TORCH_SENDNN_CACHE_DIR: {str(compilation_config_path)} or"
            f"{str(compilation_catalog_path)} does not exist"
        )

    if not compilation_catalog_path.is_file() and not compilation_config_path.is_file():
        raise ValueError(
            "{req_precompiled_decoder_env_var}=1 was set, but the "
            "pre-compiled model config is not a file"
        )

    matching_config = None

    # Note: In below implementation we don't tell user exactly what's wrong
    # but we do "warn" them about mismatch and provide the list of supported
    # configuration along with what they have given us.
    if compilation_catalog_path.is_file():
        with open(compilation_catalog_path) as f:
            try:
                pre_compile_catalog = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Precompiled catalog {str(compilation_catalog_path)} is not a valid JSON file"
                ) from e
        match_result = match_from_pre_compile_catalog(pre_compile_catalog, vllm_config)

        if match_result == -1:
            # No match found
            logger.warning(
                "[PRECOMPILED_WARN] "
                "Provided vllm configuration doesn't match any of the "
                "pre-compiled model configurations. Catalog: \n%s\n "
                "vllm_config: \n%s",
                str(compilation_catalog_path),
                str(vllm_config),
            )

            # Return with warning
            return
        else:
            matching_config = pre_compile_catalog[match_result]

    elif compilation_config_path.is_file():
        with open(compilation_config_path) as f:
            try:
                compilation_config = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Precompiled model config {str(compilation_config_path)} was not valid json"
                ) from e
        match_result = match_from_model_config_file(compilation_config, vllm_config)
        if not match_result:
            logger.warning(
                "[PRECOMPILED_WARN] "
                "Provided vllm configuration doesn't match any of the "
                "pre-compiled model"
            )
            # Return with warning
            return
        else:
            matching_config = compilation_config

    if matching_config:
        # Check sendnn_inference version
        try:
            sendnn_inference_version = importlib.metadata.version("sendnn_inference")

            config_version = matching_config.get("sendnn_inference_version")
            if config_version is None:
                logger.warning(
                    "[PRECOMPILED_WARN] Pre-compiled config missing sendnn_inference_version "
                    "field. "
                )
            elif config_version != sendnn_inference_version:
                # Can be converted to ValueError if we want to be strict
                # with checking
                logger.warning(
                    "[PRECOMPILED_WARN] "
                    "Model was compiled on sendnn-inference "
                    "%s but the current sendnn_inference version is %s",
                    config_version,
                    sendnn_inference_version,
                )
        except ImportError:
            logger.warning(
                "[PRECOMPILED_WARN] Cannot validate sendnn_inference version against "
                "pre-compiled model config"
            )

        # Check model name
        model_name = matching_config["data"]["MODEL_NAME"]

        if vllm_config.model_config.model != model_name:
            # We don't have a way to easily ensure that the compiled model
            # is the same as the one that the user is loading. We can only
            # warn here if the names do not match.
            logger.warning(
                "[PRECOMPILED_WARN] "
                "Configured model name is %s but the pre-compiled model "
                "config has name %s. Please ensure this is the correct "
                "model",
                vllm_config.model_config.model,
                model_name,
            )


def match_from_pre_compile_catalog(pre_compile_catalog: dict, vllm_config: VllmConfig) -> int:
    """Function to find the pre-compile model configuration that matches
    the provided vllm_config.
    """

    # Iterate through catalog file to find if any configuration matches,
    # otherwise, return False
    for idx, config in enumerate(pre_compile_catalog):
        # Compare each key-value pair with values in vllm_config
        match_result = match_from_model_config_file(config, vllm_config)
        if match_result:
            return idx
    return -1


def match_from_model_config_file(compilation_config: dict, vllm_config: VllmConfig) -> bool:
    """Function to validate if vllm configuration provided matches
    pre-compile model configuration
    """

    # Validate configurations
    vllm_configs = compilation_config["data"]

    # TP size
    tp_size = vllm_configs["NUM_AIUS"]
    if vllm_config.parallel_config.tensor_parallel_size != tp_size:
        return False

    if "SENDNN_INFERENCE_WARMUP_PROMPT_LENS" in vllm_configs:
        get_list = lambda x: [int(i) for i in x.split(",")]

        prompt_lens = get_list(vllm_configs["SENDNN_INFERENCE_WARMUP_PROMPT_LENS"])
        batch_sizes = get_list(vllm_configs["SENDNN_INFERENCE_WARMUP_BATCH_SIZES"])

        if prompt_lens != envs_spyre.SENDNN_INFERENCE_WARMUP_PROMPT_LENS:
            return False

        if batch_sizes != envs_spyre.SENDNN_INFERENCE_WARMUP_BATCH_SIZES:
            return False
    else:
        context_len = vllm_configs["VLLM_DT_MAX_CONTEXT_LEN"]
        batch_size = vllm_configs["VLLM_DT_MAX_BATCH_SIZE"]

        if context_len != vllm_config.model_config.max_model_len:
            return False

        if batch_size != vllm_config.scheduler_config.max_num_seqs:
            return False

    return True
