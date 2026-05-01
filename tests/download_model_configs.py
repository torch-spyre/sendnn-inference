#!/usr/bin/env python3

import os
from pathlib import Path
from urllib.request import urlretrieve

from transformers import AutoConfig, PretrainedConfig

from sendnn_inference.config.model_registry import get_model_registry

_configs_path = Path(__file__).parent / "fixtures" / "model_configs"


def download_hf_model_config(hf_model_id: str, revision: str = "main") -> PretrainedConfig:
    """
    Use CONFIG_MAPPING to match known patterns to the requested model ID. Does
    not work as reliably as direct download from HF, though (e.g. the
    `transformers_version` field is filled in from the local installation).
    """
    model_config = AutoConfig.from_pretrained(hf_model_id, revision=revision)
    config_path = _configs_path / hf_model_id
    if revision != "main":
        config_path /= revision
    model_config.save_pretrained(config_path, safe_serialization=True)
    return model_config


def download_model_config_from_hf(hf_model_id: str, revision: str = "main"):
    """
    Download the model config.json directly from HuggingFace.
    """
    config_url = f"https://huggingface.co/{hf_model_id}/resolve/{revision}/config.json"
    config_path = _configs_path / hf_model_id
    if revision != "main":
        config_path /= revision
    os.makedirs(config_path, exist_ok=True)
    urlretrieve(config_url, config_path / "hf_config.json")


if __name__ == "__main__":
    model_ids = get_model_registry().list_models()
    for model_id in model_ids:
        config = download_hf_model_config(model_id)
        # download_model_config_from_hf(model_id)
        print(f"model_id: {model_id}")
        print(os.linesep.join(str(config).split(os.linesep)[:4]))

    # model_id = "RedHatAI/granite-3.1-8b-instruct-FP8-dynamic"
    # revisions = ["main", "2f1a9431020bea1db9719c6c447a2267412b569a"]
    # # model_id = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
    # # revisions = ["2714578f54cfb744ece40df9326ee0b47e879e03",
    #                "6e9c6465a9d7e5e9fa35004a29f0c90befa7d23f"]
    # model_id = "ibm-granite/granite-3.3-8b-instruct"
    # revisions = ["3efd179a48ad7cb28ccf46568985af8cf38cbba9",
    #              "de4d3920884ea7b8f8f276d8aa286f8d82afbb83"]
    # for revision in revisions:
    #     config = download_hf_model_config(model_id, revision)
    #     download_model_config_from_hf(model_id, revision)

# TODO: make it a CLI script with parameters:
#  --hf-model-id
#  --revision
#  --all-supported-models
#  --output-dir [default: fixtures/model_configs/<hf-model-id>/config.json]
