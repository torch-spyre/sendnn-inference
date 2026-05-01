"""Generate model configuration tables from model_configs.yaml"""

# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Literal
import yaml


ROOT_DIR = Path(__file__).parent.parent.parent.parent
MODEL_CONFIG_PATH = ROOT_DIR / "sendnn_inference/config/model_configs.yaml"
SUPPORTED_MODELS_PATH = ROOT_DIR / "docs/user_guide/supported_models.md"

# Markers to identify where to insert the generated tables
GENERATIVE_START_MARKER = "<!-- GENERATED_GENERATIVE_MODELS_START -->"
GENERATIVE_END_MARKER = "<!-- GENERATED_GENERATIVE_MODELS_END -->"
POOLING_START_MARKER = "<!-- GENERATED_POOLING_MODELS_START -->"
POOLING_END_MARKER = "<!-- GENERATED_POOLING_MODELS_END -->"

# Models to exclude from documentation
EXCLUDED_MODELS = {
    "ibm-granite/granite-4-8b-dense-hybrid",
    "ibm-granite/granite-4-8b-dense",
    "ibm-granite/granite-4-8b-dense-FP8",
}


def generate_model_table(models_data, config_type):
    """Generate a markdown table for models with the specified config type.

    Args:
        models_data: Dictionary of model configurations
        config_type: Either 'continuous_batching_configs' or 'static_batching_configs'

    Returns:
        Markdown table as a string
    """
    all_keys = set()
    model_configs = {}
    is_static_batching = config_type == "static_batching_configs"

    # First pass: collect all unique keys and organize by model
    for model_name, model_config in models_data.items():
        if model_name in EXCLUDED_MODELS:
            continue
        if config_type not in model_config:
            continue

        configs = model_config[config_type]
        model_configs[model_name] = []

        for config in configs:
            # Exclude device_config
            config_data = {}
            for key, value in config.items():
                if key != "device_config":
                    # For static batching, expand warmup_shapes into separate columns
                    if is_static_batching and key == "warmup_shapes" and isinstance(value, list):
                        # Extract prompt_len and batch_size values
                        prompt_lens = []
                        batch_sizes = []
                        for shape in value:
                            if "prompt_len" in shape:
                                prompt_lens.append(str(shape["prompt_len"]))
                            if "batch_size" in shape:
                                batch_sizes.append(str(shape["batch_size"]))

                        if prompt_lens:
                            config_data["SENDNN_INFERENCE_WARMUP_PROMPT_LENS"] = prompt_lens
                            all_keys.add("SENDNN_INFERENCE_WARMUP_PROMPT_LENS")
                        if batch_sizes:
                            config_data["SENDNN_INFERENCE_WARMUP_BATCH_SIZES"] = batch_sizes
                            all_keys.add("SENDNN_INFERENCE_WARMUP_BATCH_SIZES")
                    else:
                        all_keys.add(key)
                        config_data[key] = value
            model_configs[model_name].append(config_data)

    if not all_keys or not model_configs:
        return ""

    # Sort keys for consistent column order
    sorted_keys = sorted(all_keys)

    # Helper function to format header
    def format_header(key):
        """Format key as header with proper capitalization."""
        # Special case for tp_size
        if key == "tp_size":
            return "Tensor Parallel Size"
        # Special cases for warmup environment variables
        if key == "SENDNN_INFERENCE_WARMUP_PROMPT_LENS":
            return "SENDNN_INFERENCE_WARMUP_PROMPT_LENS"
        if key == "SENDNN_INFERENCE_WARMUP_BATCH_SIZES":
            return "SENDNN_INFERENCE_WARMUP_BATCH_SIZES"
        # Default: convert snake_case to Title Case
        words = key.replace("_", " ").split()
        return " ".join(word.capitalize() for word in words)

    # Build the output with nested tables
    output = []

    for model_name, configs in model_configs.items():
        # Add model name as a section with link to Hugging Face Hub
        hf_url = f"https://huggingface.co/{model_name}"
        output.append(f"\n**[{model_name}]({hf_url})**\n")

        # Build sub-table for this model's configurations
        headers = [format_header(key) for key in sorted_keys]

        # Build table header
        table = "| " + " | ".join(headers) + " |\n"
        table += "|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|\n"

        # Build table rows for each configuration
        for config in configs:
            values = []
            for key in sorted_keys:
                value = config.get(key, "")

                # Handle list values (including the warmup columns)
                if isinstance(value, list):
                    values.append("<br>".join(str(v) for v in value))
                else:
                    values.append(str(value) if value != "" else "")

            table += "| " + " | ".join(values) + " |\n"

        output.append(table)

    return "\n".join(output)


def generate_tables():
    """Read model configs and generate both tables."""
    # Load the YAML file
    with open(MODEL_CONFIG_PATH) as f:
        config_data = yaml.safe_load(f)

    models = config_data.get("models", {})

    # Generate tables
    generative_table = generate_model_table(models, "continuous_batching_configs")
    pooling_table = generate_model_table(models, "static_batching_configs")

    return generative_table, pooling_table


def update_supported_models_doc(generative_table, pooling_table):
    """Update the supported_models.md file with generated tables."""
    with open(SUPPORTED_MODELS_PATH) as f:
        content = f.read()

    # Insert generative models table
    if GENERATIVE_START_MARKER in content and GENERATIVE_END_MARKER in content:
        start_idx = content.find(GENERATIVE_START_MARKER) + len(GENERATIVE_START_MARKER)
        end_idx = content.find(GENERATIVE_END_MARKER)

        new_content = content[:start_idx] + "\n\n" + generative_table + "\n" + content[end_idx:]
        content = new_content

    # Insert pooling models table
    if POOLING_START_MARKER in content and POOLING_END_MARKER in content:
        start_idx = content.find(POOLING_START_MARKER) + len(POOLING_START_MARKER)
        end_idx = content.find(POOLING_END_MARKER)

        new_content = content[:start_idx] + "\n\n" + pooling_table + "\n" + content[end_idx:]
        content = new_content

    # Write back to file
    with open(SUPPORTED_MODELS_PATH, "w") as f:
        f.write(content)


def on_startup(command: Literal["build", "gh-deploy", "serve"], dirty: bool):
    """MkDocs hook that runs on startup."""
    print(f"Generating model configuration tables from {MODEL_CONFIG_PATH}")

    generative_table, pooling_table = generate_tables()

    print(f"Generated {len(generative_table.splitlines())} lines for generative models")
    print(f"Generated {len(pooling_table.splitlines())} lines for pooling models")

    update_supported_models_doc(generative_table, pooling_table)

    print(f"Updated {SUPPORTED_MODELS_PATH}")


# Made with Bob
