"""Load-path test for the granite_swa model_type.

Constructing the LLM with --load-format dummy on a tiny granite_swa fixture
flexes every granite_swa code path (each raises on failure):
  1. SpyrePlatform.pre_register_and_update — GraniteSWAForCausalLM arch alias +
     `import fms.models.hf` (granite_swa AutoConfig registration).
  2. vLLM ModelConfig / AutoConfig parse of model_type="granite_swa" + arch resolution.
  3. SpyreCausalLM.load_weights granite_swa kv-cache branch (no NotImplementedError).
  4. FMS hf_configured path building a native GraniteSWA via reset_parameters.
  5. The Spyre warmup that LLM() construction triggers (compile_or_warm_up_model
     -> _warmup_spyre_dynamic_size), which runs prefill + decode forward passes
     with dummy requests on eager CPU -- so a forward is exercised without generate().

Temporary: granite_swa is loaded via the FMS AutoConfig shim because transformers
has no granite_swa implementation yet. Once transformers ships granite_swa natively,
that code and this test will be removed.
"""

from pathlib import Path

import pytest

# granite_swa ships only in FMS builds with fms/models/granite_swa.py, not in
# released ibm-fms (<=1.12.1, which CI pins via uv.lock). Without it,
# `import fms.models.hf` can't register granite_swa with AutoConfig and the load
# fails. Skip cleanly where it's absent so CI stays green until FMS ships it.
pytest.importorskip("fms.models.granite_swa")

from vllm import LLM, ModelRegistry  # noqa: E402

from sendnn_inference import envs as envs_spyre  # noqa: E402

TINY_GRANITE_SWA_DIR = str(
    Path(__file__).parent.parent / "fixtures" / "model_configs" / "ibm-granite" / "tiny-granite-swa"
)
# Reuse the cached micro-g3.3 tokenizer; the fixture's vocab_size (49159) matches it.
TOKENIZER = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
TOKENIZER_REVISION = "6e9c6465a9d7e5e9fa35004a29f0c90befa7d23f"


@pytest.mark.cpu
@pytest.mark.decoder
def test_granite_swa_dummy_load():
    """A granite_swa config loads via the dummy (random-init) path. No inference
    asserted — reaching the assertions means every granite_swa path above succeeded."""
    envs_spyre.override("SENDNN_INFERENCE_DYNAMO_BACKEND", "eager")

    llm = LLM(
        model=TINY_GRANITE_SWA_DIR,
        tokenizer=TOKENIZER,
        tokenizer_revision=TOKENIZER_REVISION,
        load_format="dummy",
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=128,
    )

    assert llm is not None
    # Cheap, version-stable confirmation the granite_swa arch alias was registered.
    assert "GraniteSWAForCausalLM" in ModelRegistry.get_supported_archs()
