# SPDX-License-Identifier: Apache-2.0
"""End-to-end vllm.LLM test for grammar bitmask routing.

Drives a real vllm.LLM (with the model patched out for MockSpyreCausalLM so
the test runs on CPU in seconds) through a many-prompt batch with varying
max_tokens. Each structured request is given a regex constrained to a single
letter ("^b{N}$", "^c{N}$", ...). MockSpyreCausalLM's canned logits favor
the letter "a", so:
  * an unstructured request always samples "a"
  * a structured request must sample only its single allowed letter
  * a request whose bitmask is misrouted samples "a" (because nothing masked
    it) or another request's letter — both violate its regex, and xgrammar
    rejects the token, terminating the request

The mix of varied max_tokens and request churn (max_num_seqs forces some
requests to wait while others run) drives non-involution permutations
between scheduler order and input-batch order during decode steps. Under
those permutations the buggy reorder math in
ChunkedPrefillModelRunner.apply_grammar_bitmask routes a request's grammar
bitmask to the wrong logit row.
"""

from __future__ import annotations

import re
import types

import pytest

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.worker,
    pytest.mark.skip_global_cleanup,
]


@pytest.fixture
def mock_llm(monkeypatch: pytest.MonkeyPatch):
    """Build a vllm.LLM whose model is replaced with MockSpyreCausalLM.

    The patch lives on ChunkedPrefillModelRunner.load_model, so the engine,
    scheduler, worker, warmup, defer-sampling, grammar building, and
    apply_grammar_bitmask paths are all the real production code (including
    async grammar compilation, the default).
    """
    from sendnn_inference.v1.worker.spyre_model_runner import ChunkedPrefillModelRunner
    from v1.worker.mock_model import MockSpyreCausalLM

    def patched_load_model(self):
        self._model = MockSpyreCausalLM(vllm_config=self.vllm_config)
        # Shims for attributes the model runner reads on the real SpyreCausalLM.
        fake_cfg = types.SimpleNamespace(
            src_vocab_size=self._model.vocab_size,
            vocab_size=self._model.vocab_size,
        )
        self._model.fms_model = types.SimpleNamespace(config=fake_cfg)
        self._model.is_multimodal = False
        self._model.mm_model_utils = None

    monkeypatch.setattr(ChunkedPrefillModelRunner, "load_model", patched_load_model)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from spyre_util import REFERENCE_MODELS, patch_environment
    from vllm import LLM

    patch_environment(backend="eager", monkeypatch=monkeypatch)

    model = REFERENCE_MODELS["ibm-ai-platform/micro-g3.3-8b-instruct-1b"]
    llm = LLM(
        model=model.name,
        revision=model.revision,
        max_model_len=512,
        # Larger seq cap so multiple requests can decode concurrently.
        max_num_seqs=8,
        max_num_batched_tokens=128,
        enable_chunked_prefill=True,
        enforce_eager=True,
    )
    return llm


def test_grammar_bitmask_routing_in_llm_with_churn(mock_llm):
    """16 prompts, max_num_seqs=8, alternating structured / unstructured, with
    varied max_tokens. The mix forces multi-request decode steps where the
    scheduler order and input-batch order diverge by non-involution
    permutations, exercising the apply_grammar_bitmask reorder path.

    Each structured request's regex is ``^X{20}$`` so the request is required
    to keep emitting the single letter X for every step (no early EOS — the
    grammar disallows EOS until the 20th X). Any other letter (including
    MockSpyreCausalLM's canned 'a') is a regex violation. xgrammar will
    either keep masking correctly (test passes) or reject a sampled token
    and terminate the request early (test fails).
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    letters = "bcdefghi"
    specs: list[tuple[str, int, str | None]] = []
    for i in range(16):
        # Alternate: even-indexed requests are structured, odd are not.
        letter = letters[(i // 2) % len(letters)] if i % 2 == 0 else None
        max_tok = 5 + (i % 5) * 3  # 5, 8, 11, 14, 17, then repeats
        specs.append((f"prompt {i}", max_tok, letter))

    sampling_params_list = []
    for _, max_tok, letter in specs:
        if letter is None:
            params = SamplingParams(temperature=0.0, max_tokens=max_tok)
        else:
            # ``{20}`` is enough that the FSM never accepts EOS within
            # max_tokens, so we get a deterministic stream of Xs the entire
            # time the request decodes.
            params = SamplingParams(
                temperature=0.0,
                max_tokens=max_tok,
                structured_outputs=StructuredOutputsParams(regex=f"^{letter}{{20}}$"),
            )
        sampling_params_list.append(params)

    prompts = [p for p, *_ in specs]
    outputs = mock_llm.generate(prompts, sampling_params_list)

    failures: list[str] = []
    for i, (out, (prompt, max_tok, letter)) in enumerate(zip(outputs, specs, strict=True)):
        completion = out.outputs[0]
        text = completion.text
        token_ids = list(completion.token_ids)
        if letter is None:
            if not re.fullmatch(r"a*", text):
                failures.append(
                    f"req {i} (no grammar): expected only 'a' tokens "
                    f"(MockSpyreCausalLM's canned output), got {text!r} — a "
                    f"bitmask leaked onto an unstructured row."
                )
        else:
            # text must be only the allowed letter, repeated
            if not re.fullmatch(rf"{letter}*", text):
                failures.append(
                    f"req {i} (grammar=^{letter}{{20}}$, max_tokens={max_tok}): "
                    f"got {text!r} — bitmask routed to the wrong logit row "
                    f"(token ids {token_ids})."
                )
            # length must equal max_tokens — early termination means xgrammar
            # rejected a sampled token (which only happens when the wrong
            # bitmask was applied to this request's row).
            if len(token_ids) != max_tok:
                failures.append(
                    f"req {i} (grammar=^{letter}{{20}}$, max_tokens={max_tok}): "
                    f"only produced {len(token_ids)} tokens "
                    f"(text={text!r}) — grammar likely rejected a sampled "
                    f"token because the bitmask was routed to the wrong row."
                )

    if failures:
        pytest.fail(
            "Grammar bitmask routing produced incorrect outputs:\n"
            + "\n".join(f"  - {f}" for f in failures)
        )
