import pytest
from scheduling_utils import create_request_for_scheduler_test, random_prompt

from v1.worker.mock_model import InstrumentedModelRunner

from spyre_util import REFERENCE_MODELS


@pytest.mark.cpu
@pytest.mark.worker
@pytest.mark.prefix_caching
def test_block_sharing_for_2_chunks(
    monkeypatch: pytest.MonkeyPatch,
):
    model = REFERENCE_MODELS[InstrumentedModelRunner.DEFAULT_TEST_MODEL]
    pc_model_runner = InstrumentedModelRunner.build(
        monkeypatch,
        max_num_batched_tokens=128,
    )
    prompt = random_prompt(model=model, seed=0, length=192)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt,
        use_golden_token_injection=False,
        generate_hf_results=False,
    )

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt,
        use_golden_token_injection=False,
        generate_hf_results=False,
    )

    chunk_plan = pc_model_runner._plan_chunking(
        request1.request.prompt_token_ids, num_computed_tokens=0
    )
    pc_model_runner.verify_chunk_plan(
        chunk_plan=chunk_plan,
        chunk_count=2,
        padding_blocks=1,
    )

    chunk_plan = pc_model_runner._plan_chunking(
        request2.request.prompt_token_ids, num_computed_tokens=128
    )
    pc_model_runner.verify_chunk_plan(
        chunk_plan=chunk_plan,
        chunk_count=2,
        padding_blocks=1,
        usable_cache_blocks=1,
        total_cache_blocks=2,
    )


@pytest.mark.cpu
@pytest.mark.worker
@pytest.mark.prefix_caching
def test_multi_chunk_partial_match_misaligned(
    monkeypatch: pytest.MonkeyPatch,
):
    """Scenario where two sequences are scheduled which share a common
    prefix. The second sequence shares 254 tokens with the first sequence,
    which is less than two chunks. We can therefore reuse only one chunk
    (254 < 2*128 = 256). This leads to recomputation of the third block.

    p1 = [AB|CD|EF]
    p2 = [AB|CX|EF]

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 384,  max tokens = 2, step joining = 0
            * 1: len = 384, max tokens = 2, step joining = 0
    """
    monkeypatch.setenv("SENDNN_INFERENCE_CP_INTERLEAVE_STEPS", "0")

    pc_model_runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        max_num_batched_tokens=128,
        # needed number of available blocks: 16 + padding block = 17
        # needs to be multiple of batch size (2): 17 -> 18
        available_blocks=18,
    )

    # twice the same seed for a sequence of length 384
    # the first sequence shares the same prefix of length 384 tokens
    # the second sequence shares the same prefix of length 254 tokens
    # hence sequence 1 shares the first 254 tokens with sequence 0

    model = REFERENCE_MODELS[InstrumentedModelRunner.DEFAULT_TEST_MODEL]
    prompt1 = random_prompt(model=model, seed=0, length=384)
    prompt2 = prompt1[0:254] + random_prompt(model=model, seed=0, length=384 - 254)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt1,
        use_golden_token_injection=False,
        generate_hf_results=False,
    )

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=0,
        max_tokens=2,
        prompt=prompt2,
        use_golden_token_injection=False,
        generate_hf_results=False,
    )

    # Schedule chunk 0 of request 0
    model_runner_output_1 = pc_model_runner.execute_new_request(request=request1.request)
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[1, 2]],
        slot_mappings=[[1, 2]],
    )
    pc_model_runner.verify_model_runner_output(
        model_runner_output_1,
        req_ids=["0"],
        num_sampled_token_ids=0,
        tkv=384,
        # actual number of available blocks: 18 - 1 (padding block) = 17
        n_free_blocks=15,  # 17 - 2 = 15
        left_padding={"0": 0},
        prefix_cache_hit_len={"0": 0},
    )

    # Schedule chunk 1 of request 0
    model_runner_output_2 = pc_model_runner.execute_running_requests()
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[1, 2, 3, 4]],
        slot_mappings=[[3, 4]],
    )
    pc_model_runner.verify_model_runner_output(
        model_runner_output_2,
        req_ids=["0"],
        num_sampled_token_ids=0,
        tkv=384,
        n_free_blocks=13,
        left_padding={"0": 0},
        prefix_cache_hit_len={"0": 0},
    )

    # Schedule chunk 2 of request 0
    model_runner_output_3 = pc_model_runner.execute_running_requests()
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[1, 2, 3, 4, 5, 6]],
        slot_mappings=[[5, 6]],
    )
    pc_model_runner.verify_model_runner_output(
        model_runner_output_3,
        req_ids=["0"],
        num_sampled_token_ids=1,
        tkv=384,
        n_free_blocks=11,
        left_padding={"0": 0},
    )

    # Schedule chunk 0 of request 1
    model_runner_output_4 = pc_model_runner.execute_new_request(request=request2.request)
    # chunk 0 loaded from cache, we skip to chunk 1
    pc_model_runner.verify_model_runner_output(
        model_runner_output_4,
        req_ids=["1"],
        num_sampled_token_ids=0,
        tkv=384,
        n_free_blocks=9,
        left_padding={"1": 0},
        prefix_cache_hit_len={"1": 128},
    )
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[1, 2, 3, 7]],
        slot_mappings=[[0, 7]],
    )

    # Schedule chunk 2 of request 1
    model_runner_output_5 = pc_model_runner.execute_running_requests()
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[1, 2, 3, 7, 8, 9]],
        slot_mappings=[[8, 9]],
    )
    pc_model_runner.verify_model_runner_output(
        model_runner_output_5,
        req_ids=["1"],
        num_sampled_token_ids=1,
        tkv=384,
        n_free_blocks=8,
        left_padding={"1": 0},
    )

    # Schedule decodes of requests 0 and 1
    model_runner_output_6 = pc_model_runner.execute_running_requests()
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[1, 2, 3, 4, 5, 6, 11], [1, 2, 3, 7, 8, 9, 10]],
        slot_mappings=[[11], [10]],
        slot_slice=slice(0, 1),
    )
    pc_model_runner.verify_model_runner_output(
        model_runner_output_6,
        req_ids=["0", "1"],
        num_sampled_token_ids=2,
        tkv=385,
        n_free_blocks=17,
        left_padding={"0": 0, "1": 0},
    )


@pytest.mark.cpu
@pytest.mark.worker
@pytest.mark.prefix_caching
def test_first_chunk_recomputation(
    monkeypatch: pytest.MonkeyPatch,
):
    """Scenario where two sequences are scheduled with 2 blocks
    each and a common 1 block prefix. Since chunk size is 4 times the block
    size, the first two blocks of the first chunk of each request will be
    padding blocks. In the second request, the third block of the chunk
    will be recomputed to prevent block table deduplication while the
    fourth block will be computed from scratch.

    p1 = [00AB]
    p2 = [00AC]

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 128,  max tokens = 2, step joining = 0
            * 1: len = 128, max tokens = 2, step joining = 0
    """
    monkeypatch.setenv("SENDNN_INFERENCE_CP_INTERLEAVE_STEPS", "0")

    pc_model_runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        max_num_batched_tokens=256,
        # needed number of available blocks: 16 + padding block = 17
        # needs to be multiple of batch size (2): 17 -> 18
        available_blocks=18,
    )

    model = REFERENCE_MODELS[InstrumentedModelRunner.DEFAULT_TEST_MODEL]
    prompt1 = random_prompt(model=model, seed=0, length=128)
    prompt2 = prompt1[0:64] + random_prompt(model=model, seed=0, length=128 - 64)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt1,
        use_golden_token_injection=False,
        generate_hf_results=False,
    )

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=0,
        max_tokens=2,
        prompt=prompt2,
        use_golden_token_injection=False,
        generate_hf_results=False,
    )

    # Schedule chunk 0 of request 0
    model_runner_output_1 = pc_model_runner.execute_new_request(request=request1.request)
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[0, 0, 1, 2]],
        slot_mappings=[[0, 0, 1, 2]],
    )
    pc_model_runner.verify_model_runner_output(
        model_runner_output_1,
        req_ids=["0"],
        num_sampled_token_ids=1,
        tkv=128,
        # actual number of available blocks: 18 - 1 (padding block) = 17
        n_free_blocks=15,  # 17 - (4 - 2 (pads)) = 15
        left_padding={"0": 128},
    )

    # Schedule chunk 0 of request 1
    model_runner_output_2 = pc_model_runner.execute_new_request(request=request2.request)
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[0, 0, 1, 3]],
        slot_mappings=[[0, 0, 0, 3]],
    )
    # ^This block table and slot mapping is the crux of this test.
    # The padding blocks align with the slot mapping pointing to block
    # 0. The third block is a cache hit, but has to be recomputed
    # because we're in the last chunk with a prefix hit. The fourth
    # block is not a prefix hit and has to be computed from scratch
    # in a new block.

    pc_model_runner.verify_model_runner_output(
        model_runner_output_2,
        req_ids=["1"],
        num_sampled_token_ids=1,
        tkv=128,
        n_free_blocks=14,  # 15 - (4 - 2 (pads) - 1 (prefix)) = 14
        left_padding={"1": 128},
    )

    # Schedule decodes of requests 0 and 1
    model_runner_output_3 = pc_model_runner.execute_running_requests()
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[1, 2, 5], [1, 3, 4]],
        slot_mappings=[[5], [4]],
        slot_slice=slice(0, 1),
    )
    pc_model_runner.verify_model_runner_output(
        model_runner_output_3,
        req_ids=["0", "1"],
        num_sampled_token_ids=2,
        tkv=129,
        n_free_blocks=17,
        left_padding={"0": 0, "1": 0},
    )


@pytest.mark.cpu
@pytest.mark.worker
@pytest.mark.prefix_caching
def test_middle_chunk_recomputation_with_padding(
    monkeypatch: pytest.MonkeyPatch,
):
    """Scenario where two sequences are scheduled. The first one has
    8 blocks and the second 10, where the first 8 blocks are the same
    as in the first sequence. Since the chunk size is 4 times the block
    size, the first two blocks of the first chunk of the second request
    will be padding blocks. Since the second chunk of the second request
    is a full hit and is not the last chunk, it can be skipped. The third
    chunk has a cache hit of two blocks, which have to be recomputed to
    prevent duplicates in the block table while the two last blocks will
    be computed from scratch.

    This test also exercises the optimization where the tkv of existing
    requests is increased and padding blocks are added to the left in the
    decode batch to accommodate new requests with larger prompts.

    p1 = [ ABCD | EFGH ]
    p2 = [ 00AB | CDEF | GHIJ ]

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 512,  max tokens = 2, step joining = 0
            * 1: len = 640, max tokens = 2, step joining = 0
    """
    monkeypatch.setenv("SENDNN_INFERENCE_CP_INTERLEAVE_STEPS", "0")

    pc_model_runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        max_num_batched_tokens=256,
        # needed number of available blocks: 32 + padding block = 33
        # needs to be multiple of batch size (2): 33 -> 34
        available_blocks=34,
        max_model_len=1024,
    )

    model = REFERENCE_MODELS[InstrumentedModelRunner.DEFAULT_TEST_MODEL]
    prompt1 = random_prompt(model=model, seed=0, length=512)
    prompt2 = prompt1[0:512] + random_prompt(model=model, seed=0, length=128)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt1,
        use_golden_token_injection=False,
        generate_hf_results=False,
    )

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=0,
        max_tokens=2,
        prompt=prompt2,
        use_golden_token_injection=False,
        generate_hf_results=False,
    )

    # Schedule chunk 0 of request 0
    model_runner_output_1 = pc_model_runner.execute_new_request(request=request1.request)
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[1, 2, 3, 4]],
        slot_mappings=[[1, 2, 3, 4]],
    )

    pc_model_runner.verify_model_runner_output(
        model_runner_output_1,
        req_ids=["0"],
        num_sampled_token_ids=0,
        tkv=512,
        # actual number of available blocks: 34 - 1 (padding block) = 33
        n_free_blocks=29,
        left_padding={"0": 0},
        prefix_cache_hit_len={"0": 0},
    )

    # Schedule chunk 1 of request 0
    model_runner_output_2 = pc_model_runner.execute_running_requests()
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[1, 2, 3, 4, 5, 6, 7, 8]],
        slot_mappings=[[5, 6, 7, 8]],
    )
    pc_model_runner.verify_model_runner_output(
        model_runner_output_2,
        req_ids=["0"],
        num_sampled_token_ids=1,
        tkv=512,
        n_free_blocks=25,
        left_padding={"0": 0},
    )

    # Skip chunk 0 and run chunk 1 of request 1
    model_runner_output_3 = pc_model_runner.execute_new_request(request=request2.request)
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        slot_mappings=[[0, 0, 9, 10]],
    )
    pc_model_runner.verify_model_runner_output(
        model_runner_output_3,
        req_ids=["1"],
        num_sampled_token_ids=1,
        tkv=640,
        n_free_blocks=23,
        left_padding={"1": 128},
    )

    # Schedule decodes of requests 0 and 1
    model_runner_output_4 = pc_model_runner.execute_running_requests()
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 12], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
        slot_mappings=[[12], [11]],
        slot_slice=slice(0, 1),
    )

    pc_model_runner.verify_model_runner_output(
        model_runner_output_4,
        req_ids=["0", "1"],
        num_sampled_token_ids=2,
        tkv=641,
        n_free_blocks=33,
        left_padding={"0": 128, "1": 0},
    )
