# SPDX-License-Identifier: Apache-2.0
# Datastructures defining an input batch

# Based on vllm/vllm/v1/worker/gpu_input_batch.py

from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar, cast

import numpy as np
import torch
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.sample.logits_processor import BatchUpdateBuilder, LogitsProcessors, MoveDirectionality
from vllm.v1.sample.metadata import SamplingMetadata

from sendnn_inference.v1.sample.spyre_logits_processor import LogitProcessorWrapper


class RequestState(Protocol):
    @property
    def num_tokens(self) -> int: ...

    @property
    def req_id(self) -> str: ...

    @property
    def prompt_token_ids(self) -> list[int]: ...


RequestStateT = TypeVar("RequestStateT", bound=RequestState)


class BaseInputBatch(Generic[RequestStateT]):
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
    ):
        assert device.type == "cpu"
        # NOTE: max_num_reqs should be consistent with the warmup shapes
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.device = device
        self.pin_memory = pin_memory
        self.vocab_size = vocab_size

        self._req_ids: list[str | None] = [None] * max_num_reqs
        self.req_id_to_index: dict[str, int] = {}

        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        # Find a way to reduce the CPU memory usage.
        # This buffer is not directly transferred to the GPU, so it does not
        # need to be pinned.
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=torch.int32,
            pin_memory=False,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
        self.num_prompt_tokens: np.ndarray = np.zeros(max_num_reqs, dtype=np.int32)

        # Initialize with max number of requests
        self.padded_batch_size = self.max_num_reqs

        # Keep tracking of number of requests
        self._num_requests = 0

    @property
    def req_ids(self) -> list[str]:
        # None elements should only be present transiently
        # while performing state updates to the batch.
        return cast(list[str], self._req_ids)

    def get_available_index(self) -> int | None:
        raise NotImplementedError

    def add_request(
        self,
        request: RequestStateT,
        req_index: int | None = None,
    ) -> int:
        if req_index is None:
            req_index = self.get_available_index()
        assert req_index is not None
        assert req_index < self.max_num_reqs

        req_id = request.req_id
        self._req_ids[req_index] = req_id

        self.req_id_to_index[req_id] = req_index

        # Copy the prompt token ids.
        num_prompt_tokens = len(request.prompt_token_ids)
        self.num_prompt_tokens[req_index] = num_prompt_tokens

        self.token_ids_cpu[req_index, :num_prompt_tokens] = request.prompt_token_ids

        self._num_requests += 1
        assert self._num_requests <= self.max_num_reqs
        return req_index

    def clear_requests(self):
        """
        Clear the batch, mostly used by static batching
        """
        self.req_id_to_index = {}

        self._req_ids = [None] * self.max_num_reqs

        self._num_requests = 0

    def remove_request(self, req_id: str) -> int | None:
        """
        Free a slot of a request from the batch

        It does the following:
        - mask out the removed request.
        - Remove reference from the sets that track the type of parameter
          e.g. greeedy_reqs
        - Update some containers by reference to update the sampling parameters
          e.g. req_output_token_ids

        For the continuous batching, the removed request indices can be
        overwritten by new requests
        """
        req_index = self.req_id_to_index.pop(req_id, None)

        if req_index is None:
            return None

        # Remove the references

        self._req_ids[req_index] = None
        self._num_requests -= 1
        return req_index

    def _get_num_prompt_tokens(self) -> np.ndarray:
        return self.num_prompt_tokens[: self._num_requests]

    def _get_token_ids(self) -> np.ndarray:
        return self.token_ids_cpu[: self._num_requests]

    def _make_prompt_token_ids_tensor(self) -> torch.Tensor:
        num_prompt_tokens = self._get_num_prompt_tokens()
        max_prompt_len = num_prompt_tokens.max()
        prompt_token_ids_tensor = torch.empty(
            (self._num_requests, max_prompt_len),
            device=self.device,
            dtype=torch.int64,
        )
        prompt_token_ids = prompt_token_ids_tensor.numpy()
        prompt_token_ids[:] = self._get_token_ids()[:, :max_prompt_len]
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.

        for i in range(self._num_requests):
            prompt_token_ids[i, num_prompt_tokens[i] :] = self.vocab_size
        return prompt_token_ids_tensor

    def get_req_index(self, req_id):
        return self.req_id_to_index.get(req_id)

    @property
    def num_reqs(self) -> int:
        return self._num_requests

    @property
    def requests_ids(self) -> list[str]:
        return list(self.req_id_to_index.keys())

    @property
    def sorted_requests_ids(self) -> list[str]:
        return sorted(self.req_id_to_index, key=self.req_id_to_index.get)  # type: ignore


@dataclass
class SamplingRequestState:
    sampling_params: SamplingParams
    req_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int] = field(default_factory=list)
    mm_features: list[MultiModalFeatureSpec] | None = None
    num_computed_tokens: int = 0
    chunk_count: int = 0
    padding_blocks: int = 0
    usable_blocks: int = 0
    total_hit_blocks: int = 0
    block_ids: list[int] = field(default_factory=list)
    generator: torch.Generator | None = None
    # Cache for full multimodal embeddings (computed once, sliced per chunk)
    cached_mm_embeddings: torch.Tensor | None = None

    @property
    def num_tokens(self) -> int:
        # NOTE: In the case of multimodal, multimodal token expansion
        # is done prior to the creation of this object.
        # i.e., "<image>" -> "<image>" * num_image_features
        #
        # This is done by vLLM, *not* in the spyre plugin.
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def append_output_token_ids(self, token_ids: int | list[int]) -> None:
        if isinstance(token_ids, list):
            self.output_token_ids.extend(token_ids)
        else:
            self.output_token_ids.append(token_ids)


class SamplingInputBatch(BaseInputBatch[SamplingRequestState]):
    """
    This class was based on the InputBatch for GPU of vLLM V1.

    The implementation of vLLM was designed to track the request parameters
    and does some optimizations to keep the data organized tight. It also
    build the sampling parameters and do lazy allocations when possible.

    For the Spyre, we do something similar, however we do not worry (for now)
    the transfer data from CPU -> GPU as vLLM does. One key difference between
    those implementations is that we have a mask for active request based on
    the indices stored in `req_indices_mask`. Sometimes we need to check it
    to get the correct index of a request see `get_unpadded_output_indices`.

    For static batching, the correct usage of this class consists in add
    requests and clear the whole batch before process more requests.

    For continuous batching, when a request is removed, it frees a slot where
    a new request can be inserted. Then, the request index mask is used to
    condense the sampling parameters.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        logitsprocs: LogitsProcessors | None = None,
    ):
        super().__init__(
            max_num_reqs,
            max_model_len,
            device,
            pin_memory,
            vocab_size,
        )

        # Sampling-related.
        self.temperature = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)
        self.temperature_cpu = self.temperature.numpy()
        self.greedy_reqs: set[str] = set()
        self.random_reqs: set[str] = set()

        self.top_p = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)

        self.top_p_cpu = self.top_p.numpy()
        self.top_p_reqs: set[str] = set()

        self.top_k = torch.empty((max_num_reqs,), dtype=torch.int32, device=device)
        self.top_k_cpu = self.top_k.numpy()
        self.top_k_reqs: set[str] = set()

        # Frequency penalty related data structures
        self.frequency_penalties = torch.empty((max_num_reqs,), dtype=torch.float, device=device)
        self.frequency_penalties_cpu = self.frequency_penalties.numpy()
        self.frequency_penalties_reqs: set[str] = set()

        # Presence penalty related data structures
        self.presence_penalties = torch.empty((max_num_reqs,), dtype=torch.float, device=device)
        self.presence_penalties_cpu = self.presence_penalties.numpy()
        self.presence_penalties_reqs: set[str] = set()

        # Repetition penalty related data structures
        self.repetition_penalties = torch.empty((max_num_reqs,), dtype=torch.float, device=device)
        self.repetition_penalties_cpu = self.repetition_penalties.numpy()
        self.repetition_penalties_reqs: set[str] = set()

        # req_index -> generator
        # NOTE(woosuk): The indices of the requests that do not have their own
        # generator should not be included in the dictionary.
        self.generators: dict[int, torch.Generator] = {}

        self.num_logprobs: dict[str, int] = {}

        # Internal representation of per-step batch state changes, used for
        # reordering persistent batch and generating logitsprocs batch state
        # updates. Should reset each step.
        self.batch_update_builder = BatchUpdateBuilder()

        self.logitsprocs = logitsprocs or LogitsProcessors()
        self.logitsprocs_wrappers = [
            lp for lp in self.logitsprocs.all if isinstance(lp, LogitProcessorWrapper)
        ]

        self.has_allowed_token_ids: set[str] = set()
        self.allowed_token_ids_mask: torch.Tensor | None = None

        # req_index -> bad_words_token_ids
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

        self.req_output_token_ids: list[list[int] | None] = []

        # Request indices to mask request, and to be padded afterwards
        # This is mapped to model.indices
        self.req_indices_mask = torch.zeros(self.max_num_reqs, dtype=torch.bool, device=device)

        # This is updated each time the batch constituents change.
        self.sampling_metadata = self._make_sampling_metadata()

    def req_id_to_dense_index(self, req_id) -> int:
        """
        This data structure has 3 types of references for data:

        - [request id | req_id] : str -> An id of the request, is passed as
        input in `add_request`.
        - [request index | req_index | req_idx] : int -> The index of the data
        in this batch. This index is aligned with `req_indices_mask` which can
        deactivate indices in the batch. In static batching, the finished
        requests are only deactivated and the data is not reorganized until
        the batch is fully processed. On the other hand, in continuous
        batching, finished request will have their slots free that can receive
        new requests, that is, the batch is continuously being updated.
        - dense_index : int -> The contiguous index of data. This is the index
        of the data of the batch when the padding/slots are removed. For
        instance, the sampling parameters are generated dense and are aligned
        to this index.

        Example:

        Given the table below, where `_` is an empty slot

        request index     |  0  |  1  |  2  |  3  |  4  |  6  |
        request id        | "A" | "B" | "F" |  _  |  _  | "X" |
        req_indices_mask  |  T  |  T  |  T  |  F  |  F  |  F  |
        dense index       |  0  |  1  |  2  |  _  |  _  |  3  |

        If we remove request "B" at request index 1 we will have:

        request index     |  0  |  1  |  2  |  3  |  4  |  6  |
        request id        | "A" |  _  | "F" |  _  |  _  | "X" |
        req_indices_mask  |  T  |  F  |  T  |  F  |  F  |  F  |
        dense index       |  0  |  _  |  1  |  _  |  _  |  2  |

        Note how the dense indices were affected by the removal.

        """

        req_index = self.req_id_to_index[req_id]
        return self.req_idx_to_dense_index(req_index)

    def req_idx_to_dense_index(self, req_index) -> int:
        """
        Convert a request index to a dense index. See `req_id_to_dense_index`
        for more.
        """
        return int(self.req_indices_mask[:req_index].sum().item())

    def get_available_index(self) -> int | None:
        """
        Find a free slot in the batching, used primarily in continuous batching
        """
        available_indices = self.req_indices_mask.logical_not().nonzero()
        available_indices_list = available_indices.squeeze(dim=-1).tolist()
        return available_indices_list[0] if available_indices_list else None

    def add_request(
        self,
        request: "SamplingRequestState",
        req_index: int | None = None,
    ) -> int:
        req_index = super().add_request(request, req_index)
        req_id = request.req_id

        # NOTE: differently from gpu input batch, self.req_output_token_ids
        # is not synced with self._req_ids, it should use
        # self.req_indices_mask to resolve its index considering masked
        # out requests.
        assert self.req_indices_mask[req_index].item() is False
        self.req_indices_mask[req_index] = True
        dense_index = self.req_idx_to_dense_index(req_index)
        self.req_output_token_ids.insert(dense_index, request.output_token_ids)

        params = request.sampling_params  # TODO add pooling params
        tmp_dense = self.num_reqs - 1
        self.batch_update_builder.added.append(
            (tmp_dense, params, request.prompt_token_ids, request.output_token_ids)
        )

        while tmp_dense > dense_index:
            self.batch_update_builder.moved.append(
                (tmp_dense, tmp_dense - 1, MoveDirectionality.SWAP)
            )
            tmp_dense = tmp_dense - 1

        # Copy the output token ids.
        start_idx = len(request.prompt_token_ids)
        end_idx = start_idx + len(request.output_token_ids)
        self.token_ids_cpu[req_index, start_idx:end_idx] = request.output_token_ids

        sampling_params = request.sampling_params
        if sampling_params.sampling_type == SamplingType.GREEDY:
            # Avoid later division by zero.
            self.temperature_cpu[req_index] = -1.0
            self.greedy_reqs.add(req_id)
        else:
            self.temperature_cpu[req_index] = sampling_params.temperature
            self.random_reqs.add(req_id)

        self.top_p_cpu[req_index] = sampling_params.top_p
        if sampling_params.top_p < 1:
            self.top_p_reqs.add(req_id)
        top_k = sampling_params.top_k
        if 0 < top_k < self.vocab_size:
            self.top_k_reqs.add(req_id)
        else:
            top_k = self.vocab_size
        self.top_k_cpu[req_index] = top_k
        self.frequency_penalties_cpu[req_index] = sampling_params.frequency_penalty
        if sampling_params.frequency_penalty != 0.0:
            self.frequency_penalties_reqs.add(req_id)
        self.presence_penalties_cpu[req_index] = sampling_params.presence_penalty
        if sampling_params.presence_penalty != 0.0:
            self.presence_penalties_reqs.add(req_id)
        self.repetition_penalties_cpu[req_index] = sampling_params.repetition_penalty
        if sampling_params.repetition_penalty != 1.0:
            self.repetition_penalties_reqs.add(req_id)

        # NOTE(woosuk): self.generators should not include the requests that
        # do not have their own generator.
        if request.generator is not None:
            self.generators[req_index] = request.generator

        if sampling_params.logprobs is not None:
            self.num_logprobs[req_id] = sampling_params.logprobs

        if sampling_params.allowed_token_ids:
            self.has_allowed_token_ids.add(req_id)
            if self.allowed_token_ids_mask is None:
                # Lazy allocation for this tensor, which can be large.
                self.allowed_token_ids_mask = torch.zeros(
                    self.max_num_reqs, self.vocab_size, dtype=torch.bool, device=self.device
                )
            self.allowed_token_ids_mask[req_index][sampling_params.allowed_token_ids] = True

        if sampling_params.bad_words_token_ids:
            self.bad_words_token_ids[req_index] = sampling_params.bad_words_token_ids
        return req_index

    def clear_requests(self):
        """
        Clear the batch, mostly used by static batching
        """
        super().clear_requests()
        self.req_indices_mask.fill_(False)
        self.req_output_token_ids = []

        self.greedy_reqs = set()
        self.random_reqs = set()
        self.top_p_reqs = set()
        self.top_k_reqs = set()
        self.frequency_penalties_reqs = set()
        self.presence_penalties_reqs = set()
        self.repetition_penalties_reqs = set()
        self.generators = {}
        self.num_logprobs = {}

        self.has_allowed_token_ids = set()
        if self.allowed_token_ids_mask is not None:
            self.allowed_token_ids_mask.fill_(False)

        self.batch_update_builder.get_and_reset(0)

    def remove_request(self, req_id: str):
        """
        Free a slot of a request from the batch

        It does the following:
        - mask out the removed request.
        - Remove reference from the sets that track the type of parameter
          e.g. greeedy_reqs
        - Update some containers by reference to update the sampling parameters
          e.g. req_output_token_ids

        For the continuous batching, the removed request indices can be
        overwritten by new requests
        """

        req_index = super().remove_request(req_id)
        if req_index is None:
            return

        # Remove the references

        # Index corrected based on the padded/deactivated requests
        dense_index = self.req_idx_to_dense_index(req_index)
        # Mask out the request
        self.req_indices_mask[req_index] = False

        # Remove and move up
        self.batch_update_builder.removed_append(dense_index)

        end_dense_idx = min(self._num_requests + 1, self.max_num_reqs - 1)
        for tmp_dense in range(dense_index, end_dense_idx):
            self.batch_update_builder.moved.append(
                (tmp_dense, tmp_dense + 1, MoveDirectionality.UNIDIRECTIONAL)
            )

        # Remove the references
        self.req_output_token_ids.pop(dense_index)

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)

        self.frequency_penalties_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)

        self.has_allowed_token_ids.discard(req_id)

        if self.allowed_token_ids_mask is not None:
            self.allowed_token_ids_mask[req_index].fill_(False)

        self.bad_words_token_ids.pop(req_index, None)

    def refresh_metadata(self):
        """Apply batch updates, reset input batch at end of step

        * Apply batch add/remove/permute to logits procs' states
        * If batch state is modified, update sampling metadata
        """
        batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
        for logit_proc in self.logitsprocs.all:
            logit_proc.update_state(batch_update)
        if batch_update:
            self.sampling_metadata = self._make_sampling_metadata()

    def _make_sampling_metadata(self) -> SamplingMetadata:
        # Mask truncated by the num of requests
        indices_mask = self.req_indices_mask

        if not self.all_greedy:
            temperature = self.temperature[indices_mask]
        else:
            temperature = None

        if not self.no_penalties:
            # The prompt tokens are used only for applying penalties during
            # the sampling process. Hence copy these tensors only when
            # there are requests which need penalties to be applied.
            prompt_token_ids = self._make_prompt_token_ids_tensor()
        else:
            prompt_token_ids = None

        allowed_token_ids_mask: torch.Tensor | None = None
        if not self.no_allowed_token_ids:
            assert self.allowed_token_ids_mask is not None
            allowed_token_ids_mask = self.allowed_token_ids_mask[indices_mask]

        indices = indices_mask.nonzero().squeeze(dim=-1).tolist()

        generators = {
            i: self.generators[idx]
            for i, idx in enumerate(indices)
            if self.generators.get(idx) is not None
        }

        return SamplingMetadata(
            temperature=temperature,
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=None if self.no_top_p else self.top_p[indices_mask],
            top_k=None if self.no_top_k else self.top_k[indices_mask],
            generators=generators,
            max_num_logprobs=self.max_num_logprobs,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=self.frequency_penalties[indices_mask],
            presence_penalties=self.presence_penalties[indices_mask],
            repetition_penalties=self.repetition_penalties[indices_mask],
            # WARN: dangerous side-effect. Here output_token_ids is a reference
            # and may be updated from other contexts. For instance,
            # spyre_model_runner updates this data at _update_states.
            output_token_ids=cast(list[list[int]], self.req_output_token_ids),
            no_penalties=self.no_penalties,
            allowed_token_ids_mask=allowed_token_ids_mask,
            bad_words_token_ids=self.bad_words_token_ids,
            logitsprocs=self.logitsprocs,
        )

    def _get_num_prompt_tokens(self) -> np.ndarray:
        req_indices_mask_cpu = self.req_indices_mask.numpy()
        return self.num_prompt_tokens[req_indices_mask_cpu]

    def _get_token_ids(self) -> np.ndarray:
        req_indices_mask_cpu = self.req_indices_mask.numpy()
        return self.token_ids_cpu[req_indices_mask_cpu]

    def get_unpadded_output_indices(self) -> dict[str, int]:
        """The inputs to the model are all padded to a constant batch size, and
        self.req_id_to_index is the map of request id -> padded index.
        However, finished requests and padded requests are stripped from the
        output, so the mapping of request id -> unpadded output index needs to
        be created to be returned in `ModelRunnerOutput`.

        For example if:
        - self.req_indices_mask = [F, T, T, F]
        - self.req_id_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
        This will output: {"B": 0, "C": 1}
        """

        indices = self.req_indices_mask.nonzero().squeeze(dim=-1).tolist()
        return {self._req_ids[idx]: i for i, idx in enumerate(indices)}

    def get_model_indices(self):
        return self.req_indices_mask[: self.padded_batch_size]

    @property
    def all_greedy(self) -> bool:
        return len(self.random_reqs) == 0

    @property
    def all_random(self) -> bool:
        return len(self.greedy_reqs) == 0

    @property
    def no_top_p(self) -> bool:
        return len(self.top_p_reqs) == 0

    @property
    def no_top_k(self) -> bool:
        return len(self.top_k_reqs) == 0

    @property
    def no_penalties(self) -> bool:
        return (
            len(self.presence_penalties_reqs) == 0
            and len(self.frequency_penalties_reqs) == 0
            and len(self.repetition_penalties_reqs) == 0
        )

    @property
    def max_num_logprobs(self) -> int | None:
        return max(self.num_logprobs.values()) if self.num_logprobs else None

    @property
    def no_allowed_token_ids(self) -> bool:
        return len(self.has_allowed_token_ids) == 0

    @property
    def request_indices(self) -> list[int]:
        return self.req_indices_mask.nonzero().reshape(-1).tolist()


@dataclass
class PoolingRequestState:
    req_id: str
    prompt_token_ids: list[int]
    pooling_params: PoolingParams = PoolingParams()

    def __post_init__(self):
        self.num_prompt_tokens = len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens


class PoolingInputBatch(BaseInputBatch[PoolingRequestState]):
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
    ):
        super().__init__(
            max_num_reqs,
            max_model_len,
            device,
            pin_memory,
            vocab_size,
        )
        self.pooling_params: dict[str, PoolingParams] = {}

    def get_available_index(self) -> int | None:
        return self._num_requests

    def add_request(
        self,
        request: "PoolingRequestState",
        req_index: int | None = None,
    ) -> int:
        req_index = super().add_request(request, req_index)

        assert request.pooling_params is not None
        self.pooling_params[request.req_id] = request.pooling_params
        return req_index

    def clear_requests(self):
        """
        Clear the batch, mostly used by static batching
        """
        super().clear_requests()
        self.pooling_params = {}

    def remove_request(self, req_id: str):
        req_index = super().remove_request(req_id)
        if req_index is None:
            return

        self.pooling_params.pop(req_id, None)

    def make_pooling_metadata(self) -> PoolingMetadata:
        prompt_token_ids = self._make_prompt_token_ids_tensor()

        # Note, for now this assumes that all request in the batch
        # are either sampling or pooling requests
        assert len(self.requests_ids) == len(self.pooling_params)
        pooling_params = [self.pooling_params[req_id] for req_id in self.requests_ids]

        return PoolingMetadata(
            prompt_lens=torch.from_numpy(self._get_num_prompt_tokens()).to(self.device),
            prompt_token_ids=prompt_token_ids,
            pooling_params=pooling_params,
            pooling_states=[],
            prompt_token_ids_cpu=(
                prompt_token_ids.cpu()
                if prompt_token_ids is not None
                and any(p.requires_token_ids for p in pooling_params)
                else None
            ),
        )
