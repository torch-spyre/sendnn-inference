# Plugin Architecture

The Spyre plugin extends or replaces three main components in vLLM:

1. Scheduler
2. Model worker and model runner
3. Modeling code

To better understand these modifications, it's helpful to
consider the state of the native vllm for GPU architecture.

![vLLM architecture](images/vllm_v1.svg)

The API server, the engine core, and the workers live in
different processes. All three refer to the platform API for backend
specific concerns.

In SenDNN Inference, we implement a platform API that is
loaded at the vLLM startup time and bootstraps all other components.

![SenDNN Inference architecture](images/vllm_v1_spyre.svg)

As we can see in the diagram, the plugin mainly modifies the engine core
and worker processes. The platform API includes request validation hooks
that the API server invokes to ensure that the requests
can be handled by the backend.

In the engine core, we customize the scheduler to handle the specific batching
constraints for Spyre cards.

The changes are broader in the worker process. Most of the main
classes have Spyre-specific implementations. From the vLLM code, we mainly
reuse the sampling code (including logits processing) and the pooling
code for non-generative use cases.

We provide model runners for two cases: generative models with chunked prefill, and pooling
models with static batching. The pooling model runner uses the `transformers` modeling code
instead of the `foundation-model-stack` code.
