# Multimodal Models in SenDNN Inference

In order to understand how to get multimodal models running through SenDNN Inference, it is important to understand the differences between how models are implemented in vLLM & SenDNN Inference. To illustrate this, we use `llava_next` as an example, because `granite vision` is the only multimodal model currently supported.

NOTE: for those unfamiliar, granite vision is a special instance of llava next, and tends to run as an instance of llava next. The primary differences are:

- For the LLM, we use a granite LLM.
- For the vision encoder, we use SigLIP instead of CLIP.
- Instead of taking the output of one feature layer from the vision encoder to form the visual features, we instead take the output of multiple layers and concatenate them.

## For vLLM

In vLLM, models are implemented as their own model class. The class implementation generally inherits from `SupportsMultiModal`, and importantly, it registers multimodal processing information.

```python
@MULTIMODAL_REGISTRY.register_processor(
    LlavaNextMultiModalProcessor,
    info=LlavaNextProcessingInfo,
    dummy_inputs=LlavaDummyInputsBuilder,
)
```

If you are coming from a background of working with non-multimodal models, the more important pieces to be aware of are how the preprocessing differs, and how things differ in prefill. More specifically:

- While text only models typically use a tokenizer, multimodal models generally have interleaved inputs. The manner in which this is accomplished is by using a *model specific* token that indicates that the corresponding positions should be replaced with multimodal features. Logically this essentially means something like the following:

    - Given the text: `<image> Describe this image` & an example image
    - We preprocess the text *and* the example image
    - Then, we run the preprocessed image through the corresponding part of the model for encoding that modality, e.g., vision encoder + projector
    - Finally, we create merged multimodal embeddings, where the indices for the special `<image>` token are replaced with the extracted visual features, and the non multimodal tokens have embeddings extracted from the LLM

This has a few implications that may be nonobvious. Namely:

1. A picture is worth a lot of tokens; The multimodal features corresponding to each special token are not a single embedding, and tend to vary based on a few factors, e.g., aspect ratio / image size. Bigger images tend to take up more context.

2. Because of the above ^, an expansion step is generally needed to offset the input IDs. For example, if the `<image>` token represents an image that will take up `num_features` in the context, we can replace the `<image>` with `<image>`*`num_features`; this is done in the vLLM model specific preprocessing class & related utils, and lets us to directly mask the extracted multimodal features into the embeddings.

3. Multimodal is most relevant at prefill time, because at decode time, we simply have embeddings from the space of the LLM, and we do not need to encode the multimodal data again. As such, the original data can essentially be dropped after encoding during prefill.

4. Due to the nature of how multimodal embeddings are merged, the model needs to be able to accept embeddings as inputs, and not just token IDs.

5. As a result of ^, we must be careful to handle warmup correctly with respect to `torch.compile`, *especially* when it comes to AIU. More details on this below.

For more extensive documentation in how to implement multimodal in vLLM, see the [official docs for multimodal on vLLM](https://docs.vllm.ai/en/latest/contributing/model/multimodal) - the above is mostly meant as context for how think of these models with respect to SenDNN Inference.

## Extending to SenDNN Inference

In SenDNN Inference, models are implemented with a generic wrapper around FMS; the implementation is *not* model specific. This adds several points of awkwardness in porting multimodal FMS wrappers into SenDNN Inference. In general, the best way to get the model working is as follows:

1. Make sure it runs correctly with vLLM and the HuggingFace implementation *before* porting the FMS implementation into SenDNN Inference.*

2. In `sendnn_inference.multimodal.mm_mappings`, create a new utils class for the model architecture and map FMS/Transformers configs to it in `MM_CFG_MAPPING`.

3. Implement the abstract methods for warmup features, multimodal embedding creation and so on.

** Aside from uniformity, the main reason it's desirable to get the model running in vLLM *before* SenDNN Inference is that even though the model implementation is different, the preprocessor that vLLM uses to initialize it when it is running through SenDNN Inference is based on the underlying config, and is the *same*. This means that to implement the model in FMS, we do not have to reimplement any of the preprocessing wrapping or prompt substitution/multimodal token expansion logic, which is very well patterned in vLLM. This is ideal for keeping changes for specific model architectures in our generic wrapper to a minimum.

### FAQ

- If things aren't working correctly, where should I start?
    - Ensure the text config is being correctly unwrapped and that the model instance is being recognized as `is_multimodal`. This will cause prefill/decode to use embeddings as inputs instead of token IDs, even in the case when only text is provided.

    - Ensure that the preprocessor object being used by the wrapping LLM class is the correct one, otherwise your inputs may be prepared incorrectly.

    - Based on the above, verify that you are handling the dictionary (e.g., `pixel_values` etc) to be passed to FMS in its input preparation correctly; as FMS currently only has one multimodal model, the interface and design patterns may not be stable yet.

    - Ensure that the results of prefill/decode are actually embeddings; if you pass things like the wrong `iteration` to FMS, it is easy to do things like getting embeddings in prefill, then getting input IDs in decode by mistake, which can cause confusing compiler errors.

    - Test without compile first. If all of the above are correct and compile is still running into issues, ensure that your warmup features also include multimodal inputs and not just embeddings, because you need to ensure all parts of the model are traced properly. If you pass something like pre-merged embeddings, it's the same as just passing text embeddings since the vision encoder won't be used, so it's important to pass the raw multimodal objects.

- What is the state of multimodal support with respect to model runners?
    - Currently it's supported for generative models with chunked prefill. It is *not* yet enabled for the pooling model runner.

- There is a new model runner! How do I add multimodal support to it?
    - Ensure multimodal features are passed all the way through
    - Conditionally use embeddings in prefill if it's multimodal; you should do this after all of the runner specific manipulation for padding etc.
    - Conditionally use embeddings in decode; careful not to re-encode multimodal features in decode steps, since we should typically *only* consider multimodality at prefill time.
