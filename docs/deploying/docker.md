# Using Docker

<!--
TODO: Add section on RHOAI officially supported images, once they exist
!-->

## Spyre base images

Base images containing the driver stack for IBM Spyre accelerators are available from the [ibm-aiu](https://quay.io/repository/ibm-aiu/) organization on Quay. This includes the `torch_sendnn` package, which is required for using torch with Spyre cards.

!!! attention
    These images contain an install of the `torch` package. The specific version installed is guaranteed to be compatible with `torch_sendnn`. Overwriting this install with a different version of `torch` may cause issues.

## Using community built images

Community maintained images are also [available on Quay](https://quay.io/repository/ibm-aiu/sendnn-inference?tab=tags), the latest x86 build is `quay.io/ibm-aiu/sendnn-inference:latest.amd64`.

!!! caution
    These images are provided as a reference and come with no support guarantees.

## Building SenDNN Inference's Docker Image from Source

You can build and run SenDNN Inference from source via the provided <gh-file:docker/Dockerfile.amd64>. To build SenDNN Inference:

```shell
DOCKER_BUILDKIT=1 docker build . --target release --tag vllm/sendnn-inference --file docker/Dockerfile.amd64
```

!!! note
    This Dockerfile currently only supports the x86 platform

## Running SenDNN Inference in a Docker Container

To run your SenDNN Inference image on a host with Spyre cards installed:

```shell
$ docker run \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /dev/vfio:/dev/vfio \
    -p 8000:8000 \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    vllm/sendnn-inference <model> <args...>
```

To run your SenDNN Inference image on a host without Spyre cards installed:

```shell
$ docker run \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    --env "SENDNN_INFERENCE_DYNAMO_BACKEND=eager" \
    vllm/sendnn-inference <model> <args...>
```
