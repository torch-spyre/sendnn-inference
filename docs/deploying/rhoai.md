# Using Red Hat OpenShift AI

[Red Hat OpenShift AI](https://www.redhat.com/en/products/ai/openshift-ai) is a cloud-native AI platform that bundles together many popular model management projects, including [KServe](https://kserve.github.io/website/latest/).

This example shows how to use KServe with RHOAI to deploy a model on OpenShift, using a modelcar image to load the model without requiring any connection to Huggingface Hub.

## Deploying with KServe

!!! note "Prerequisites"
    * A running Kubernetes cluster with RHOAI installed
    * Image pull credentials for `registry.redhat.io/rhelai1`
    * Spyre accelerators available in the cluster

<!-- TODO: Link to public docs for cluster setup -->

1. Create a ServingRuntime to serve your models.

      ```yaml
        oc apply -f - <<EOF
        apiVersion: serving.kserve.io/v1alpha1
        kind: ServingRuntime
        metadata:
          name: sendnn-inference-runtime
          annotations:
            openshift.io/display-name: vLLM IBM Spyre ServingRuntime for KServe
            opendatahub.io/recommended-accelerators: '["ibm.com/aiu_pf"]'
          labels:
            opendatahub.io/dashboard: "true"
        spec:
          multiModel: false
          supportedModelFormats:
            - autoSelect: true
              name: vLLM
          containers:
            - name: kserve-container
              image: quay.io/ibm-aiu/sendnn-inference:latest.amd64
              args:
                - /mnt/models
                - --served-model-name={{.Name}}
              env:
                - name: HF_HOME
                  value: /tmp/hf_home
              ports:
                - containerPort: 8000
                  protocol: TCP
        EOF
      ```

2. Create an InferenceService for each model you want to deploy. This example demonstrates how to deploy the [Granite](https://www.ibm.com/granite) model `ibm-granite/granite-3.1-8b-instruct`.

      ```yaml
      oc apply -f - <<EOF
      apiVersion: serving.kserve.io/v1beta1
      kind: InferenceService
      metadata:
        annotations:
          openshift.io/display-name: granite-3-1-8b-instruct
          serving.kserve.io/deploymentMode: RawDeployment
        name: granite-3-1-8b-instruct
        labels:
          opendatahub.io/dashboard: 'true'
      spec:
        predictor:
          imagePullSecrets:
            - name: oci-registry
          maxReplicas: 1
          minReplicas: 1
          model:
            modelFormat:
              name: vLLM
            name: ''
            resources:
              limits:
                ibm.com/aiu_pf: '1'
              requests:
                ibm.com/aiu_pf: '1'
            runtime: sendnn-inference-runtime
            storageUri: 'oci://registry.redhat.io/rhelai1/modelcar-granite-3-1-8b-instruct:1.5'
            volumeMounts:
              - mountPath: /dev/shm
                name: shm
          schedulerName: aiu-scheduler
          tolerations:
            - effect: NoSchedule
              key: ibm.com/aiu_pf
              operator: Exists
                    spec:
          volumes:
            # This volume may need to be larger for bigger models and running tensor-parallel inference with more cards
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "2Gi"
      EOF
      ```

3. To test your InferenceService, refer to the [KServe documentation on model inference with vLLM](https://kserve.github.io/website/latest/modelserving/v1beta1/llm/huggingface/text_generation/#perform-model-inference_1).
