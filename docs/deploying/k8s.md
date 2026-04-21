# Using Kubernetes

The vLLM Documentation on [Deploying with Kubernetes](https://docs.vllm.ai/en/latest/deployment/k8s.html) is a comprehensive guide for configuring deployments of models on kubernetes. This guide highlights some key differences when deploying on kubernetes with Spyre accelerators.

## Deploying on Spyre Accelerators

!!! note
    **Prerequisite**: Ensure that you have a running Kubernetes cluster with Spyre accelerators.

<!-- TODO: Link to public docs for cluster setup -->

1. (Optional) Create PVCs and secrets for vLLM.

      ```yaml
      apiVersion: v1
      kind: PersistentVolumeClaim
      metadata:
        name: hf-cache
        namespace: default
      spec:
        accessModes:
        - ReadWriteOnce
        resources:
          requests:
            storage: 50Gi
        storageClassName: default
        volumeMode: Filesystem
      ---
      apiVersion: v1
      kind: PersistentVolumeClaim
      metadata:
        name: graph-cache
        namespace: default
      spec:
        accessModes:
        - ReadWriteOnce
        resources:
          requests:
            storage: 50Gi
        storageClassName: default
        volumeMode: Filesystem
      ---
      apiVersion: v1
      kind: Secret
      metadata:
        name: hf-token-secret
        namespace: default
      type: Opaque
      stringData:
        token: "REPLACE_WITH_TOKEN"
      ```

2. Create a deployment and service for the model you want to deploy. This example demonstrates how to deploy `ibm-granite/granite-3.3-8b-instruct`.

      ```yaml
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: granite-8b-instruct
        namespace: default
        labels:
          app: granite-8b-instruct
      spec:
        # Defaults to 600 and must be set higher if your startupProbe needs to wait longer than that 
        progressDeadlineSeconds: 1200
        replicas: 1
        selector:
          matchLabels:
            app: granite-8b-instruct
        template:
          metadata:
            labels:
              app: granite-8b-instruct
          spec:
            # Required for scheduling spyre cards
            schedulerName: aiu-scheduler
            volumes:
            - name: hf-cache-volume
              persistentVolumeClaim:
                claimName: hf-cache
            # vLLM needs to access the host's shared memory for tensor parallel inference.
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "2Gi"
            # vLLM can cache model graphs previously compiled on Spyre cards
            - name: graph-cache-volume
              persistentVolumeClaim:
                claimName: graph-cache
            containers:
            - name: vllm
              image: quay.io/ibm-aiu/sendnn-inference:latest.amd64
              args: [
                "ibm-granite/granite-3.3-8b-instruct"
              ]
              env:
              - name: HUGGING_FACE_HUB_TOKEN
                valueFrom:
                  secretKeyRef:
                    name: hf-token-secret
                    key: token
              - name: TORCH_SENDNN_CACHE_ENABLE
                value: "1"
              - name: TORCH_SENDNN_CACHE_DIR
                value: /root/.cache/torch
              ports:
              - containerPort: 8000
              resources:
                limits:
                  cpu: "10"
                  memory: 20G
                  ibm.com/aiu_pf: "1"
                requests:
                  cpu: "2"
                  memory: 6G
                  ibm.com/aiu_pf: "1"
              volumeMounts:
              - mountPath: /root/.cache/huggingface
                name: hf-cache-volume
              - mountPath: /dev/shm
                name: shm
              - mountPath: /root/.cache/torch
                name: graph-cache-volume
              livenessProbe:
                httpGet:
                  path: /health
                  port: 8000
                periodSeconds: 10
              readinessProbe:
                httpGet:
                  path: /health
                  port: 8000
                periodSeconds: 5
              startupProbe:
                httpGet:
                  path: /health
                  port: 8000
                periodSeconds: 10
                # Long startup delays are necessary for graph compilation
                failureThreshold: 120
      ---
      apiVersion: v1
      kind: Service
      metadata:
        name: granite-8b-instruct
        namespace: default
      spec:
        ports:
        - name: http-granite-8b-instruct
          port: 80
          protocol: TCP
          targetPort: 8000
        selector:
          app: granite-8b-instruct
        sessionAffinity: None
        type: ClusterIP
      ```

3. Deploy and Test

      Apply the manifests using `kubectl apply -f <filename>`:

      ```console
      kubectl apply -f pvcs.yaml
      kubectl apply -f deployment.yaml
      ```

      To test the deployment, run the following `curl` command:

      ```console
      curl http://granite-8b-instruct.default.svc.cluster.local/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
              "model": "ibm-granite/granite-3.3-8b-instruct",
              "prompt": "San Francisco is a",
              "max_tokens": 7,
              "temperature": 0
            }'
      ```

      If the service is correctly deployed, you should receive a response from the vLLM model.
