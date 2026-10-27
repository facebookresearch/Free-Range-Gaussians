# Checkpoint

Place the model weights here as:

```text
checkpoint/frg_ckpt.pt
```

After approval, run these commands from the code checkout:

```bash
hf auth login
hf download Ahanio/free-range-gaussians frg_ckpt.pt --local-dir checkpoint
```

The inference checkpoint contains only `model_state_dict`, with the
original model tensors preserved. It does not include optimizer state or
training configuration.

SHA-256 of `frg_ckpt.pt`:

```text
0ef8b21860c44fab5c385d57f6f24e221060a463e3f41fce29672a9b2fd190a9
```

The Free-Range Gaussians weights are licensed under the
[FAIR Noncommercial Research License](../MODEL_LICENSE), including the FAIR
Acceptable Use Policy. The bundled DINOv2 encoder parameters retain their
upstream Apache 2.0 license. See the model repository's
[third-party notices](https://huggingface.co/Ahanio/free-range-gaussians/blob/main/THIRD_PARTY_NOTICES.md).

