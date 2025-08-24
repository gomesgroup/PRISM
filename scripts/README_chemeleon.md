# CheMeleon + ChemProp usage

- Quick smoke test (CPU):
  ```bash
  chemprop train -i data/demo/chemeleon_demo.csv -o runs/chemeleon_demo \
    --target-columns y --from-foundation CheMeleon --epochs 1 \
    --warmup-epochs 0 --batch-size 1 --accelerator cpu --devices 1
  ```

- Helper script (auto-activates `chemprop-v2` if not active):
  ```bash
  ./scripts/finetune_chemeleon.sh data/demo/chemeleon_demo.csv y runs/chemeleon_demo_gpu \
    --epochs 10 --batch-size 64 --accelerator gpu --devices 1
  ```

- Embeddings without finetuning:
  ```python
  from external.chemeleon.chemeleon_fingerprint import CheMeleonFingerprint
  fpr = CheMeleonFingerprint(device="cuda")
  X = fpr(["C", "CCO"])  # shape (n, 2048)
  ```

Reference: `CheMeleon` repository [GitHub: JacksonBurns/chemeleon](https://github.com/JacksonBurns/chemeleon)
