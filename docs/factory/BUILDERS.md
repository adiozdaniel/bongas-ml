# 🛡️ Factory Builders: The Obfuscation Process

[🏠 Hub](../README.md) | [🏗️ Factory](./WORKFLOWS.md) | [🛡️ Builders](./BUILDERS.md)

---

## The Goal

Convert proprietary PyTorch training loops into opaque binary blobs (`trainer.so`) using Cython.

## The Cython Build Process

1. **Define Target:** `factory/trainer/` modules.
2. **Transpile:** Convert Python source to C.
3. **Compile:** Generate `.so` shared objects.
4. **Obfuscate:** Directives strip docstrings and annotations.

### Build Command

```bash
python factory/builders/compile_trainer.py
```

---
[🏠 Hub](../README.md) | [🔝 Top](#- factory-builders-the-obfuscation-process)
