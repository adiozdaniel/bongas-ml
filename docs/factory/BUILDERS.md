# 🛡️ Factory Builders: Integrated Binary Security

[🏠 Hub](../README.md) | [🏗️ Factory](./WORKFLOWS.md) | [🛡️ Builders](./BUILDERS.md)

---

## The Goal

Ensure **Intellectual Property Protection** and **System Integrity** by utilizing Rust's native compilation and the integrated Sovereign Guard.

## Transition to Native Security (Symphony 3.0)

With the migration to a **100% Rust-native ML stack**, the legacy Cython-based `trainer.so` obfuscation process has been deprecated.

1. **Native Machine-Code Obfuscation:** By compiling all training and security logic into the `bongas-ai` Rust binary, we achieve deep protection through machine-level code generation.
2. **Integrated Sovereign Guard:** The binary now includes native modules for:
    * **SHA-256 Integrity Monitoring**
    * **Mutual TLS Heartbeats**
    * **Silent Hot-Swapping**
3. **Zero Python Dependencies:** The engine no longer requires a local Python environment or external shared objects, drastically reducing the attack surface.

### Legacy Build Process (Deprecated)

*Previously, we used `factory/builders/compile_trainer.py` to generate `trainer.so`. This is no longer required for Symphony 3.0 deployments.*

---
[🏠 Hub](../README.md) | [🔝 Top](#-factory-builders-integrated-binary-security)
