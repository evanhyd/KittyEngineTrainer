# KittyEngineTrainer

High-performance GPU-accelerated NNUE trainer for **KittyEngine**, powered by the [Bullet](https://github.com/jw1912/bullet) training library and CUDA.

---

## Architecture & Specifications

| Property | Value |
| :--- | :--- |
| **Feature Transformer** | HalfKA (`49,152` features: 64 king squares × 768 piece planes) |
| **Hidden Layers** | `(49152 -> 256) x 2 -> 32 -> 32 -> 1` |
| **Activation** | Squared Clipped ReLU (SCReLU: $\text{clamp}(x, 0, 1)^2$) |
| **Weight Clamping** | AdamW bounds strictly enforced at $[-1.98, +1.98]$ |
| **Output Scaling** | $400.0\text{ cp}$ centipawn evaluation scale |
| **Weight Format** | `NNUESGD5` binary layout (exactly `50,402,756 bytes`) |

---

## Prerequisites

- **Rust**: 1.80+ (2024 edition)
- **NVIDIA GPU**: CUDA 12+ runtime and drivers
- **C++ Compiler**: MSVC (Windows) or GCC/Clang (Linux)

---

## Quickstart Tutorial

### 1. Preparing Training Data
Convert raw JSON engine evaluation logs into memory-mapped binary chunks:
```bash
cargo run --release -- pack-chunks <path/to/eval.json> data/ 50000000
```

### 2. Training the Model
Train across multi-pack interleaved streams with cosine learning rate scheduling and automatic validation after every superbatch:

**Start from an existing checkpoint:**
```bash
cargo run --release -- train data/ --weights checkpoints/kitty_bullet_screlu-609 --start-superbatch 610 --superbatches 640 --lr 0.000015 --wdl 0.20
```

**Train from scratch:**
```bash
cargo run --release -- train data/ --superbatches 36 --lr 0.0025 --wdl 0.20
```

### 3. Evaluating Weights
Stream full validation across all dataset packs to measure MAE, SMSE, BCE, and mating accuracy:
```bash
cargo run --release -- eval checkpoints/weights_screlu_step609.bin data/ 2000000
```

### 4. Exporting & Importing Checkpoints
Convert Bullet's GPU `raw.bin` directly into KittyEngine's binary weight format:
```bash
cargo run --release -- export checkpoints/kitty_bullet_screlu-609/raw.bin weights.bin
```

Convert a KittyEngine `weights.bin` back into Bullet format for fine-tuning:
```bash
cargo run --release -- import weights.bin checkpoints/init_bullet_weights.bin
```

### 5. Diagnostics & Quality Audits
- **Dead Neuron Detection**: Check activation telemetry across all layers to ensure gradient flow:
  ```bash
  cargo run --release -- dead-neurons weights.bin data/ 1000000
  ```
- **Inversion Scan**: Detect high centipawn evaluation sign disagreements against ground truth:
  ```bash
  cargo run --release -- inversions weights.bin data/ 1000000
  ```
- **Parity Check**: Validate numerical parity between GPU Bullet and CPU NNUE evaluations:
  ```bash
  cargo run --release -- compare checkpoints/kitty_bullet_screlu-609
  ```

---

## Project Structure

```text
KittyEngineTrainer/
├── src/
│   ├── main.rs         # CLI entry point, training loop, scheduler, and commands
│   ├── inputs.rs       # KittyEngine HalfKA feature mapper (49,152 inputs)
│   ├── eval.rs         # Multi-threaded validation engine and metric computation
│   ├── export.rs       # Checkpoint conversion between Bullet and NNUESGD5
│   ├── convert.rs      # JSON evaluation log to binary pack chunker
│   └── diagnostics.rs  # Layer-by-layer neuron activation and dead neuron analyzer
├── Cargo.toml          # Dependencies (bullet_lib with CUDA, rayon, serde)
└── README.md           # Documentation
```
