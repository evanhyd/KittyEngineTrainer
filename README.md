# KittyEngineTrainer

High-performance GPU-accelerated NNUE trainer for **KittyEngine**, powered by [Bullet](https://github.com/jw1912/bullet) and CUDA. Features a config-driven architecture, automated checkpoint evaluation across multi-pack streams, and standalone tournament playtest weight preservation.

---

## 🧠 Architecture & Specifications

| Property | Value |
| :--- | :--- |
| **Feature Transformer** | HalfKA Folded (`24,576` inputs: 32 king squares × 768 piece planes) |
| **Dual Accumulator** | `256` neurons per perspective (`512` combined) |
| **Hidden Layer 1** | `512 -> 32` (Dense + SCReLU) |
| **Hidden Layer 2** | `32 -> 32` (Dense + SCReLU) |
| **Output Layer** | `32 -> 1 x 8` Material Count Buckets |
| **Activation** | Squared Clipped ReLU (SCReLU: $\text{clamp}(x, 0, 1)^2$) |
| **Weight Clamping** | AdamW bounds strictly enforced at $[-1.98, +1.98]$ |
| **Evaluation Scale** | $400.0\text{ cp}$ centipawn evaluation scale |
| **Binary Weight Layout**| `NNUESGD6` format (exactly `25,237,864 bytes`) |

---

## ⚙️ Configuration (`config.yaml`)

Training parameters, learning rate decay, validation batch sizes, positive bias initializations, and tournament export intervals are centrally managed in `config.yaml`:

```yaml
# Total target superbatches for the training run
superbatches: 200

# WDL score distillation factor (0.0 = pure centipawns, 1.0 = pure WDL outcome)
wdl: 0.20

# Peak learning rate for Cosine Annealing decay
lr: 0.0025

# Minimum learning rate at cycle completion
min_lr: 0.0005

# Training batch size in positions (GTX 1660 Ti safe: 16384)
batch_size: 16384

# Validation positions evaluated per pack during checkpoint evaluation
eval_positions_per_pack: 2000000

# Periodic weight export interval
# Automatically exports and preserves 'checkpoints/weights_step_<N>.bin' every N superbatches
# Set to 1 to preserve every superbatch for standalone engine playtesting tournaments (SPRT)
save_interval: 1

# Positive Bias Initialization (Layer 0, Layer 1, Layer 2)
bias_l0: 0.05
bias_l1: 0.35
bias_l2: 0.35
```

---

## 🚀 Command Reference

The trainer provides a clean, single-word command interface:

### Core Commands

| Command | Usage | Description |
| :--- | :--- | :--- |
| **`status`** | `cargo run --release -- status` | **Instant inspection (<100ms)**: Displays active configuration, current champion metrics, latest Bullet step, and recent training history. |
| **`resume`** | `cargo run --release -- resume` | **One-click resume**: Auto-detects the latest checkpoint in `checkpoints/`, sets the next step, preserves smooth Cosine Annealing, and trains up to `superbatches`. |
| **`train`** | `cargo run --release -- train` | **Start fresh**: Initializes fresh weights with positive biases (`0.05, 0.35, 0.35`), streams data from `data/`, and evaluates checkpoints after every superbatch. |
| **`check`** | `cargo run --release -- check [weights.bin]` | **Unified audit**: Runs empirical Dead Neuron telemetry across all layers and detects high-centipawn polar sign disagreements against ground truth. |
| **`eval`** | `cargo run --release -- eval [weights.bin]` | **Benchmark evaluation**: Measures MAE, SMSE, log-loss (BCE), MSE, RMSE, and mating accuracy across all dataset packs in `data/`. |
| **`export`** | `cargo run --release -- export [step] [out.bin]` | **Weight export**: Converts any Bullet checkpoint (e.g. `export 115`) or raw checkpoint into KittyEngine binary format (`weights.bin` or custom name). |
| **`compare`**| `cargo run --release -- compare [checkpoint_dir]`| **Numerical parity**: Validates GPU vs CPU mathematical parity on standard benchmark FEN positions. |
| **`scratch`**| `cargo run --release -- scratch [out.bin]` | Generates positive-bias scratch weights for initialization. |
| **`info`** | `cargo run --release -- info` | Displays neural network architecture specifications and memory layouts. |

---

## 📁 Standardized Project Layout

Paths are permanently standardized to keep the workflow clean and simple:

```text
KittyEngineTrainer/
├── config.yaml          # Active training configuration & hyperparameters
├── data/                # Standardized training datasets (*.binpack)
├── checkpoints/         # Checkpoints, champion weights, & playtest weight library
│   ├── champion_weights.bin     # Active benchmark champion
│   ├── baseline.json            # Champion metric record
│   ├── training_history.json    # Full step-by-step training telemetry log
│   ├── kitty_bullet_screlu-*/   # Bullet GPU checkpoints for seamless resuming
│   └── weights_step_*.bin       # Preserved tournament weights for playtesting
├── src/
│   ├── main.rs          # CLI entry point, scheduler, and commands
│   ├── config.rs        # Configuration loader, defaults, and checkpoint detection
│   ├── inputs.rs        # KittyHalfKA folded feature mapper (24,576 inputs)
│   ├── eval.rs          # Multi-threaded validation engine and metric computation
│   ├── export.rs        # Weight conversion between Bullet and NNUESGD6
│   ├── convert.rs       # JSON dataset to binary pack chunker
│   └── diagnostics.rs   # Layer-by-layer empirical dead neuron analyzer
├── Cargo.toml           # Dependencies (bullet_lib, CUDA, rayon, serde, serde_yaml)
└── README.md            # Project documentation & usage guide
```
