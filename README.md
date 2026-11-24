# KittyEngineTrainer

High-performance GPU NNUE trainer and native SPSA search heuristic optimizer for **KittyEngine**.

---

## 🧠 Architecture

| Component | Specification |
| :--- | :--- |
| **Feature Transformer** | HalfKA Folded (`24,576` inputs: 32 king squares × 768 piece planes) |
| **Dual Accumulator** | `256` neurons per perspective (`512` combined) |
| **Pairwise (Hadamard)** | `(128 + 128) -> 256` (Pairwise interaction within STM/NTM) |
| **Hidden Layers** | `256 -> 64` (SCReLU) $\to$ `64 -> 32` (SCReLU) |
| **Output Layer** | `32 -> 1 x 8` Material Count Buckets |
| **Weight Format** | `NNUESGD6` binary format (25,242,088 bytes) |

---

## 🚀 Quick Start

```bash
# 1. Train NNUE (auto-resumes from latest checkpoint)
cargo run --release -- train

# 2. Run SPSA search tuner (auto-resumes with saved state)
cargo run --release -- spsa
```

---

## 📖 CLI Commands

### 🏋️ NNUE Training

| Command | Usage | Description |
| :--- | :--- | :--- |
| **`train`** | `cargo run --release -- train` | Train NNUE with automated SPRT promotion (pass `--new` to start fresh). |
| **`stop`** | `cargo run --release -- stop` | Gracefully stop active training after current superbatch without corruption. |
| **`status`** | `cargo run --release -- status` | View active champion metrics, training progress, and SPRT history. |
| **`export`** | `cargo run --release -- export [step] [out.bin]` | Export Bullet checkpoint to KittyEngine binary format (default: `weights.bin`). |

### 🎯 SPSA Search Tuner

| Command | Usage | Description |
| :--- | :--- | :--- |
| **`spsa`** | `cargo run --release -- spsa` | Run continuous SPSA optimization loop with clean in-place scoreboard. |
| **`spsa status`** | `cargo run --release -- spsa status` | Compare continuous SPSA parameters against deployed desktop config. |
| **`spsa verify`** | `cargo run --release -- spsa verify` | Run SPRT verification match against desktop config and deploy if superior. |
| **`spsa deploy`** | `cargo run --release -- spsa deploy` | Force-deploy current SPSA parameters to Desktop `search_config.yaml`. |
| **`spsa reset`** | `cargo run --release -- spsa reset [--force]` | Reset SPSA state back to deployed desktop configuration. |

### 🔬 Diagnostics & Evaluation

| Command | Usage | Description |
| :--- | :--- | :--- |
| **`sprt`** | `cargo run --release -- sprt [cand] [champ]` | Run Cutechess SPRT match between two weight files. |
| **`check`** | `cargo run --release -- check [weights.bin]` | Run comprehensive health checks (Dead Neurons + Inversions). |
| **`eval`** | `cargo run --release -- eval [weights.bin]` | Benchmark MAE, MSE, and mating accuracy across all packs in `data/`. |
| **`compare`** | `cargo run --release -- compare [ckpt_dir]` | Validate GPU vs CPU mathematical forward-pass parity. |
| **`scratch`** | `cargo run --release -- scratch [out.bin]` | Generate positive-bias scratch weights for initialization. |
| **`info`** | `cargo run --release -- info` | Display neural network architecture and layout specs. |

---

## ⚙️ Configuration Files

- **`config.yaml`**: Training hyperparameters (learning rate, batch size, WDL distillation, save interval) and trainer SPRT settings (`engine_name` resolved under `engine/`).
- **`spsa_settings.json`**: SPSA tuner settings (`engine_name` resolved under `engine/`, time control, concurrency, games per iteration, SPRT verification bounds).
