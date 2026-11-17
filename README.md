# KittyEngineTrainer

High-performance GPU-accelerated NNUE trainer for **KittyEngine**, powered by [Bullet](https://github.com/jw1912/bullet) and CUDA. Features a config-driven pipeline, multi-pack validation streaming, standalone playtest weight archiving, and automated Cutechess-CLI Sequential Probability Ratio Test (SPRT) verification as the exclusive gate for champion promotion.

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

## ⚔️ Automated SPRT Verification & Promotion

KittyEngineTrainer uses live engine-vs-engine playtesting as the **exclusive gate** for champion weight promotion. Static validation loss (MAE/SMSE) is monitored for health checks, but a candidate weight is promoted to champion **only** if it statistically proves superiority over the active champion in game play.

### How It Works

1. **Hypothesis Testing**: Evaluates $H_0: \Delta\text{Elo} \le \text{elo0}$ against $H_1: \Delta\text{Elo} \ge \text{elo1}$ with significance parameters $\alpha$ (Type I error) and $\beta$ (Type II error).
2. **Zero-Binary Sandboxing**: Candidate and champion weights are staged as `weights.bin` in isolated temporary sandboxes (`checkpoints/sprt_sandbox/candidate` and `checkpoints/sprt_sandbox/champion`) alongside `search_config.yaml`. The engine binary runs from its build directory with `dir=` set to the sandbox, preventing file-locking collisions and binary duplication.
3. **Automated Adjudication**: Fast-forwards obvious positions using Polyglot opening books, draw adjudication (e.g. 8 consecutive plies with score $\le 10\text{ cp}$ after move 40), and resign adjudication (score $\le -550\text{ cp}$ for 4 plies).
4. **Promotion Decision**:
   - **$H_1$ Accepted**: The candidate is statistically superior $\to$ instantly promoted to `checkpoints/champion_weights.bin` and recorded in `baseline.json`.
   - **$H_0$ Accepted / Inconclusive**: Candidate failed to beat the champion $\to$ champion retained; unpromoted checkpoint weights are cleaned up unless `save_interval` preserves them.

---

## 📊 Telemetry & Logging

All logs use standard default paths under `checkpoints/` without configuration clutter:

| Log File | Format | Description |
| :--- | :--- | :--- |
| **`checkpoints/sprt_matches.log`** | Human-readable text audit | Timestamped log of every SPRT match, including candidate vs. champion paths, final verdicts, scores (+W -L =D), win percentages, Elo estimates with error margins, SPRT LLR, and explicit promotion notices (`👑 CHAMPION PROMOTED`). |
| **`checkpoints/sprt_games.pgn`** | Standard chess PGN | Live-flushed (`fi`) by Cutechess after every finished game. Records full PGN move history, search depths, evaluations, clock times, book annotations, and termination reasons for replay in any chess GUI. |
| **`checkpoints/baseline.json`** | Structured JSON | Tracks active champion metrics (MAE, SMSE, WDL) and verified SPRT match telemetry. |
| **`checkpoints/training_history.json`** | Structured JSON array | Complete chronological history of all superbatch steps, validation metrics, promotion outcomes, and SPRT statistics. |
| **Console Stream** | Real-time stdout | Live updates of finished games, running match scores, and current SPRT LLR bounds. |

---

## ⚙️ Configuration (`config.yaml`)

Training parameters, learning rate decay, validation batch sizes, positive bias initializations, and SPRT match settings are centrally configured:

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
# Set to 1 to preserve every superbatch for tournament playtesting
save_interval: 1

# Positive Bias Initialization (Layer 0, Layer 1, Layer 2)
bias_l0: 0.05
bias_l1: 0.35
bias_l2: 0.35

# ==============================================================================
# SPRT (Sequential Probability Ratio Test) Automated Verification
# ==============================================================================
# SPRT is the EXCLUSIVE gate for champion promotion.
sprt:
  enabled: true
  cutechess_path: "C:/Users/evanh/works/software/cutechess/cutechess-cli.exe"
  engine_path: "C:/Users/evanh/Desktop/KittyEngineV5/x64/Release/KittyEngineV5.exe"
  book_path: "C:/Users/evanh/works/software/cutechess/book/book.bin"
  book_depth: 8
  time_control: "5+0.1"
  concurrency: 2
  max_rounds: 200
  elo0: 0.0
  elo1: 5.0
  alpha: 0.05
  beta: 0.05
  draw_movenumber: 40
  draw_movecount: 8
  draw_score: 10
  resign_movecount: 4
  resign_score: 550
```

---

## 🚀 Command Reference

| Command | Usage | Description |
| :--- | :--- | :--- |
| **`status`** | `cargo run --release -- status` | **Instant inspection (<100ms)**: Displays active configuration, SPRT settings, current champion metrics, latest Bullet step, and recent training history. |
| **`resume`** | `cargo run --release -- resume` | **One-click resume**: Auto-detects the latest checkpoint in `checkpoints/`, sets the next step, preserves smooth Cosine Annealing, and trains up to `superbatches`. |
| **`train`** | `cargo run --release -- train` | **Start fresh**: Initializes fresh weights with positive biases (`0.05, 0.35, 0.35`), streams data from `data/`, evaluates checkpoints, and runs automated SPRT verification for champion promotion. |
| **`sprt`** | `cargo run --release -- sprt [cand] [champ]` | **Playtesting verification**: Runs an automated Cutechess SPRT match between candidate and champion weights with live score and Elo telemetry. Defaults to latest step vs champion. |
| **`check`** | `cargo run --release -- check [weights.bin]` | **Unified audit**: Runs empirical Dead Neuron telemetry across all layers and detects high-centipawn polar sign disagreements against ground truth. |
| **`eval`** | `cargo run --release -- eval [weights.bin]` | **Benchmark evaluation**: Measures MAE, SMSE, log-loss (BCE), MSE, RMSE, and mating accuracy across all dataset packs in `data/`. |
| **`export`** | `cargo run --release -- export [step] [out.bin]` | **Weight export**: Converts any Bullet checkpoint (e.g. `export 115`) into KittyEngine binary format (`weights.bin` or custom name). |
| **`compare`**| `cargo run --release -- compare [checkpoint_dir]`| **Numerical parity**: Validates GPU vs CPU mathematical parity on standard benchmark FEN positions. |
| **`scratch`**| `cargo run --release -- scratch [out.bin]` | Generates positive-bias scratch weights for initialization. |
| **`info`** | `cargo run --release -- info` | Displays neural network architecture specifications and memory layouts. |

---

## 📁 Standardized Project Layout

```text
KittyEngineTrainer/
├── config.yaml          # Active training configuration & SPRT settings
├── data/                # Standardized training datasets (*.binpack)
├── checkpoints/         # Checkpoints, champion weights, & playtest weight library
│   ├── champion_weights.bin     # Active benchmark champion
│   ├── baseline.json            # Champion metric record & SPRT telemetry
│   ├── training_history.json    # Step-by-step training & SPRT history log
│   ├── sprt_matches.log         # Human-readable SPRT match & promotion audit
│   ├── sprt_games.pgn           # Complete PGN game replay archive
│   ├── kitty_bullet_screlu-*/   # Bullet GPU checkpoints for seamless resuming
│   └── weights_step_*.bin       # Preserved tournament weights for playtesting
├── src/
│   ├── main.rs          # CLI entry point, scheduler, and commands
│   ├── sprt.rs          # Automated Cutechess SPRT match pipeline & parser
│   ├── config.rs        # Configuration loader, defaults, and checkpoint detection
│   ├── inputs.rs        # KittyHalfKA folded feature mapper (24,576 inputs)
│   ├── eval.rs          # Multi-threaded validation engine and metric computation
│   ├── export.rs        # Weight conversion between Bullet and NNUESGD6
│   ├── convert.rs       # JSON dataset to binary pack chunker
│   └── diagnostics.rs   # Layer-by-layer empirical dead neuron analyzer
├── Cargo.toml           # Dependencies (bullet_lib, CUDA, rayon, serde, serde_yaml)
└── README.md            # Project documentation & usage guide
```
