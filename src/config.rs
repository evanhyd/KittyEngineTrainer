use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const DATA_DIR: &str = "data";
pub const CHECKPOINTS_DIR: &str = "checkpoints";
pub const CHAMPION_FILE: &str = "checkpoints/champion_weights.bin";
pub const SCRATCH_FILE: &str = "checkpoints/scratch_weights_init.bin";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    pub superbatches: usize,
    pub wdl: f32,
    pub lr: f32,
    pub min_lr: f32,
    pub batch_size: usize,
    pub eval_positions_per_pack: usize,
    pub save_interval: usize,
    pub bias_l0: f32,
    pub bias_l1: f32,
    pub bias_l2: f32,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            superbatches: 200,
            wdl: 0.20,
            lr: 0.0025,
            min_lr: 0.0005,
            batch_size: 16_384,
            eval_positions_per_pack: 2_000_000,
            save_interval: 1,
            bias_l0: 0.05,
            bias_l1: 0.35,
            bias_l2: 0.35,
        }
    }
}

const DEFAULT_CONFIG_YAML: &str = r#"# ==============================================================================
# KittyEngineTrainer Configuration
# ==============================================================================

# Total target superbatches for the training run
superbatches: 200

# WDL lambda distillation factor (0.0 = pure centipawns, 1.0 = pure WDL outcome)
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
# Set to 1 to keep every superbatch, or 2, 5, etc. for tournament playtesting checkpoints
save_interval: 1

# Positive Bias Initialization (Layer 0, Layer 1, Layer 2)
bias_l0: 0.05
bias_l1: 0.35
bias_l2: 0.35
"#;

impl TrainerConfig {
    pub fn load_or_default() -> Self {
        let yaml_path = Path::new("config.yaml");
        let json_path = Path::new("config.json");

        if yaml_path.exists() {
            if let Ok(content) = fs::read_to_string(yaml_path) {
                if let Ok(cfg) = serde_yaml::from_str::<TrainerConfig>(&content) {
                    println!("[config] Loaded configuration from config.yaml");
                    return cfg;
                } else {
                    eprintln!("[config] Warning: Failed to parse config.yaml, using defaults.");
                }
            }
        } else if json_path.exists() {
            if let Ok(content) = fs::read_to_string(json_path) {
                if let Ok(cfg) = serde_json::from_str::<TrainerConfig>(&content) {
                    println!("[config] Loaded configuration from config.json");
                    return cfg;
                } else {
                    eprintln!("[config] Warning: Failed to parse config.json, using defaults.");
                }
            }
        } else {
            // Write default config.yaml for user convenience
            if let Err(e) = fs::write(yaml_path, DEFAULT_CONFIG_YAML) {
                eprintln!("[config] Warning: Failed to create default config.yaml: {}", e);
            } else {
                println!("[config] Created default config.yaml with recommended settings.");
            }
        }

        Self::default()
    }
}

/// Detects the highest completed superbatch step in the checkpoints directory.
/// Returns (highest_step, checkpoint_folder_path).
pub fn detect_latest_checkpoint() -> Option<(usize, PathBuf)> {
    let cp_dir = Path::new(CHECKPOINTS_DIR);
    if !cp_dir.exists() {
        return None;
    }

    let mut highest_step = 0;
    let mut highest_path = None;

    if let Ok(entries) = fs::read_dir(cp_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    if let Some(step_str) = name.strip_prefix("kitty_bullet_screlu-") {
                        if let Ok(step) = step_str.parse::<usize>() {
                            if step >= highest_step && path.join("raw.bin").exists() {
                                highest_step = step;
                                highest_path = Some(path);
                            }
                        }
                    }
                }
            }
        }
    }

    if highest_step > 0 && highest_path.is_some() {
        return Some((highest_step, highest_path.unwrap()));
    }

    // Fallback: check baseline.json or training_history.json
    let history_path = cp_dir.join("training_history.json");
    if history_path.exists() {
        if let Ok(content) = fs::read_to_string(&history_path) {
            if let Ok(records) = serde_json::from_str::<Vec<serde_json::Value>>(&content) {
                if let Some(last) = records.last() {
                    if let Some(step) = last.get("step").and_then(|v| v.as_u64()).map(|v| v as usize) {
                        let folder = cp_dir.join(format!("kitty_bullet_screlu-{}", step));
                        return Some((step, folder));
                    }
                }
            }
        }
    }

    None
}
