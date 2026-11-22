use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const DATA_DIR: &str = "data";
pub const CHECKPOINTS_DIR: &str = "checkpoints";
pub const CHAMPION_FILE: &str = "checkpoints/champion_weights.bin";
pub const SCRATCH_FILE: &str = "checkpoints/scratch_weights_init.bin";
pub const SPRT_MATCH_LOG: &str = "checkpoints/sprt_matches.log";
pub const SPRT_GAMES_PGN: &str = "checkpoints/sprt_games.pgn";

pub const INPUT_SIZE: usize = 24_576;
pub const ACCUM_SIZE: usize = 256;
pub const PAIRWISE_SIZE: usize = 256;
pub const HIDDEN1_SIZE: usize = 64;
pub const HIDDEN2_SIZE: usize = 32;
pub const NUM_OUTPUT_BUCKETS: usize = 8;
pub const CHECKPOINT_SIZE: u64 = 25_242_088;

fn default_true() -> bool { true }
fn default_cutechess_path() -> String { "C:/Users/evanh/works/software/cutechess/cutechess-cli.exe".to_string() }
fn default_engine_path() -> String { "C:/Users/evanh/Desktop/KittyEngineV5/x64/Release/KittyEngineV5.exe".to_string() }
fn default_engine_args() -> String { String::new() }

fn deserialize_engine_args<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct StringOrVec;

    impl<'de> serde::de::Visitor<'de> for StringOrVec {
        type Value = String;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string, list of arguments, or empty")
        }

        fn visit_str<E>(self, s: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(s.to_string())
        }

        fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
        where
            S: serde::de::SeqAccess<'de>,
        {
            let mut list = Vec::new();
            while let Some(elem) = seq.next_element::<String>()? {
                list.push(elem);
            }
            Ok(list.join(" "))
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(String::new())
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(String::new())
        }
    }

    deserializer.deserialize_any(StringOrVec)
}
fn default_book_path() -> String { "C:/Users/evanh/works/software/cutechess/book/book.bin".to_string() }
fn default_book_depth() -> usize { 8 }
fn default_time_control() -> String { "5+0.1".to_string() }
fn default_concurrency() -> usize { 2 }
fn default_max_rounds() -> usize { 200 }
fn default_elo0() -> f64 { 0.0 }
fn default_elo1() -> f64 { 10.0 }
fn default_alpha() -> f64 { 0.05 }
fn default_beta() -> f64 { 0.15 }
fn default_draw_movenumber() -> usize { 40 }
fn default_draw_movecount() -> usize { 8 }
fn default_draw_score() -> i32 { 10 }
fn default_resign_movecount() -> usize { 4 }
fn default_resign_score() -> i32 { 550 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SprtConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_cutechess_path")]
    pub cutechess_path: String,
    #[serde(default = "default_engine_path")]
    pub engine_path: String,
    #[serde(default = "default_engine_args", deserialize_with = "deserialize_engine_args")]
    pub engine_args: String,
    #[serde(default = "default_book_path")]
    pub book_path: String,
    #[serde(default = "default_book_depth")]
    pub book_depth: usize,
    #[serde(default = "default_time_control")]
    pub time_control: String,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    #[serde(default = "default_max_rounds")]
    pub max_rounds: usize,
    #[serde(default = "default_elo0")]
    pub elo0: f64,
    #[serde(default = "default_elo1")]
    pub elo1: f64,
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    #[serde(default = "default_beta")]
    pub beta: f64,
    #[serde(default = "default_draw_movenumber")]
    pub draw_movenumber: usize,
    #[serde(default = "default_draw_movecount")]
    pub draw_movecount: usize,
    #[serde(default = "default_draw_score")]
    pub draw_score: i32,
    #[serde(default = "default_resign_movecount")]
    pub resign_movecount: usize,
    #[serde(default = "default_resign_score")]
    pub resign_score: i32,
}

impl Default for SprtConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            cutechess_path: default_cutechess_path(),
            engine_path: default_engine_path(),
            engine_args: default_engine_args(),
            book_path: default_book_path(),
            book_depth: default_book_depth(),
            time_control: default_time_control(),
            concurrency: default_concurrency(),
            max_rounds: default_max_rounds(),
            elo0: default_elo0(),
            elo1: default_elo1(),
            alpha: default_alpha(),
            beta: default_beta(),
            draw_movenumber: default_draw_movenumber(),
            draw_movecount: default_draw_movecount(),
            draw_score: default_draw_score(),
            resign_movecount: default_resign_movecount(),
            resign_score: default_resign_score(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    pub superbatches: usize,
    pub wdl: f32,
    pub lr: f32,
    pub min_lr: f32,
    pub batch_size: usize,
    #[serde(default)]
    pub batches_per_superbatch: Option<usize>,
    pub eval_positions_per_pack: usize,
    pub save_interval: usize,
    pub bias_l0: f32,
    pub bias_l1: f32,
    pub bias_l2: f32,
    #[serde(default)]
    pub sprt: SprtConfig,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            superbatches: 300,
            wdl: 0.08,
            lr: 0.0015,
            min_lr: 0.00005,
            batch_size: 16_384,
            batches_per_superbatch: Some(4096),
            eval_positions_per_pack: 100_000,
            save_interval: 15,
            bias_l0: 0.05,
            bias_l1: 0.35,
            bias_l2: 0.55,
            sprt: SprtConfig::default(),
        }
    }
}

const DEFAULT_CONFIG_YAML: &str = r#"# ==============================================================================
# KittyEngineTrainer Configuration (Optimized for High-Throughput Training)
# ==============================================================================

# Target 300 superbatches * 4096 batches * 16384 batch_size = ~20.13 Billion positions
# (~1.0 epoch of clean filtered positions)
superbatches: 300

# WDL lambda distillation factor (92% deep search score, 8% game outcome)
wdl: 0.08

# Peak learning rate for Cosine Annealing decay
lr: 0.0015

# Minimum learning rate at cycle completion
min_lr: 0.00005

# Training batch size in positions (safe for 6 GB GTX 1660 Ti VRAM)
batch_size: 16384

# Number of batches per superbatch (4096 * 16384 = ~67.11M positions/sb)
batches_per_superbatch: 4096

# Fast CPU validation sample: 100k positions per pack (600k total)
# Evaluates in ~2-3 seconds instead of stalling training for minutes
eval_positions_per_pack: 100000

# Checkpoint & SPRT interval: every 15 superbatches (~every 1.0 Billion positions)
# Evaluates 20 times total across the 300-sb run instead of interrupting every step!
save_interval: 15

# Positive Bias Initialization (Layer 0, Layer 1, Layer 2)
# L1=0.35, L2=0.55 provide robust safety cushion against SCReLU dying neurons
bias_l0: 0.05
bias_l1: 0.35
bias_l2: 0.55

# ==============================================================================
# SPRT (Sequential Probability Ratio Test) Automated Verification
# ==============================================================================
sprt:
  enabled: true
  cutechess_path: "C:/Users/evanh/works/software/cutechess/cutechess-cli.exe"
  engine_path: "C:/Users/evanh/works/software/cutechess/engines/KittyEngineV5_experimental/KittyEngineV5.exe"
  engine_args: "--mode bullet"
  book_path: "C:/Users/evanh/works/software/cutechess/book/book.bin"
  book_depth: 1
  # Fast bullet (2s + 0.02s) with 12 parallel games: finishes 40-60 games in ~45 seconds!
  time_control: "2+0.02"
  concurrency: 12
  max_rounds: 60
  elo0: 0.0
  elo1: 15.0
  alpha: 0.05
  beta: 0.15
  draw_movenumber: 40
  draw_movecount: 8
  draw_score: 10
  resign_movecount: 6
  resign_score: 500
"#;

impl TrainerConfig {
    pub fn load_or_default() -> Self {
        let yaml_path = Path::new("config.yaml");
        let json_path = Path::new("config.json");

        if yaml_path.exists() {
            match fs::read_to_string(yaml_path) {
                Ok(content) => match serde_yaml::from_str::<TrainerConfig>(&content) {
                    Ok(cfg) => {
                        println!("[config] Loaded configuration from config.yaml");
                        return cfg;
                    }
                    Err(e) => {
                        eprintln!("[config] Warning: Failed to parse config.yaml: {}. Using defaults.", e);
                    }
                },
                Err(e) => {
                    eprintln!("[config] Warning: Failed to read config.yaml: {}. Using defaults.", e);
                }
            }
        } else if json_path.exists() {
            match fs::read_to_string(json_path) {
                Ok(content) => match serde_json::from_str::<TrainerConfig>(&content) {
                    Ok(cfg) => {
                        println!("[config] Loaded configuration from config.json");
                        return cfg;
                    }
                    Err(e) => {
                        eprintln!("[config] Warning: Failed to parse config.json: {}. Using defaults.", e);
                    }
                },
                Err(e) => {
                    eprintln!("[config] Warning: Failed to read config.json: {}. Using defaults.", e);
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
            if path.is_dir()
                && let Some(name) = path.file_name().and_then(|s| s.to_str())
                    && let Some(step_str) = name.strip_prefix("kitty_bullet_screlu-")
                        && let Ok(step) = step_str.parse::<usize>()
                            && step >= highest_step && path.join("raw.bin").exists() {
                                highest_step = step;
                                highest_path = Some(path);
                            }
        }
    }

    if highest_step > 0
        && let Some(path) = highest_path {
            return Some((highest_step, path));
        }

    // Fallback: check baseline.json or training_history.json
    let history_path = cp_dir.join("training_history.json");
    if history_path.exists()
        && let Ok(content) = fs::read_to_string(&history_path)
            && let Ok(records) = serde_json::from_str::<Vec<serde_json::Value>>(&content)
                && let Some(last) = records.last()
                    && let Some(step) = last.get("step").and_then(|v| v.as_u64()).map(|v| v as usize) {
                        let folder = cp_dir.join(format!("kitty_bullet_screlu-{}", step));
                        return Some((step, folder));
                    }

    None
}
