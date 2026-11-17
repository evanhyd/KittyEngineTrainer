use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const DATA_DIR: &str = "data";
pub const CHECKPOINTS_DIR: &str = "checkpoints";
pub const CHAMPION_FILE: &str = "checkpoints/champion_weights.bin";
pub const SCRATCH_FILE: &str = "checkpoints/scratch_weights_init.bin";
pub const SPRT_MATCH_LOG: &str = "checkpoints/sprt_matches.log";
pub const SPRT_GAMES_PGN: &str = "checkpoints/sprt_games.pgn";

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
            sprt: SprtConfig::default(),
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

# ==============================================================================
# SPRT (Sequential Probability Ratio Test) Automated Verification
# ==============================================================================
# SPRT is the EXCLUSIVE gate for champion promotion.
# When a checkpoint is reached, an SPRT match is conducted against the active champion.
# Candidate is promoted ONLY if H1 is accepted (statistically verified stronger).
sprt:
  enabled: true
  cutechess_path: "C:/Users/evanh/works/software/cutechess/cutechess-cli.exe"
  engine_path: "C:/Users/evanh/Desktop/KittyEngineV5/x64/Release/KittyEngineV5.exe"
  engine_args: ""
  book_path: "C:/Users/evanh/works/software/cutechess/book/book.bin"
  book_depth: 8
  time_control: "5+0.1"
  concurrency: 2
  max_rounds: 200
  elo0: 0.0
  elo1: 10.0
  alpha: 0.05
  beta: 0.15
  draw_movenumber: 40
  draw_movecount: 8
  draw_score: 10
  resign_movecount: 4
  resign_score: 550
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
