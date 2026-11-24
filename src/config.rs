use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const DATA_DIR: &str = "data";
pub const ENGINE_DIR: &str = "engine";
pub const ENGINE_WEIGHTS_FILE: &str = "engine/weights.bin";
pub const ENGINE_SEARCH_CONFIG_FILE: &str = "engine/search_config.yaml";
pub const CHECKPOINTS_DIR: &str = "checkpoints";
pub const CHAMPION_FILE: &str = "checkpoints/champion_weights.bin";
pub const SPRT_MATCH_LOG: &str = "checkpoints/sprt_matches.log";
pub const SPRT_GAMES_PGN: &str = "checkpoints/sprt_games.pgn";

pub const INPUT_SIZE: usize = 12_288;
pub const NUM_KING_BUCKETS: usize = 16;
pub const ACCUM_SIZE: usize = 768;
pub const PAIRWISE_SIZE: usize = 768;
pub const HIDDEN1_SIZE: usize = 16;
pub const HIDDEN2_SIZE: usize = 32;
pub const NUM_OUTPUT_BUCKETS: usize = 8;
pub const CHECKPOINT_SIZE: u64 = 38_164_072;

fn default_true() -> bool { true }
fn default_cutechess_path() -> String { "C:/Users/evanh/works/software/cutechess/cutechess-cli.exe".to_string() }
pub fn default_engine_name() -> String { "KittyEngineV5.exe".to_string() }
fn default_engine_args() -> String { String::new() }

/// Canonicalizes path if possible, stripping Windows `\\?\` UNC prefix.
pub fn to_clean_path(p: &Path) -> PathBuf {
    if let Ok(canon) = fs::canonicalize(p) {
        let s = canon.to_string_lossy().to_string();
        let stripped = s.strip_prefix(r"\\?\").unwrap_or(&s);
        PathBuf::from(stripped)
    } else if p.is_relative() {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(p)
    } else {
        p.to_path_buf()
    }
}

/// Resolves an engine name (e.g. "KittyEngineV5.exe", "KittyEngineV5", or empty/default)
/// to `(engine_executable_path, engine_dir)` under the `engine/` directory.
pub fn resolve_engine_path(spec: &str) -> (PathBuf, PathBuf) {
    resolve_engine_path_in_dir(spec, Path::new(ENGINE_DIR))
}

fn resolve_engine_path_in_dir(spec: &str, engine_dir: &Path) -> (PathBuf, PathBuf) {
    let clean_dir = to_clean_path(engine_dir);
    let trimmed = spec.trim();

    // Helper to scan directory for the first .exe (sorted for determinism)
    let find_first_exe = |dir: &Path| -> Option<PathBuf> {
        if dir.is_dir() {
            if let Ok(entries) = fs::read_dir(dir) {
                let mut exes: Vec<PathBuf> = entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.is_file()
                            && p.extension()
                                .and_then(|e| e.to_str())
                                .map(|e| e.eq_ignore_ascii_case("exe"))
                                .unwrap_or(false)
                    })
                    .collect();
                exes.sort();
                return exes.into_iter().next();
            }
        }
        None
    };

    // 1. If empty or "default", auto-detect in engine directory
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("default") {
        if let Some(exe) = find_first_exe(&clean_dir) {
            return (to_clean_path(&exe), clean_dir);
        }
        return (clean_dir.join(default_engine_name()), clean_dir);
    }

    // 2. Always resolve the engine file name under the `engine/` directory
    //    (even if a legacy full/relative path was supplied)
    let file_name = Path::new(trimmed)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(trimmed);

    let candidate1 = clean_dir.join(file_name);
    if candidate1.is_file() {
        return (to_clean_path(&candidate1), clean_dir);
    }

    if !file_name.to_ascii_lowercase().ends_with(".exe") {
        let with_exe = format!("{}.exe", file_name);
        let candidate2 = clean_dir.join(&with_exe);
        if candidate2.is_file() {
            return (to_clean_path(&candidate2), clean_dir);
        }
        return (clean_dir.join(with_exe), clean_dir);
    }

    (candidate1, clean_dir)
}

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
    #[serde(default = "default_engine_name", alias = "engine_path", alias = "engine")]
    pub engine_name: String,
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

impl SprtConfig {
    pub fn resolve_engine(&self) -> (PathBuf, PathBuf) {
        resolve_engine_path(&self.engine_name)
    }
}

impl Default for SprtConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            cutechess_path: default_cutechess_path(),
            engine_name: default_engine_name(),
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
pub struct EngineConfig {
    #[serde(default = "default_engine_name", alias = "engine_name", alias = "engine_path", alias = "engine")]
    pub name: String,
    #[serde(
        default = "default_engine_args",
        alias = "engine_args",
        alias = "engine_cli_flags",
        deserialize_with = "deserialize_engine_args"
    )]
    pub args: String,
    #[serde(default = "default_cutechess_path")]
    pub cutechess_path: String,
    #[serde(default = "default_book_path")]
    pub book_path: String,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            name: default_engine_name(),
            args: "--mode bullet".to_string(),
            cutechess_path: default_cutechess_path(),
            book_path: default_book_path(),
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
    #[serde(default)]
    pub engine: EngineConfig,
    #[serde(default)]
    pub sprt: SprtConfig,
    #[serde(default)]
    pub spsa: crate::spsa::SpsaSettings,
}

impl TrainerConfig {
    /// Propagates the unified top-level `engine` settings into both `sprt` and `spsa`
    /// so users only configure engine name/flags/cutechess/book once.
    pub fn apply_unified_engine(&mut self) {
        self.sprt.engine_name = self.engine.name.clone();
        self.sprt.engine_args = self.engine.args.clone();
        self.sprt.cutechess_path = self.engine.cutechess_path.clone();
        self.sprt.book_path = self.engine.book_path.clone();

        self.spsa.engine_name = self.engine.name.clone();
        self.spsa.engine_cli_flags = self.engine.args.clone();
        self.spsa.cutechess_path = self.engine.cutechess_path.clone();
        self.spsa.book_path = self.engine.book_path.clone();
    }
}

impl Default for TrainerConfig {
    fn default() -> Self {
        let mut cfg = Self {
            superbatches: 360,
            wdl: 0.08,
            lr: 0.0003,
            min_lr: 0.000005,
            batch_size: 16_384,
            batches_per_superbatch: Some(4096),
            eval_positions_per_pack: 100_000,
            save_interval: 10,
            engine: EngineConfig::default(),
            sprt: SprtConfig::default(),
            spsa: crate::spsa::SpsaSettings::default(),
        };
        cfg.apply_unified_engine();
        cfg
    }
}

const DEFAULT_CONFIG_YAML: &str = r#"# ==============================================================================
# KittyEngineTrainer Configuration (Optimized for High-Throughput Training)
# ==============================================================================

# Target 360 superbatches * 4096 batches * 16384 batch_size = ~24.16 Billion positions
# (~1.0 full epoch over 24B unique filtered positions)
superbatches: 360

# WDL lambda distillation factor (92% deep search score, 8% game outcome)
wdl: 0.08

# Peak learning rate for Cosine Annealing decay (tuned for 360 * 4096 = 1.47M AdamW steps)
lr: 0.0003

# Minimum learning rate at cycle completion
min_lr: 0.000005

# Training batch size in positions (safe for 6 GB GTX 1660 Ti VRAM)
batch_size: 16384

# Number of batches per superbatch (4096 * 16384 = ~67.11M positions/sb)
batches_per_superbatch: 4096

# Fast CPU validation sample: 100k positions per pack (600k total)
# Evaluates in ~2-3 seconds instead of stalling training for minutes
eval_positions_per_pack: 100000

# Checkpoint & SPRT interval: every 10 superbatches (~every 671M positions)
save_interval: 10

# ==============================================================================
# Unified Engine & Match Runner Settings (Shared by SPRT & SPSA)
# ==============================================================================
engine:
  name: "KittyEngineV5.exe" # Resolved automatically inside ./engine/
  args: "--mode bullet"
  cutechess_path: "C:/Users/evanh/works/software/cutechess/cutechess-cli.exe"
  book_path: "C:/Users/evanh/works/software/cutechess/book/book.bin"

# ==============================================================================
# SPRT (Sequential Probability Ratio Test) Automated Verification
# ==============================================================================
sprt:
  enabled: true
  book_depth: 1
  time_control: "8+0.1"
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

# ==============================================================================
# SPSA (Simultaneous Perturbation Stochastic Approximation) Search Tuner
# ==============================================================================
spsa:
  time_control: "10+1"
  concurrency: 12
  games_per_iteration: 24
  book_depth: 1
  draw_movenumber: 40
  draw_movecount: 20
  draw_score: 10
  resign_movecount: 10
  resign_score: 500
  max_moves: 200
  sprt_elo0: 0.0
  sprt_elo1: 20.0
  sprt_alpha: 0.15
  sprt_beta: 0.15
  sprt_max_rounds: 50
  verify_interval_iterations: 25
  verify_win_rate_threshold: 0.58
  verify_elo_threshold: 20.0
"#;

impl TrainerConfig {
    pub fn load_or_default() -> Self {
        let yaml_path = Path::new("config.yaml");
        let json_path = Path::new("config.json");

        if yaml_path.exists() {
            match fs::read_to_string(yaml_path) {
                Ok(content) => match serde_yaml::from_str::<TrainerConfig>(&content) {
                    Ok(mut cfg) => {
                        cfg.apply_unified_engine();
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
                    Ok(mut cfg) => {
                        cfg.apply_unified_engine();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spsa::SpsaSettings;

    #[test]
    fn test_resolve_engine_path_always_uses_engine_dir() {
        let tmp = std::env::temp_dir().join("kitty_trainer_engine_resolve_test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let exe_file = tmp.join("KittyEngineV5.exe");
        fs::write(&exe_file, b"").unwrap();
        let expected_exe = to_clean_path(&exe_file);
        let expected_dir = to_clean_path(&tmp);

        // Bare name with .exe
        assert_eq!(
            resolve_engine_path_in_dir("KittyEngineV5.exe", &tmp),
            (expected_exe.clone(), expected_dir.clone())
        );

        // Bare name without .exe
        assert_eq!(
            resolve_engine_path_in_dir("KittyEngineV5", &tmp),
            (expected_exe.clone(), expected_dir.clone())
        );

        // Legacy relative / absolute paths still resolve under engine_dir
        assert_eq!(
            resolve_engine_path_in_dir(r"engine\KittyEngineV5.exe", &tmp),
            (expected_exe.clone(), expected_dir.clone())
        );
        assert_eq!(
            resolve_engine_path_in_dir(
                "C:/Users/evanh/works/software/cutechess/engines/other/KittyEngineV5.exe",
                &tmp
            ),
            (expected_exe.clone(), expected_dir.clone())
        );

        // Empty or "default" auto-detects in engine_dir
        assert_eq!(
            resolve_engine_path_in_dir("", &tmp),
            (expected_exe.clone(), expected_dir.clone())
        );
        assert_eq!(
            resolve_engine_path_in_dir("default", &tmp),
            (expected_exe.clone(), expected_dir.clone())
        );

        // Missing explicit name does not silently fall back to another binary
        let (missing_path, dir) = resolve_engine_path_in_dir("KittyEngineV6", &tmp);
        assert_eq!(dir, expected_dir);
        assert_eq!(missing_path, expected_dir.join("KittyEngineV6.exe"));
        assert!(!missing_path.exists());

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_unified_engine_name_deserialization() {
        // YAML SprtConfig with engine_name
        let sprt_new: SprtConfig = serde_yaml::from_str("engine_name: KittyEngineV6").unwrap();
        assert_eq!(sprt_new.engine_name, "KittyEngineV6");

        // YAML SprtConfig with legacy engine_path + engine_dir
        let sprt_legacy: SprtConfig = serde_yaml::from_str(
            "engine_path: C:/old/KittyEngineV5.exe\nengine_dir: old_dir\n",
        )
        .unwrap();
        assert_eq!(sprt_legacy.engine_name, "C:/old/KittyEngineV5.exe");

        // JSON SpsaSettings with engine_name
        let spsa_new: SpsaSettings =
            serde_json::from_str(r#"{"engine_name": "KittyEngineV6.exe"}"#).unwrap();
        assert_eq!(spsa_new.engine_name, "KittyEngineV6.exe");

        // JSON SpsaSettings with legacy engine_path + engine_dir
        let spsa_legacy: SpsaSettings = serde_json::from_str(
            r#"{"engine_path": "engine\\KittyEngineV5.exe", "engine_dir": "engine"}"#,
        )
        .unwrap();
        assert_eq!(spsa_legacy.engine_name, r"engine\KittyEngineV5.exe");

        // Unified TrainerConfig propagates top-level `engine:` into both `sprt` and `spsa`
        let mut unified: TrainerConfig = serde_yaml::from_str(DEFAULT_CONFIG_YAML).unwrap();
        unified.apply_unified_engine();
        assert_eq!(unified.sprt.engine_name, "KittyEngineV5.exe");
        assert_eq!(unified.spsa.engine_name, "KittyEngineV5.exe");
        assert_eq!(unified.sprt.engine_args, "--mode bullet");
        assert_eq!(unified.spsa.engine_cli_flags, "--mode bullet");
        assert_eq!(unified.sprt.cutechess_path, unified.spsa.cutechess_path);
        assert_eq!(unified.sprt.book_path, unified.spsa.book_path);
        assert_eq!(unified.spsa.games_per_iteration, 24);
    }
}
