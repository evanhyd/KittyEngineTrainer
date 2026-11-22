mod inputs;
mod export;
mod convert;
mod eval;
mod diagnostics;
mod config;
mod sprt;
mod signal;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use bullet_lib::{
    nn::optimiser::{AdamW, AdamWParams},
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader, ValueTrainerBuilder},
};

use inputs::{KittyHalfKAHM, MajorMinorCount};
use config::{
    TrainerConfig, CHAMPION_FILE, DATA_DIR, SCRATCH_FILE,
    INPUT_SIZE, ACCUM_SIZE, PAIRWISE_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, NUM_OUTPUT_BUCKETS,
};

const SCALE: f32 = 400.0;

fn print_usage() {
    println!("KittyEngineTrainer - High-Performance NNUE Trainer via Bullet");
    println!("============================================================");
    println!("Architecture: (24576 -> 256) x 2 -> 256 (pairwise) -> 64 -> 32 -> 1 x 8 (KittyEngine Layout)");
    println!();
    println!("CONFIGURATION:");
    println!("  Training & SPRT parameters are loaded from 'config.yaml'.");
    println!("  Data is loaded from 'data/' and checkpoints are saved to 'checkpoints/'.");
    println!();
    println!("CORE COMMANDS:");
    println!("  train [OPTIONS]             Train NNUE (auto-resumes from latest checkpoint; pass --new to start fresh)");
    println!("  stop                        Gracefully signal active trainer to stop without data corruption");
    println!("  status                      Display active champion metrics, SPRT settings & history");
    println!("  sprt [cand] [champ]         Run automated Cutechess SPRT match between two weight files");
    println!("  check [weights.bin]         Run comprehensive health checks (Dead Neurons + Inversions)");
    println!("  eval [weights.bin]          Evaluate model metrics across all packs in 'data/'");
    println!("  export [step] [out.bin]     Export checkpoint to KittyEngine binary format (default: weights.bin)");
    println!("  compare [checkpoint_dir]    Validate GPU vs CPU forward-pass parity");
    println!("  scratch [output.bin]        Generate positive-bias scratch weights for initialization");
    println!("  info                        Display neural network architecture and layout specs");
    println!();
    println!("DIAGNOSTIC & UTILITY COMMANDS:");
    println!("  dead-neurons [weights.bin]  Detailed per-layer empirical dead neuron telemetry");
    println!("  inversions [weights.bin]    Detect high-centipawn polar sign disagreements");
    println!("  import <in.bin> <out.bin>   Convert KittyEngine binary weights into Bullet format");
    println!("  help, --help, -h            Display this help and usage menu");
    println!();
}

fn run_info() {
    println!("KittyEngine NNUE Layout Specification:");
    println!("  Input Feature Space:   {} (HalfKAv2 HM, 32 folded king squares x 768 piece planes)", INPUT_SIZE);
    println!("  Dual Accumulator:      {} per perspective", ACCUM_SIZE);
    println!("  Pairwise Interaction:  (128 + 128) -> {} (Hadamard product within STM/NTM)", PAIRWISE_SIZE);
    println!("  Hidden Layer 1:        {} -> {} (Dense + Activation)", PAIRWISE_SIZE, HIDDEN1_SIZE);
    println!("  Hidden Layer 2:        64 -> {} (Dense + Activation)", HIDDEN2_SIZE);
    println!("  Output Layer:          32 -> 1 x {} Buckets (Major/Minor Piece Count)", NUM_OUTPUT_BUCKETS);
    println!("  Centipawn Scale:       {:.1} cp", SCALE);
    println!("  Target Weight Size:    25,242,088 bytes (NNUESGD6 format)");
}

fn run_stop() {
    println!("================================================================================");
    println!("🛑 SENDING SAFE MANUAL STOP SIGNAL");
    println!("================================================================================");
    match signal::create_stop_file() {
        Ok(()) => {
            println!("  Trigger file created: {}", signal::STOP_TRIGGER_FILE);
            println!("  The active training process will finish its current superbatch, cleanly");
            println!("  export and verify weights to 'checkpoints/weights_step_<N>.bin' and");
            println!("  'weights.bin', and exit safely without any data corruption.");
            println!("================================================================================");
        }
        Err(e) => {
            eprintln!("Failed to write stop trigger file: {}", e);
        }
    }
}

fn run_export(args: &[String]) {
    let target_step_or_path = args.get(2).map(|s| s.as_str());
    let custom_out = args.get(3).map(PathBuf::from);

    let (src_path, out_path) = if let Some(target) = target_step_or_path {
        if let Ok(step) = target.parse::<usize>() {
            let folder = PathBuf::from(format!("checkpoints/kitty_bullet_screlu-{}", step));
            let raw = folder.join("raw.bin");
            let out = custom_out.unwrap_or_else(|| PathBuf::from(format!("weights_step_{}.bin", step)));
            (raw, out)
        } else {
            let path = PathBuf::from(target);
            let out = custom_out.unwrap_or_else(|| PathBuf::from("weights.bin"));
            (path, out)
        }
    } else {
        let champ = PathBuf::from(CHAMPION_FILE);
        if champ.exists() {
            let out = custom_out.unwrap_or_else(|| PathBuf::from("weights.bin"));
            (champ, out)
        } else if let Some((step, folder)) = config::detect_latest_checkpoint() {
            let raw = folder.join("raw.bin");
            let out = custom_out.unwrap_or_else(|| PathBuf::from(format!("weights_step_{}.bin", step)));
            (raw, out)
        } else {
            eprintln!("Error: No champion weights or checkpoint found in 'checkpoints/' to export.");
            return;
        }
    };

    if !src_path.exists() {
        eprintln!("Error: Source file '{}' does not exist.", src_path.display());
        return;
    }

    println!("Exporting weights from '{}' to '{}'...", src_path.display(), out_path.display());

    if src_path.file_name().and_then(|s| s.to_str()) != Some("raw.bin") && src_path.extension().and_then(|s| s.to_str()) == Some("bin") {
        match fs::copy(&src_path, &out_path) {
            Ok(_) => println!("Export completed successfully: {}", out_path.display()),
            Err(e) => eprintln!("Failed to copy weights: {}", e),
        }
    } else {
        match export::export_from_raw_checkpoint(&src_path, &out_path) {
            Ok(()) => println!("Export completed successfully: {}", out_path.display()),
            Err(e) => eprintln!("Failed to export weights: {}", e),
        }
    }
}

fn run_sprt(args: &[String]) {
    let cfg = TrainerConfig::load_or_default();
    let default_champ = PathBuf::from(CHAMPION_FILE);
    let default_cand = if let Some((step, _)) = config::detect_latest_checkpoint() {
        let p = PathBuf::from(format!("checkpoints/weights_step_{}.bin", step));
        if p.exists() { p } else { default_champ.clone() }
    } else {
        default_champ.clone()
    };

    let cand_path = args.get(2).map(PathBuf::from).unwrap_or(default_cand);
    let champ_path = args.get(3).map(PathBuf::from).unwrap_or(default_champ);

    if !cand_path.exists() {
        eprintln!("Error: Candidate weights file '{}' not found.", cand_path.display());
        return;
    }
    if !champ_path.exists() {
        eprintln!("Error: Champion weights file '{}' not found.", champ_path.display());
        return;
    }

    match sprt::run_sprt_match(&cfg.sprt, &cand_path, &champ_path) {
        Ok(res) => {
            let (passed, reason) = res.should_promote(cfg.sprt.elo1);
            if passed {
                println!("Candidate verified superior to Champion under SPRT ({}).", reason);
            } else {
                println!("Candidate did not meet criteria to defeat Champion under SPRT.");
            }
        }
        Err(e) => {
            eprintln!("SPRT execution failed: {}", e);
        }
    }
}

fn run_status() {
    let cfg = TrainerConfig::load_or_default();
    let baseline_path = Path::new("checkpoints/baseline.json");
    let history_path = Path::new("checkpoints/training_history.json");

    println!("================================================================================");
    println!("KittyEngineTrainer - Active Training & Champion Status");
    println!("================================================================================");
    println!("⚙️ ACTIVE CONFIGURATION (config.yaml):");
    println!("  - Target Superbatches: {}", cfg.superbatches);
    println!("  - Distillation WDL:    {:.2}", cfg.wdl);
    println!("  - Learning Rate:       {:.6} (peak) -> {:.6} (min)", cfg.lr, cfg.min_lr);
    println!("  - Batch Size:          {} positions", cfg.batch_size);
    println!("  - Checkpoint Save:     Every {} superbatch(es) (permanent playtest export)", cfg.save_interval);
    println!("  - Positive Biases:     L0={:.2}, L1={:.2}, L2={:.2}", cfg.bias_l0, cfg.bias_l1, cfg.bias_l2);
    println!();
    println!("⚔️ SPRT VERIFICATION SETTINGS:");
    println!("  - SPRT Enabled:        {}", cfg.sprt.enabled);
    println!("  - Cutechess CLI:       {}", cfg.sprt.cutechess_path);
    println!("  - Engine Executable:   {}", cfg.sprt.engine_path);
    if !cfg.sprt.engine_args.is_empty() {
        println!("  - Engine Arguments:    {}", cfg.sprt.engine_args);
    }
    println!("  - Time Control:        {}", cfg.sprt.time_control);
    println!("  - Concurrency:         {} threads", cfg.sprt.concurrency);
    println!("  - Max Rounds:          {}", cfg.sprt.max_rounds);
    println!("  - SPRT Bounds:         Elo [{}, {}] (alpha={}, beta={})",
        cfg.sprt.elo0, cfg.sprt.elo1, cfg.sprt.alpha, cfg.sprt.beta);
    println!();

    if baseline_path.exists() {
        if let Ok(content) = fs::read_to_string(baseline_path)
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                let step = v.get("step").and_then(|x| x.as_u64()).unwrap_or(0);
                let epoch = v.get("epoch").and_then(|x| x.as_u64()).unwrap_or(0);
                let mae = v.get("mae").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let smse = v.get("smse").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let bce = v.get("bce").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let mse = v.get("mse").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let mate_mae = v.get("mate_mae").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let non_mate_mae = v.get("non_mate_mae").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let wdl = v.get("wdl").and_then(|x| x.as_f64()).unwrap_or(0.20);
                let pos = v.get("positions").and_then(|x| x.as_u64()).unwrap_or(0);

                println!("👑 ACTIVE CHAMPION (checkpoints/champion_weights.bin):");
                println!("  - Superbatch Step:  {}", step);
                println!("  - Completed Epoch:  {}", epoch);
                println!("  - Distillation WDL: {:.2}", wdl);
                println!("  - Overall MAE:      {:.7}", mae);
                println!("  - Overall SMSE:     {:.7}", smse);
                println!("  - Log-Loss (BCE):   {:.7}", bce);
                println!("  - Overall MSE:      {:.7}", mse);
                println!("  - Mate MAE:         {:.7}", mate_mae);
                println!("  - Non-Mate MAE:     {:.7}", non_mate_mae);
                println!("  - Validation Pos:   {} (across dataset packs)", pos);

                if let Some(sprt_obj) = v.get("sprt") {
                    let w = sprt_obj.get("wins").and_then(|x| x.as_u64()).unwrap_or(0);
                    let l = sprt_obj.get("losses").and_then(|x| x.as_u64()).unwrap_or(0);
                    let d = sprt_obj.get("draws").and_then(|x| x.as_u64()).unwrap_or(0);
                    let elo = sprt_obj.get("elo_diff").and_then(|x| x.as_f64());
                    let err = sprt_obj.get("elo_error").and_then(|x| x.as_f64());
                    println!("  - SPRT Verification: Defeated Prior Champ (+{} -{} ={})", w, l, d);
                    if let Some(e) = elo {
                        println!("  - SPRT Elo Advantage: {:+.1} +/- {:.1}", e, err.unwrap_or(0.0));
                    }
                }
            }
    } else {
        println!("👑 ACTIVE CHAMPION: None recorded in checkpoints/baseline.json");
    }

    if let Some((step, folder)) = config::detect_latest_checkpoint() {
        println!("\n📁 LATEST BULLET CHECKPOINT DETECTED:");
        println!("  - Step:             {}", step);
        println!("  - Folder:           {}", folder.display());
        println!("  - Ready to resume:  'cargo run --release -- train' (or pass '--new' to start fresh)");
    }

    if history_path.exists()
        && let Ok(content) = fs::read_to_string(history_path)
            && let Ok(records) = serde_json::from_str::<Vec<serde_json::Value>>(&content) {
                println!("\n📜 RECENT TRAINING HISTORY (last 10 steps):");
                println!("{:<8} | {:<5} | {:<12} | {:<12} | {:<12} | {:<10} | {:<10}",
                    "Step", "Epoch", "Positions", "Overall MAE", "Overall SMSE", "Mate MAE", "Status");
                println!("{:-<8}-+-{:-<5}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<10}-+-{:-<10}", "", "", "", "", "", "", "");

                let count = records.len();
                let start = count.saturating_sub(10);
                for r in &records[start..] {
                    let st = r.get("step").and_then(|x| x.as_u64()).unwrap_or(0);
                    let ep = r.get("epoch").and_then(|x| x.as_u64()).unwrap_or(0);
                    let pos = r.get("positions").and_then(|x| x.as_u64()).unwrap_or(0);
                    let mae = r.get("mae").and_then(|x| x.as_f64()).unwrap_or(0.0);
                    let smse = r.get("smse").and_then(|x| x.as_f64()).unwrap_or(0.0);
                    let m_mae = r.get("mate_mae").and_then(|x| x.as_f64()).unwrap_or(0.0);
                    let prom = r.get("promoted").and_then(|x| x.as_bool()).unwrap_or(false);

                    let pos_str = if pos >= 1_000_000_000 {
                        format!("{:.2}B", pos as f64 / 1_000_000_000.0)
                    } else {
                        format!("{:.1}M", pos as f64 / 1_000_000.0)
                    };
                    let status_str = if prom { "👑 Champ" } else { "Trained" };

                    println!("{:<8} | {:<5} | {:<12} | {:<12.7} | {:<12.7} | {:<10.7} | {:<10}",
                        format!("#{}", st), ep, pos_str, mae, smse, m_mae, status_str);
                }
            }

    println!("================================================================================\n");
}

fn run_check(args: &[String]) {
    let default_weights = PathBuf::from(CHAMPION_FILE);
    let weights_path = args.get(2).map(Path::new).unwrap_or(&default_weights);
    let default_data = PathBuf::from(DATA_DIR);
    let target_data = args.get(3).map(Path::new).unwrap_or(&default_data);
    let positions_per_pack: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(100_000);

    if !weights_path.exists() {
        eprintln!("Error: weights file '{}' not found.", weights_path.display());
        return;
    }

    println!("================================================================================");
    println!("🔍 RUNNING UNIFIED HEALTH CHECK ON {}", weights_path.display());
    println!("================================================================================");

    // 1. Dead Neuron Empirical Telemetry
    let dn_args = vec![
        "main".to_string(),
        "dead-neurons".to_string(),
        weights_path.to_string_lossy().to_string(),
        target_data.to_string_lossy().to_string(),
        positions_per_pack.to_string(),
    ];
    run_dead_neurons(&dn_args);

    // 2. Polar Inversion & Sign Disagreement Analysis
    let inv_args = vec![
        "main".to_string(),
        "inversions".to_string(),
        weights_path.to_string_lossy().to_string(),
        target_data.to_string_lossy().to_string(),
        positions_per_pack.to_string(),
    ];
    run_inversions(&inv_args);
}

fn run_import(args: &[String]) {
    if args.len() < 4 {
        println!("Error: missing arguments for import command.");
        println!("Usage: cargo run --release -- import <input_weights.bin> <output_bullet.bin>");
        return;
    }

    let input_path = Path::new(&args[2]);
    let output_path = Path::new(&args[3]);

    match export::import_from_kitty_weights(input_path, output_path) {
        Ok(()) => {
            println!("Import completed successfully.");
        }
        Err(e) => {
            eprintln!("Failed to import weights: {}", e);
        }
    }
}

fn archive_old_checkpoints() -> Option<PathBuf> {
    let cp_dir = Path::new("checkpoints");
    if !cp_dir.exists() {
        return None;
    }

    let timestamp = sprt::current_timestamp().replace([':', ' ', '-'], "_");
    let archive_dir = cp_dir.join(format!("archive_{}", timestamp));

    let mut moved_any = false;
    if let Ok(entries) = fs::read_dir(cp_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
            if name.starts_with("archive_") {
                continue;
            }
            if name.starts_with("kitty_bullet_screlu-")
                || name == "champion_weights.bin"
                || name == "baseline.json"
                || name == "training_history.json"
                || name == "sprt_matches.log"
                || name == "sprt_games.pgn"
                || name.starts_with("weights_step_")
                || name.starts_with("weights_screlu_step")
                || name == "scratch_weights_init.bin"
                || name == "init_bullet_weights.bin"
            {
                if !moved_any {
                    let _ = fs::create_dir_all(&archive_dir);
                }
                let target = archive_dir.join(&name);
                let _ = fs::rename(&path, &target);
                moved_any = true;
            }
        }
    }

    if moved_any {
        Some(archive_dir)
    } else {
        None
    }
}

pub fn prune_outdated_checkpoints(keep_step: usize) {
    let cp_dir = Path::new("checkpoints");
    if !cp_dir.exists() {
        return;
    }

    let mut deleted_dirs = 0;
    let mut deleted_files = 0;
    let mut freed_bytes: u64 = 0;

    if let Ok(entries) = fs::read_dir(cp_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();

            // Check for checkpoint directories: kitty_bullet_screlu-<step>
            if path.is_dir() {
                if let Some(step_str) = name.strip_prefix("kitty_bullet_screlu-") {
                    if let Ok(step) = step_str.parse::<usize>() {
                        if step < keep_step {
                            let dir_size: u64 = fs::read_dir(&path)
                                .map(|dir| dir.flatten().map(|e| e.metadata().map(|m| m.len()).unwrap_or(0)).sum())
                                .unwrap_or(0);
                            if fs::remove_dir_all(&path).is_ok() {
                                deleted_dirs += 1;
                                freed_bytes += dir_size;
                            }
                        }
                    }
                }
            } else if path.is_file() {
                // Check for weights_step_<step>.bin or weights_screlu_step<step>.bin
                let step_opt = if let Some(s) = name.strip_prefix("weights_step_").and_then(|s| s.strip_suffix(".bin")) {
                    s.parse::<usize>().ok()
                } else if let Some(s) = name.strip_prefix("weights_screlu_step").and_then(|s| s.strip_suffix(".bin")) {
                    s.parse::<usize>().ok()
                } else {
                    None
                };

                if let Some(step) = step_opt {
                    if step < keep_step {
                        let file_size = path.metadata().map(|m| m.len()).unwrap_or(0);
                        if fs::remove_file(&path).is_ok() {
                            deleted_files += 1;
                            freed_bytes += file_size;
                        }
                    }
                }
            }
        }
    }

    if deleted_dirs > 0 || deleted_files > 0 {
        let freed_mb = freed_bytes as f64 / (1024.0 * 1024.0);
        println!(
            "🧹 [Cleanup] Pruned {} outdated checkpoint folders and {} weight files older than champion step {} (freed {:.1} MB).",
            deleted_dirs, deleted_files, keep_step, freed_mb
        );
    }
}

fn run_train(args: &[String]) {
    let cfg = TrainerConfig::load_or_default();

    let mut target_dir_or_file = DATA_DIR.to_string();
    let mut weights_path: Option<String> = None;
    let mut superbatches: usize = cfg.superbatches;
    let mut start_superbatch: usize = 1;
    let mut _schedule_start: Option<usize> = Some(1);
    let mut start_fresh: bool = false;
    let mut is_external_weights: bool = false;
    let mut batch_size: usize = cfg.batch_size;
    let mut user_batches_per_sb: Option<usize> = cfg.batches_per_superbatch;
    let mut user_lr: Option<f32> = Some(cfg.lr);
    let mut user_min_lr: Option<f32> = Some(cfg.min_lr);
    let mut wdl_val: f32 = cfg.wdl;
    let mut eval_positions_per_pack: usize = cfg.eval_positions_per_pack;
    let save_interval: usize = cfg.save_interval.max(1);

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--new" => {
                start_fresh = true;
            }
            "--weights" => {
                i += 1;
                if i < args.len() {
                    weights_path = Some(args[i].clone());
                    is_external_weights = true;
                }
            }
            "--resume" => {
                i += 1;
                if i < args.len() && !args[i].starts_with("--") {
                    weights_path = Some(args[i].clone());
                }
            }
            "--start-superbatch" => {
                i += 1;
                if i < args.len()
                    && let Ok(v) = args[i].parse::<usize>() {
                        start_superbatch = v;
                    }
            }
            "--schedule-start" | "--cycle-start" => {
                i += 1;
                if i < args.len()
                    && let Ok(v) = args[i].parse::<usize>() {
                        _schedule_start = Some(v);
                    }
            }
            "--superbatches" => {
                i += 1;
                if i < args.len()
                    && let Ok(v) = args[i].parse::<usize>() {
                        superbatches = v;
                    }
            }
            "--batches-per-superbatch" => {
                i += 1;
                if i < args.len()
                    && let Ok(v) = args[i].parse::<usize>() {
                        user_batches_per_sb = Some(v);
                    }
            }
            "--batch-size" => {
                i += 1;
                if i < args.len()
                    && let Ok(v) = args[i].parse::<usize>() {
                        batch_size = v;
                    }
            }
            "--lr" => {
                i += 1;
                if i < args.len()
                    && let Ok(v) = args[i].parse::<f32>() {
                        user_lr = Some(v);
                    }
            }
            "--min-lr" => {
                i += 1;
                if i < args.len()
                    && let Ok(v) = args[i].parse::<f32>() {
                        user_min_lr = Some(v);
                    }
            }
            "--eval-per-pack" => {
                i += 1;
                if i < args.len()
                    && let Ok(v) = args[i].parse::<usize>() {
                        eval_positions_per_pack = v;
                    }
            }
            "--wdl" => {
                i += 1;
                if i < args.len()
                    && let Ok(v) = args[i].parse::<f32>() {
                        wdl_val = v;
                    }
            }
            other => {
                let p = Path::new(other);
                if p.is_dir() || p.extension().and_then(|s| s.to_str()) == Some("binpack") {
                    target_dir_or_file = other.to_string();
                } else if weights_path.is_none() && (other.ends_with(".bin") || p.is_dir()) {
                    weights_path = Some(other.to_string());
                    is_external_weights = true;
                } else if let Ok(val) = other.parse::<usize>() {
                    superbatches = val;
                }
            }
        }
        i += 1;
    }

    let run_state_desc = if weights_path.is_none() {
        if start_fresh {
            if let Some(archive_dir) = archive_old_checkpoints() {
                format!("Starting fresh from scratch (--new) [Previous checkpoints archived to {}]", archive_dir.display())
            } else {
                "Starting fresh from scratch (--new)".to_string()
            }
        } else if let Some((step, folder)) = config::detect_latest_checkpoint() {
            start_superbatch = step + 1;
            _schedule_start = Some(1);
            weights_path = Some(folder.to_string_lossy().to_string());
            if start_superbatch > superbatches {
                println!("Training is already complete up to target superbatches ({})!", superbatches);
                println!("To train further, increase 'superbatches' in config.yaml.");
                return;
            }
            format!("Auto-resuming from step {} ({})", step, folder.display())
        } else {
            "Starting from scratch (positive-bias init)".to_string()
        }
    } else {
        format!("Using weights from {}", weights_path.as_deref().unwrap_or("unknown"))
    };

    if target_dir_or_file.is_empty() {
        eprintln!("Error: no data path specified.");
        return;
    }

    let target_path = Path::new(&target_dir_or_file);
    let mut data_files: Vec<PathBuf> = Vec::new();

    if target_path.is_dir() {
        if let Ok(entries) = fs::read_dir(target_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("binpack") {
                    data_files.push(path);
                }
            }
        }
        data_files.sort();
    } else if target_path.exists() {
        data_files.push(target_path.to_path_buf());
    } else {
        eprintln!("Error: target path '{}' does not exist.", target_path.display());
        return;
    }

    if data_files.is_empty() {
        eprintln!("Error: no .binpack files found in '{}'.", target_path.display());
        return;
    }

    let train_files = data_files.clone();

    let is_sfbinpack = train_files.first().map_or(false, |p| convert::is_sfbinpack_file(p));

    let total_positions: usize = if is_sfbinpack {
        let total_bytes: u64 = train_files.iter().map(|f| fs::metadata(f).map(|m| m.len()).unwrap_or(0)).sum();
        // Compressed Stockfish BINP (~2.62 bytes/entry); ~82% retained after check & mate filters
        ((total_bytes as f64 / 2.62) * 0.818) as usize
    } else {
        train_files
            .iter()
            .map(|f| fs::metadata(f).map(|m| (m.len() / 32) as usize).unwrap_or(0))
            .sum()
    };
    let total_dataset_batches = (total_positions / batch_size).max(1);
    let batches_per_superbatch = user_batches_per_sb.unwrap_or_else(|| {
        if is_sfbinpack {
            2048
        } else {
            (total_dataset_batches / 9).max(1)
        }
    });
    let positions_per_superbatch = batches_per_superbatch * batch_size;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(KittyHalfKAHM)
        .output_buckets(MajorMinorCount::<8>)
        .save_format(&[
            SavedFormat::id("l0w"),
            SavedFormat::id("l0b"),
            SavedFormat::id("l1w"),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w"),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w"),
            SavedFormat::id("l3b"),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", INPUT_SIZE, ACCUM_SIZE);
            let l1 = builder.new_affine("l1", PAIRWISE_SIZE, HIDDEN1_SIZE);
            let l2 = builder.new_affine("l2", HIDDEN1_SIZE, HIDDEN2_SIZE);
            let l3 = builder.new_affine("l3", HIDDEN2_SIZE, NUM_OUTPUT_BUCKETS);

            // Vectorized pairwise interaction (Hadamard product) between halves of each perspective
            let stm_act = l0.forward(stm_inputs).screlu();
            let stm_hidden = stm_act.slice_rows(0, ACCUM_SIZE / 2) * stm_act.slice_rows(ACCUM_SIZE / 2, ACCUM_SIZE);

            let ntm_act = l0.forward(ntm_inputs).screlu();
            let ntm_hidden = ntm_act.slice_rows(0, ACCUM_SIZE / 2) * ntm_act.slice_rows(ACCUM_SIZE / 2, ACCUM_SIZE);

            let pairwise = stm_hidden.concat(ntm_hidden);

            let h1 = l1.forward(pairwise).screlu();
            let h2 = l2.forward(h1).screlu();
            l3.forward(h2).select(output_buckets)
        });

    if let Some(ref w_path) = weights_path {
        let p = Path::new(w_path);
        if !p.exists() {
            eprintln!("Error: initial weights file '{}' not found.", p.display());
            return;
        }

        if p.is_dir() {
            trainer.load_from_checkpoint(p.to_str().unwrap());
        } else {
            let init_bullet_path = Path::new("checkpoints/init_bullet_weights.bin");
            if let Err(e) = export::import_from_kitty_weights(p, init_bullet_path) {
                eprintln!("Failed to import KittyEngine weights: {}", e);
                return;
            }

            if let Err(e) = trainer.optimiser.load_weights_from_file(init_bullet_path.to_str().unwrap()) {
                eprintln!("Failed to load weights into trainer: {:?}", e);
                return;
            }
        }
    } else {
        let scratch_kitty_path = Path::new(SCRATCH_FILE);
        let init_bullet_path = Path::new("checkpoints/init_bullet_weights.bin");

        if let Err(e) = export::generate_scratch_kitty_weights(scratch_kitty_path, cfg.bias_l0, cfg.bias_l1, cfg.bias_l2) {
            eprintln!("Failed to generate scratch weights: {}", e);
            return;
        }

        if let Err(e) = export::import_from_kitty_weights(scratch_kitty_path, init_bullet_path) {
            eprintln!("Failed to import scratch weights into Bullet format: {}", e);
            return;
        }

        if let Err(e) = trainer.optimiser.load_weights_from_file(init_bullet_path.to_str().unwrap()) {
            eprintln!("Failed to load scratch weights into trainer: {:?}", e);
            return;
        }
    }

    trainer.optimiser.set_params(AdamWParams {
        decay: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        min_weight: -1.98,
        max_weight: 1.98,
    });

    let max_lr = user_lr.unwrap_or(if is_external_weights { 0.000015 } else { cfg.lr });
    let min_lr = user_min_lr.unwrap_or(cfg.min_lr);
    let net_id = "kitty_bullet_screlu".to_string();
    let settings = LocalSettings {
        threads: 8,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 64,
    };

    let baseline_path = Path::new("checkpoints/baseline.json");
    let mut baseline_mae = f64::MAX;
    let mut baseline_smse = f64::MAX;

    if !start_fresh
        && baseline_path.exists()
        && let Ok(content) = fs::read_to_string(baseline_path)
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                baseline_mae = v.get("mae").and_then(|x| x.as_f64()).unwrap_or(f64::MAX);
                baseline_smse = v.get("smse").and_then(|x| x.as_f64()).unwrap_or(f64::MAX);
            }

    let pack_strs: Vec<String> = train_files.iter().map(|p| p.to_string_lossy().to_string()).collect();
    let pack_refs: Vec<&str> = pack_strs.iter().map(|s| s.as_str()).collect();

    let direct_loader = if !is_sfbinpack {
        Some(loader::DirectSequentialDataLoader::new(&pack_refs))
    } else {
        None
    };

    let sf_loader = if is_sfbinpack {
        Some(loader::sfbinpack::SfBinpackLoader::new_concat_multiple(
            &pack_refs,
            512,
            8,
            |entry| !entry.pos.is_checked(entry.pos.side_to_move()) && entry.score.abs() <= 10000,
        ))
    } else {
        None
    };

    let total_pos_str = if is_sfbinpack {
        let total_bytes: u64 = train_files.iter().map(|f| fs::metadata(f).map(|m| m.len()).unwrap_or(0)).sum();
        format!("~{:.1}B (Stockfish BINP)", total_bytes as f64 / 2.62 / 1_000_000_000.0)
    } else {
        format!("{:.1}M (ChessBoard)", total_positions as f64 / 1_000_000.0)
    };

    let total_eval_pos = eval_positions_per_pack * train_files.len();

    println!("================================================================================");
    println!("KittyEngineTrainer | Architecture: HalfKA (24576->256)x2 -> 256 -> 64 -> 32 -> 1x8");
    println!("  Dataset:    {} files ({})", train_files.len(), total_pos_str);
    println!("  Schedule:   Steps {}..{} | {} batches/sb ({:.1}M pos/sb) | Batch: {}",
        start_superbatch, superbatches, batches_per_superbatch, positions_per_superbatch as f64 / 1_000_000.0, batch_size);
    println!("  Learning:   LR: {:.6} -> {:.6} (Cosine) | WDL: {:.2} | Save: Every {} sb ({:.1}M pos)",
        max_lr, min_lr, wdl_val, save_interval, (save_interval * positions_per_superbatch) as f64 / 1_000_000.0);
    println!("  Validation: {} pos/pack ({:.2}M total) | Biases: L0={:.2}, L1={:.2}, L2={:.2}",
        eval_positions_per_pack, total_eval_pos as f64 / 1_000_000.0,
        cfg.bias_l0, cfg.bias_l1, cfg.bias_l2);
    if cfg.sprt.enabled {
        println!("  SPRT Gate:  Enabled | TC: {} | Concurrency: {} | Max Rounds: {} | Elo [{:.1}, {:.1}]",
            cfg.sprt.time_control, cfg.sprt.concurrency, cfg.sprt.max_rounds, cfg.sprt.elo0, cfg.sprt.elo1);
    } else {
        println!("  SPRT Gate:  Disabled");
    }
    println!("  State:      {}", run_state_desc);
    println!("  Controls:   Ctrl+C or type 'stop' + Enter to stop cleanly");
    println!("================================================================================\n");

    // Initialize graceful signal handler and interactive console listener
    signal::clear_stop_files();
    signal::init();

    let mut current_step = start_superbatch;

    while current_step <= superbatches {
        if signal::is_stop_requested() {
            println!("\n🛑 [STOP SIGNAL DETECTED] Safe stop before starting superbatch {}.", current_step);
            println!("   Checkpoints preserved. Exiting cleanly.");
            signal::clear_stop_files();
            return;
        }

        let next_flush = if current_step % save_interval == 0 {
            current_step
        } else {
            ((current_step / save_interval) + 1) * save_interval
        };
        let chunk_end = next_flush.min(superbatches);

        let schedule = TrainingSchedule {
            net_id: net_id.clone(),
            eval_scale: SCALE,
            steps: TrainingSteps {
                batch_size,
                batches_per_superbatch,
                start_superbatch: current_step,
                end_superbatch: chunk_end,
            },
            wdl_scheduler: wdl::ConstantWDL { value: wdl_val },
            lr_scheduler: lr::CosineDecayLR {
                initial_lr: max_lr,
                final_lr: min_lr,
                final_superbatch: superbatches,
            },
            save_rate: save_interval,
        };

        if let Some(ref l) = sf_loader {
            trainer.run(&schedule, &settings, l);
        } else if let Some(ref l) = direct_loader {
            trainer.run(&schedule, &settings, l);
        }

        let stop_requested = signal::is_stop_requested();
        if stop_requested {
            println!("\n🛑 [STOP SIGNAL DETECTED] Completing step {} weight export cleanly...", chunk_end);
        }

        let checkpoint_dir = format!("checkpoints/{net_id}-{chunk_end}");
        let raw_path = format!("{checkpoint_dir}/raw.bin");
        let exported_bin = format!("checkpoints/weights_screlu_step{chunk_end}.bin");
        let periodic_bin = format!("checkpoints/weights_step_{chunk_end}.bin");

        if Path::new(&raw_path).exists() {
            if let Err(e) = export::export_from_raw_checkpoint(Path::new(&raw_path), Path::new(&periodic_bin)) {
                eprintln!("Failed to export weights for step {}: {}", chunk_end, e);
                if stop_requested {
                    signal::clear_stop_files();
                    return;
                }
                current_step = chunk_end + 1;
                continue;
            }
            let _ = fs::copy(&periodic_bin, &exported_bin);
            let _ = fs::copy(&periodic_bin, "weights.bin");
            println!("💾 [Step {}/{}] Checkpoint exported -> {}", chunk_end, superbatches, periodic_bin);

            if stop_requested {
                println!("\n================================================================================");
                println!("🛑 SAFE MANUAL STOP COMPLETE: Stopped cleanly at step {}/{}", chunk_end, superbatches);
                println!("  Exported weights: {} | Active: weights.bin", periodic_bin);
                println!("  To resume: cargo run --release -- train");
                println!("================================================================================");
                signal::clear_stop_files();
                return;
            }

            match eval::evaluate_checkpoint_packs(Path::new(&exported_bin), &data_files, eval_positions_per_pack, wdl_val) {
                Ok((overall, non_mate, mate)) => {
                    let diff_mae_str = if baseline_mae < f64::MAX {
                        format!(" ({:+.6} vs champ)", overall.mae() - baseline_mae)
                    } else {
                        String::new()
                    };
                    let diff_smse_str = if baseline_smse < f64::MAX {
                        format!(" ({:+.6} vs champ)", overall.smse() - baseline_smse)
                    } else {
                        String::new()
                    };

                    println!(
                        "📊 [Eval Step {}/{}] Overall MAE: {:.6}{} | SMSE: {:.6}{} | Mate MAE: {:.6} ({:.1}M pos)",
                        chunk_end, superbatches,
                        overall.mae(), diff_mae_str,
                        overall.smse(), diff_smse_str,
                        mate.mae(),
                        overall.count as f64 / 1_000_000.0
                    );

                    // SPRT Promotion Policy: SPRT is the EXCLUSIVE gate for champion promotion
                    let champion_file = Path::new(CHAMPION_FILE);
                    let (promoted, sprt_stats) = if !champion_file.exists() {
                        println!("👑 [Baseline Established] Step {} registered as active champion.", chunk_end);
                        let _ = fs::copy(&exported_bin, CHAMPION_FILE);
                        baseline_mae = overall.mae();
                        baseline_smse = overall.smse();
                        let base_entry = format!(
                            "[{}] BASELINE CHAMPION ESTABLISHED\n  Step:    {}\n  Weights: {}\n  MAE:     {:.6} | SMSE: {:.6}\n  --------------------------------------------------------------------------------",
                            sprt::current_timestamp(), chunk_end, CHAMPION_FILE, baseline_mae, baseline_smse
                        );
                        sprt::append_to_match_log(&base_entry);
                        prune_outdated_checkpoints(chunk_end);
                        (true, None)
                    } else if cfg.sprt.enabled {
                        println!("⚔️ [SPRT Step {}] Running match vs active champion...", chunk_end);
                        match sprt::run_sprt_match(&cfg.sprt, Path::new(&exported_bin), champion_file) {
                            Ok(res) => {
                                let (passed, reason) = res.should_promote(cfg.sprt.elo1);
                                if passed {
                                    println!("🏆 [SPRT] Champion Promoted! ({}) +{} -{} ={} (Elo {:+.1} +/- {:.1}, LLR: {:.4})",
                                        reason,
                                        res.wins, res.losses, res.draws,
                                        res.elo_diff.unwrap_or(0.0), res.elo_error.unwrap_or(0.0),
                                        res.llr.unwrap_or(0.0));
                                    let _ = fs::copy(&exported_bin, CHAMPION_FILE);
                                    baseline_mae = overall.mae();
                                    baseline_smse = overall.smse();
                                    let promo_entry = format!(
                                        "[{}] 👑 CHAMPION PROMOTED\n  Step:     {}\n  Reason:   {}\n  Score:    +{} -{} ={} | Elo: {:+.1} +/- {:.1} | LLR: {:.4}\n  Promoted: {}\n  --------------------------------------------------------------------------------",
                                        sprt::current_timestamp(), chunk_end, reason, res.wins, res.losses, res.draws,
                                        res.elo_diff.unwrap_or(0.0), res.elo_error.unwrap_or(0.0),
                                        res.llr.unwrap_or(0.0),
                                        CHAMPION_FILE
                                    );
                                    sprt::append_to_match_log(&promo_entry);
                                    prune_outdated_checkpoints(chunk_end);
                                } else {
                                    println!("🛡️ [SPRT] Champion Retained: +{} -{} ={} (Elo {:+.1} +/- {:.1})",
                                        res.wins, res.losses, res.draws,
                                        res.elo_diff.unwrap_or(0.0), res.elo_error.unwrap_or(0.0));
                                }
                                (passed, Some(res))
                            }
                            Err(e) => {
                                eprintln!("SPRT execution failed: {}", e);
                                (false, None)
                            }
                        }
                    } else {
                        (false, None)
                    };

                    let global_batch_end = chunk_end * batches_per_superbatch;
                    let epoch = global_batch_end / total_dataset_batches + 1;
                    if promoted {
                        let mut rec = serde_json::json!({
                            "step": chunk_end,
                            "pack": format!("unified_epoch_{}_step_{}", epoch, chunk_end),
                            "epoch": epoch,
                            "wdl": wdl_val,
                            "mae": overall.mae(),
                            "smse": overall.smse(),
                            "bce": overall.bce(),
                            "mse": overall.mse(),
                            "rmse": overall.rmse(),
                            "non_mate_mae": non_mate.mae(),
                            "non_mate_smse": non_mate.smse(),
                            "mate_mae": mate.mae(),
                            "mate_smse": mate.smse(),
                            "positions": overall.count,
                            "weights_file": CHAMPION_FILE,
                        });
                        if let Some(ref s) = sprt_stats {
                            let (_, reason) = s.should_promote(cfg.sprt.elo1);
                            rec["sprt"] = serde_json::json!({
                                "decision": format!("{:?}", s.decision),
                                "promotion_reason": reason,
                                "wins": s.wins,
                                "losses": s.losses,
                                "draws": s.draws,
                                "total_games": s.total_games,
                                "elo_diff": s.elo_diff,
                                "elo_error": s.elo_error,
                                "llr": s.llr,
                                "lbound": s.lbound,
                                "ubound": s.ubound,
                            });
                        }
                        if let Ok(json_str) = serde_json::to_string_pretty(&rec) {
                            let _ = fs::write(baseline_path, json_str);
                        }
                    }

                    record_training_history(
                        chunk_end,
                        epoch,
                        global_batch_end * batch_size,
                        wdl_val,
                        overall.mae(),
                        overall.smse(),
                        overall.mse(),
                        mate.mae(),
                        non_mate.mae(),
                        promoted,
                        sprt_stats.as_ref(),
                    );
                }
                Err(e) => {
                    eprintln!("Evaluation failed for step {}: {}", chunk_end, e);
                }
            }
        }

        current_step = chunk_end + 1;
    }

    signal::clear_stop_files();
    println!("\n================================================================================");
    println!("🎉 TRAINING COMPLETE - ALL {} SUPERBATCHES REACHED!", superbatches);
    println!("================================================================================");
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct HistoryEntry {
    step: usize,
    epoch: usize,
    positions: usize,
    wdl: f32,
    mae: f64,
    smse: f64,
    mse: f64,
    mate_mae: f64,
    non_mate_mae: f64,
    promoted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    sprt_wins: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sprt_losses: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sprt_draws: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sprt_elo: Option<f64>,
}

#[allow(clippy::too_many_arguments)]
fn record_training_history(
    step: usize,
    epoch: usize,
    positions: usize,
    wdl: f32,
    mae: f64,
    smse: f64,
    mse: f64,
    mate_mae: f64,
    non_mate_mae: f64,
    promoted: bool,
    sprt_res: Option<&sprt::SprtMatchResult>,
) {
    let history_path = Path::new("checkpoints/training_history.json");
    let mut history: Vec<HistoryEntry> = if history_path.exists() {
        fs::read_to_string(history_path)
            .ok()
            .and_then(|c| serde_json::from_str(&c).ok())
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    history.retain(|h| h.step != step);
    history.push(HistoryEntry {
        step,
        epoch,
        positions,
        wdl,
        mae,
        smse,
        mse,
        mate_mae,
        non_mate_mae,
        promoted,
        sprt_wins: sprt_res.map(|s| s.wins),
        sprt_losses: sprt_res.map(|s| s.losses),
        sprt_draws: sprt_res.map(|s| s.draws),
        sprt_elo: sprt_res.and_then(|s| s.elo_diff),
    });
    history.sort_by_key(|h| h.step);

    if let Ok(json_str) = serde_json::to_string_pretty(&history) {
        let _ = fs::write(history_path, json_str);
    }
}

fn run_eval(args: &[String]) {
    let cfg = TrainerConfig::load_or_default();
    let default_weights = PathBuf::from(CHAMPION_FILE);
    let weights_path = args.get(2).map(Path::new).unwrap_or(&default_weights);
    let default_data = PathBuf::from(DATA_DIR);
    let target_data = args.get(3).map(Path::new).unwrap_or(&default_data);
    let positions_per_pack: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(cfg.eval_positions_per_pack);

    if !weights_path.exists() {
        eprintln!("Error: weights file '{}' not found.", weights_path.display());
        return;
    }

    let mut pack_paths: Vec<PathBuf> = Vec::new();
    if target_data.is_dir() {
        if let Ok(entries) = fs::read_dir(target_data) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|s| s.to_str()) == Some("binpack") {
                    pack_paths.push(p);
                }
            }
        }
        pack_paths.sort();
    } else if target_data.exists() {
        pack_paths.push(target_data.to_path_buf());
    }

    if pack_paths.is_empty() {
        eprintln!("Error: no .binpack files found in '{}'.", target_data.display());
        return;
    }

    let mut wdl_val: f32 = 0.20;
    for (idx, a) in args.iter().enumerate() {
        if a == "--wdl" && idx + 1 < args.len()
            && let Ok(v) = args[idx + 1].parse::<f32>() {
                wdl_val = v;
            }
    }

    println!("Starting Benchmark Evaluation across ALL {} pack(s):", pack_paths.len());
    println!("  Weights File:        {}", weights_path.display());
    println!("  Positions per pack:  {}", positions_per_pack);
    println!("  Total Est Positions: {}", pack_paths.len() * positions_per_pack);
    println!("  Activation:          SCReLU (NNUESGD6)");
    println!("  WDL lambda:          {:.2}", wdl_val);

    let start = std::time::Instant::now();
    match eval::evaluate_checkpoint_packs(weights_path, &pack_paths, positions_per_pack, wdl_val) {
        Ok((overall, non_mate, mate)) => {
            let elapsed = start.elapsed();
            println!("\n================================================================================");
            println!(
                "Evaluation Results across ALL {} packs ({} positions in {:.2}s, {:.0} pos/sec):",
                pack_paths.len(),
                overall.count,
                elapsed.as_secs_f64(),
                overall.count as f64 / elapsed.as_secs_f64()
            );
            println!(
                "  Overall:   MAE = {:.7}  |  SMSE = {:.7}  |  MSE = {:.7}  |  RMSE = {:.7}",
                overall.mae(), overall.smse(), overall.mse(), overall.rmse()
            );
            println!(
                "  Non-mate:  MAE = {:.7}  |  SMSE = {:.7}  |  MSE = {:.7}  ({} pos)",
                non_mate.mae(), non_mate.smse(), non_mate.mse(), non_mate.count
            );
            println!(
                "  Mate:      MAE = {:.7}  |  SMSE = {:.7}  |  MSE = {:.7}  ({} pos)",
                mate.mae(), mate.smse(), mate.mse(), mate.count
            );
            println!("================================================================================");
        }
        Err(e) => {
            eprintln!("Evaluation failed: {}", e);
        }
    }
}

fn run_inversions(args: &[String]) {
    let default_weights = PathBuf::from(CHAMPION_FILE);
    let weights_path = args.get(2).map(Path::new).unwrap_or(&default_weights);
    let default_data = PathBuf::from(DATA_DIR);
    let target_data = args.get(3).map(Path::new).unwrap_or(&default_data);
    let positions_per_pack: usize = args.get(4)
        .and_then(|s| if s == "all" { Some(0) } else { s.parse().ok() })
        .unwrap_or(2_000_000);

    if !weights_path.exists() {
        eprintln!("Error: weights file '{}' not found.", weights_path.display());
        return;
    }

    let mut pack_paths: Vec<PathBuf> = Vec::new();
    if target_data.is_dir() {
        if let Ok(entries) = fs::read_dir(target_data) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|s| s.to_str()) == Some("binpack") {
                    pack_paths.push(p);
                }
            }
        }
        pack_paths.sort();
    } else if target_data.exists() {
        pack_paths.push(target_data.to_path_buf());
    }

    if pack_paths.is_empty() {
        eprintln!("Error: no .binpack files found in '{}'.", target_data.display());
        return;
    }

    println!("================================================================================");
    println!("NNUE High-CP Evaluation Inversion & Sign Disagreement Analysis");
    println!("================================================================================");
    println!("  Weights File:          {}", weights_path.display());
    println!("  Packs Analyzed:        {}", pack_paths.len());
    if positions_per_pack == 0 {
        println!("  Positions per pack:    ALL available positions");
    } else {
        println!("  Positions per pack:    {} (Target Sample: {} positions)", 
            positions_per_pack, positions_per_pack * pack_paths.len());
    }
    println!("================================================================================\n");

    let start = std::time::Instant::now();
    match eval::evaluate_inversions(weights_path, &pack_paths, positions_per_pack) {
        Ok(stats) => {
            let elapsed = start.elapsed();
            let total = stats.total as f64;
            let pct = |n: u64| (n as f64 / total.max(1.0)) * 100.0;

            println!("\nAnalysis completed in {:.2}s ({:.0} pos/sec across {} positions).\n",
                elapsed.as_secs_f64(),
                total / elapsed.as_secs_f64().max(0.001),
                stats.total
            );

            println!("--------------------------------------------------------------------------------");
            println!("Inversion & Polarity Telemetry ({} positions, {:.2}s):", stats.total, elapsed.as_secs_f64());
            let pct_300_win = (stats.win_pred_loss_300 as f64 / stats.count_target_ge_300.max(1) as f64) * 100.0;
            let pct_300_loss = (stats.loss_pred_win_300 as f64 / stats.count_target_ge_300.max(1) as f64) * 100.0;
            println!("  Polar Inversions:    >=200 cp: {:>6} ({:.3}%) | >=300 cp: {:>5} ({:.3}%) | >=500 cp: {:>4} ({:.3}%)",
                stats.inv_200, pct(stats.inv_200), stats.inv_300, pct(stats.inv_300), stats.inv_500, pct(stats.inv_500));
            println!("  Polarity Mismatches: False Loss (+300): {:>5} ({:.2}%) | False Win (-300): {:>5} ({:.2}%)",
                stats.win_pred_loss_300, pct_300_win, stats.loss_pred_win_300, pct_300_loss);
            println!("  Error Spread:        >=300 cp: {:>7} ({:.1}%) | >=500 cp: {:>7} ({:.1}%) | >=800 cp: {:>7} ({:.1}%)",
                stats.err_ge_300, pct(stats.err_ge_300), stats.err_ge_500, pct(stats.err_ge_500), stats.err_ge_800, pct(stats.err_ge_800));

            if !stats.worst_inversions.is_empty() {
                println!("\n  Top Extreme Inversions:");
                for (idx, (tgt, pred, err, desc)) in stats.worst_inversions.iter().take(3).enumerate() {
                    println!("    #{}: Target = {:+7.1} cp | Model = {:+7.1} cp (Δ {:6.1} cp) | {}",
                        idx + 1, tgt, pred, err, desc);
                }
            }
            println!("--------------------------------------------------------------------------------");
        }
        Err(e) => {
            eprintln!("Inversion analysis failed: {}", e);
        }
    }
}

fn run_dead_neurons(args: &[String]) {
    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v" || a == "--all");
    let filtered_args: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    let default_weights = PathBuf::from(CHAMPION_FILE);
    let weights_path = filtered_args.get(2).map(|s| Path::new(s.as_str())).unwrap_or(&default_weights);
    let default_data = PathBuf::from(DATA_DIR);
    let target_data = filtered_args.get(3).map(|s| Path::new(s.as_str())).unwrap_or(&default_data);
    let positions_per_pack: usize = filtered_args.get(4).and_then(|s| s.parse().ok()).unwrap_or(50_000);

    if !weights_path.exists() {
        eprintln!("Error: weights file '{}' not found.", weights_path.display());
        return;
    }

    let mut pack_paths: Vec<PathBuf> = Vec::new();
    if target_data.is_dir() {
        if let Ok(entries) = fs::read_dir(target_data) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|s| s.to_str()) == Some("binpack") {
                    pack_paths.push(p);
                }
            }
        }
        pack_paths.sort();
    } else if target_data.exists() {
        pack_paths.push(target_data.to_path_buf());
    }

    if pack_paths.is_empty() {
        eprintln!("Error: no .binpack files found in '{}'.", target_data.display());
        return;
    }

    let weights = match eval::NnueWeights::load_from_file(weights_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to load weights: {}", e);
            return;
        }
    };

    println!("================================================================================");
    println!("NNUE Dead Neuron & Network Activity Analysis");
    println!("================================================================================");
    println!("  Weights File:          {}", weights_path.display());
    println!("  Activation:            SCReLU (clamp(x, 0, 1)^2)");
    println!("  Packs Analyzed:        {}", pack_paths.len());
    println!("  Positions per pack:    {}", positions_per_pack);
    println!("  Total Sample Target:   {} positions", pack_paths.len() * positions_per_pack);
    println!("================================================================================\n");

    let start = std::time::Instant::now();
    let diag = match diagnostics::analyze_weights(weights_path, &pack_paths, positions_per_pack) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Diagnostic analysis failed: {}", e);
            return;
        }
    };
    let elapsed = start.elapsed();

    println!("\nAnalysis completed in {:.2}s ({:.0} pos/sec across {} positions).\n",
        elapsed.as_secs_f64(),
        diag.total_positions as f64 / elapsed.as_secs_f64().max(0.001),
        diag.total_positions
    );

    // 1. Accumulator (L0: 256 neurons)
    let mut accum_dead = Vec::new();
    let mut accum_low = Vec::new();
    let mut accum_rates = Vec::new();

    let h1_size = weights.hidden1_size;
    let h2_size = weights.hidden2_size;
    let num_buckets = weights.num_buckets;
    let l1_in_size = weights.l1_input_size;

    for i in 0..256 {
        let comb = &diag.accum_combined.neurons[i];
        let rate = comb.firing_rate();
        accum_rates.push(rate);

        let mut out_norm_sq = 0.0f32;
        if weights.is_pairwise {
            let p = i % 128;
            for h in 0..h1_size {
                let w_stm = weights.l1w[h * 256 + p];
                let w_ntm = weights.l1w[h * 256 + 128 + p];
                out_norm_sq += w_stm * w_stm + w_ntm * w_ntm;
            }
        } else {
            for h in 0..h1_size {
                let w_stm = weights.l1w[h * 512 + i];
                let w_ntm = weights.l1w[h * 512 + 256 + i];
                out_norm_sq += w_stm * w_stm + w_ntm * w_ntm;
            }
        }
        let out_norm = out_norm_sq.sqrt();

        if comb.fires == 0 || out_norm < 1e-7 {
            accum_dead.push((i, rate, comb.mean_activation(), comb.min_pre, comb.max_pre, weights.l0b[i], out_norm));
        } else if rate < 1.0 {
            accum_low.push((i, rate, comb.mean_activation(), comb.min_pre, comb.max_pre, weights.l0b[i], out_norm));
        }
    }

    accum_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let accum_avg_rate: f64 = accum_rates.iter().sum::<f64>() / 256.0;
    let accum_median_rate = accum_rates[128];
    let accum_min_rate = accum_rates[0];
    let accum_max_rate = accum_rates[255];

    // 2. Hidden Layer 1
    let mut l1_dead = Vec::new();
    let mut l1_low = Vec::new();
    let mut l1_rates = Vec::new();
    let mut l1_details = Vec::new();

    for i in 0..h1_size {
        let n = &diag.l1.neurons[i];
        let rate = n.firing_rate();
        l1_rates.push(rate);

        let in_norm: f32 = weights.l1w[i * l1_in_size..(i + 1) * l1_in_size].iter().map(|w| w * w).sum::<f32>().sqrt();
        let out_norm: f32 = (0..h2_size).map(|j| {
            let w = weights.l2w[j * h1_size + i];
            w * w
        }).sum::<f32>().sqrt();

        let is_dead = n.fires == 0 || in_norm < 1e-7 || out_norm < 1e-7;
        if is_dead {
            l1_dead.push(i);
            l1_details.push((i, rate, n.mean_activation(), n.min_pre, n.max_pre, weights.l1b[i], in_norm, out_norm, true));
        } else if rate < 1.0 {
            l1_low.push(i);
            l1_details.push((i, rate, n.mean_activation(), n.min_pre, n.max_pre, weights.l1b[i], in_norm, out_norm, false));
        } else if verbose {
            l1_details.push((i, rate, n.mean_activation(), n.min_pre, n.max_pre, weights.l1b[i], in_norm, out_norm, false));
        }
    }
    l1_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let l1_min_rate = l1_rates[0];
    let l1_med_rate = l1_rates[h1_size / 2];
    let l1_avg_rate: f64 = l1_rates.iter().sum::<f64>() / h1_size as f64;
    let l1_max_rate = l1_rates[h1_size - 1];

    // 3. Hidden Layer 2
    let mut l2_dead = Vec::new();
    let mut l2_low = Vec::new();
    let mut l2_rates = Vec::new();
    let mut l2_details = Vec::new();

    for j in 0..h2_size {
        let n = &diag.l2.neurons[j];
        let rate = n.firing_rate();
        l2_rates.push(rate);

        let in_norm: f32 = weights.l2w[j * h1_size..(j + 1) * h1_size].iter().map(|w| w * w).sum::<f32>().sqrt();
        let out_norm: f32 = (0..num_buckets).map(|b| {
            let w = weights.l3w[b * h2_size + j];
            w * w
        }).sum::<f32>().sqrt();

        let is_dead = n.fires == 0 || in_norm < 1e-7 || out_norm < 1e-7;
        if is_dead {
            l2_dead.push(j);
            l2_details.push((j, rate, n.mean_activation(), n.min_pre, n.max_pre, weights.l2b[j], in_norm, out_norm, true));
        } else if rate < 1.0 {
            l2_low.push(j);
            l2_details.push((j, rate, n.mean_activation(), n.min_pre, n.max_pre, weights.l2b[j], in_norm, out_norm, false));
        } else if verbose {
            l2_details.push((j, rate, n.mean_activation(), n.min_pre, n.max_pre, weights.l2b[j], in_norm, out_norm, false));
        }
    }
    l2_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let l2_min_rate = l2_rates[0];
    let l2_med_rate = l2_rates[h2_size / 2];
    let l2_avg_rate: f64 = l2_rates.iter().sum::<f64>() / h2_size as f64;
    let l2_max_rate = l2_rates[h2_size - 1];

    let l1_label = format!("Layer 1 (Dense {}->{})", l1_in_size, h1_size);
    let l2_label = format!("Layer 2 (Dense {}->{})", h1_size, h2_size);

    println!("--------------------------------------------------------------------------------");
    println!("{:<24} | {:>7} | {:>10} | {:>6} | {:<32}",
        "Layer", "Neurons", "Active", "Dead", "Firing Rates (Min / Med / Mean / Max)");
    println!("{:-<24}-+-{:-<7}-+-{:-<10}-+-{:-<6}-+-{:-<32}", "", "", "", "", "");
    println!("{:<24} | {:>7} | {:>9.1}% | {:>6} | {:>5.1}% / {:>5.1}% / {:>5.1}% / {:>5.1}%",
        "Layer 0 (Accumulator)", 256, (256 - accum_dead.len()) as f64 / 256.0 * 100.0, accum_dead.len(),
        accum_min_rate, accum_median_rate, accum_avg_rate, accum_max_rate);
    println!("{:<24} | {:>7} | {:>9.1}% | {:>6} | {:>5.1}% / {:>5.1}% / {:>5.1}% / {:>5.1}%",
        l1_label, h1_size, (h1_size - l1_dead.len()) as f64 / h1_size as f64 * 100.0, l1_dead.len(),
        l1_min_rate, l1_med_rate, l1_avg_rate, l1_max_rate);
    println!("{:<24} | {:>7} | {:>9.1}% | {:>6} | {:>5.1}% / {:>5.1}% / {:>5.1}% / {:>5.1}%",
        l2_label, h2_size, (h2_size - l2_dead.len()) as f64 / h2_size as f64 * 100.0, l2_dead.len(),
        l2_min_rate, l2_med_rate, l2_avg_rate, l2_max_rate);
    println!("--------------------------------------------------------------------------------");

    let total_dead = accum_dead.len() + l1_dead.len() + l2_dead.len();
    if total_dead > 0 || !accum_low.is_empty() || !l1_low.is_empty() || !l2_low.is_empty() || verbose {
        let label = if verbose { "Neuron Details" } else { "Problematic / Inactive Neurons" };
        println!("\n⚠️ {label}:");
        println!("{:<8} | {:<10} | {:>11} | {:>10} | {:<24} | {:>8} | {:>8} | {:>8} | Status",
            "Neuron", "Layer", "Firing Rate", "Mean Act", "Pre-Act [Min, Max]", "Bias", "In-Norm", "Out-Norm");
        println!("{:-<8}-+-{:-<10}-+-{:-<11}-+-{:-<10}-+-{:-<24}-+-{:-<8}-+-{:-<8}-+-{:-<8}-+-------", "", "", "", "", "", "", "", "");
        for &(i, rate, mean_a, min_p, max_p, bias, out_n) in &accum_dead {
            println!("{:<8} | {:<10} | {:>10.2}% | {:>10.6} | [{:>10.4}, {:>10.4}] | {:>8.4} | {:>8} | {:>8.4} | ❌ DEAD",
                format!("#{i}"), "Layer 0", rate, mean_a, min_p, max_p, bias, "-", out_n);
        }
        for &(i, rate, mean_a, min_p, max_p, bias, in_n, out_n, is_dead) in &l1_details {
            println!("{:<8} | {:<10} | {:>10.2}% | {:>10.6} | [{:>10.4}, {:>10.4}] | {:>8.4} | {:>8.4} | {:>8.4} | {}",
                format!("#{i}"), "Layer 1", rate, mean_a, min_p, max_p, bias, in_n, out_n, if is_dead { "❌ DEAD" } else { "⚠️ LOW" });
        }
        for &(j, rate, mean_a, min_p, max_p, bias, in_n, out_n, is_dead) in &l2_details {
            println!("{:<8} | {:<10} | {:>10.2}% | {:>10.6} | [{:>10.4}, {:>10.4}] | {:>8.4} | {:>8.4} | {:>8.4} | {}",
                format!("#{j}"), "Layer 2", rate, mean_a, min_p, max_p, bias, in_n, out_n, if is_dead { "❌ DEAD" } else { "⚠️ LOW" });
        }
    } else {
        println!("✅ All {} neurons across all layers are active and healthy!", 256 + HIDDEN1_SIZE + HIDDEN2_SIZE);
    }

    let total_neurons = 256 + HIDDEN1_SIZE + HIDDEN2_SIZE;
    let total_active = total_neurons - total_dead;
    println!("\nVerdict: {}/{} neurons active ({:.1}% network utilization).{}",
        total_active, total_neurons, total_active as f64 / total_neurons as f64 * 100.0,
        if total_dead == 0 { " (All layers healthy)" } else { " (Dead neurons listed above)" }
    );
    println!("================================================================================\n");
}

fn run_compare(args: &[String]) {
    let latest_folder = config::detect_latest_checkpoint().map(|(_, p)| p.to_string_lossy().to_string());
    let default_dir = latest_folder.unwrap_or_else(|| "checkpoints/kitty_bullet_screlu-115".to_string());
    let checkpoint_dir = args.get(2).map(|s| s.as_str()).unwrap_or(&default_dir);

    println!("Loading Bullet model from {}...", checkpoint_dir);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(KittyHalfKAHM)
        .output_buckets(MajorMinorCount::<8>)
        .save_format(&[
            SavedFormat::id("l0w"),
            SavedFormat::id("l0b"),
            SavedFormat::id("l1w"),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w"),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w"),
            SavedFormat::id("l3b"),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", INPUT_SIZE, ACCUM_SIZE);
            let l1 = builder.new_affine("l1", PAIRWISE_SIZE, HIDDEN1_SIZE);
            let l2 = builder.new_affine("l2", HIDDEN1_SIZE, HIDDEN2_SIZE);
            let l3 = builder.new_affine("l3", HIDDEN2_SIZE, NUM_OUTPUT_BUCKETS);

            // Vectorized pairwise interaction (Hadamard product) between halves of each perspective
            let stm_act = l0.forward(stm_inputs).screlu();
            let stm_hidden = stm_act.slice_rows(0, ACCUM_SIZE / 2) * stm_act.slice_rows(ACCUM_SIZE / 2, ACCUM_SIZE);

            let ntm_act = l0.forward(ntm_inputs).screlu();
            let ntm_hidden = ntm_act.slice_rows(0, ACCUM_SIZE / 2) * ntm_act.slice_rows(ACCUM_SIZE / 2, ACCUM_SIZE);

            let pairwise = stm_hidden.concat(ntm_hidden);

            let h1 = l1.forward(pairwise).screlu();
            let h2 = l2.forward(h1).screlu();
            l3.forward(h2).select(output_buckets)
        });

    trainer.load_from_checkpoint(checkpoint_dir);
    println!("Successfully loaded Bullet checkpoint into GPU!");

    let exported_path = Path::new(CHAMPION_FILE);
    let cpu_weights = eval::NnueWeights::load_from_file(exported_path).expect("Failed to load exported weights");

    let test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
        "8/8/8/4k3/8/8/4K3/4R3 w - - 0 1",
    ];

    println!("\n{:<60} | {:<25} | {:<12}", "FEN", "Bullet GPU", "CPU weights.bin");
    println!("{:-<60}-+-{:-<25}-+-{:-<12}", "", "", "");

    for fen in test_fens {
        let bullet_raw = trainer.eval_raw_output(fen)[0];
        let bullet_sig = 1.0 / (1.0 + (-bullet_raw).exp());
        let board = format!("{fen} | 0 | 0.0").parse::<bulletformat::ChessBoard>().unwrap();
        let cpu_pred = cpu_weights.evaluate(&board);
        println!("{:<60} | raw={:8.4} (sig={:.4}) | sig={:.4}", 
            if fen.len() > 60 { &fen[..60] } else { fen },
            bullet_raw, bullet_sig, cpu_pred);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let command = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match command {
        "train" | "resume" => run_train(&args),
        "stop" => run_stop(),
        "status" => run_status(),
        "sprt" => run_sprt(&args),
        "check" => run_check(&args),
        "eval" => run_eval(&args),
        "export" => run_export(&args),
        "scratch" => {
            let cfg = TrainerConfig::load_or_default();
            let default_out = PathBuf::from(SCRATCH_FILE);
            let out = args.get(2).map(PathBuf::from).unwrap_or(default_out);
            export::generate_scratch_kitty_weights(&out, cfg.bias_l0, cfg.bias_l1, cfg.bias_l2).expect("Failed to generate scratch weights");
            println!("Scratch weights generated successfully at {}", out.display());
        }
        "compare" => run_compare(&args),
        "info" => run_info(),
        "dead-neurons" | "dead" => run_dead_neurons(&args),
        "inversions" => run_inversions(&args),
        "import" => run_import(&args),
        "help" | "--help" | "-h" => print_usage(),
        other => {
            if other != "help" {
                eprintln!("Unknown command: '{}'\n", other);
            }
            print_usage();
        }
    }
}
