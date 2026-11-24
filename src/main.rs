mod inputs;
mod export;
mod convert;
mod eval;
mod diagnostics;
mod config;
mod sprt;
mod signal;
mod spsa;

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

use inputs::{KittyHalfKAHM, TotalPieceCount};
use config::{
    TrainerConfig, CHAMPION_FILE, DATA_DIR, ENGINE_WEIGHTS_FILE,
    INPUT_SIZE, NUM_KING_BUCKETS, ACCUM_SIZE, PAIRWISE_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, NUM_OUTPUT_BUCKETS, CHECKPOINT_SIZE,
};

const SCALE: f32 = 400.0;

fn print_usage() {
    println!("KittyEngineTrainer - High-Performance NNUE Trainer via Bullet");
    println!("============================================================");
    println!("Architecture: HalfKA-16 (12288 -> 768)x2 -> 768 (Pairwise CReLU) -> [16 -> 32 -> 1] x 8 (NNUESGD8)");
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
    println!("  spsa [SUBCOMMAND]           Autonomous SPSA search heuristic tuner (run, status, verify, deploy, reset)");
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
    println!("KittyEngine NNUE v8 Layout Specification:");
    println!("  Input Feature Space:   {} (HalfKAv2-16 HM, {} king buckets x 768 planes + l0f factoriser)", INPUT_SIZE, NUM_KING_BUCKETS);
    println!("  Dual Accumulator:      {} per perspective (QA = 255)", ACCUM_SIZE);
    println!("  Pairwise Interaction:  (384 + 384) -> {} (Pairwise CReLU within STM/NTM)", PAIRWISE_SIZE);
    println!("  Hidden Layer 1:        {} -> {} x {} Buckets (SCReLU, int8 QB = 64)", PAIRWISE_SIZE, HIDDEN1_SIZE, NUM_OUTPUT_BUCKETS);
    println!("  Hidden Layer 2:        {} -> {} x {} Buckets (SCReLU, int16 QB = 64)", HIDDEN1_SIZE, HIDDEN2_SIZE, NUM_OUTPUT_BUCKETS);
    println!("  Output Layer:          {} -> 1 x {} Buckets (Total Piece Count, logit QB = 64)", HIDDEN2_SIZE, NUM_OUTPUT_BUCKETS);
    println!("  Centipawn Scale:       {:.1} cp", SCALE);
    println!("  Target Weight Size:    {} bytes (NNUESGD8 format)", CHECKPOINT_SIZE);
}

fn run_stop() {
    println!("================================================================================");
    println!("🛑 SENDING SAFE MANUAL STOP SIGNAL");
    println!("================================================================================");
    match signal::create_stop_file() {
        Ok(()) => {
            println!("  Trigger file created: {}", signal::STOP_TRIGGER_FILE);
            println!("  The active training process will finish its current superbatch, cleanly");
            println!("  export and verify weights to 'checkpoints/weights_step_<N>.bin',");
            println!("  and exit safely without any data corruption.");
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
            let quantised = folder.join("quantised.bin");
            let out = custom_out.unwrap_or_else(|| PathBuf::from(format!("weights_step_{}.bin", step)));
            (quantised, out)
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
            let quantised = folder.join("quantised.bin");
            let out = custom_out.unwrap_or_else(|| PathBuf::from(format!("weights_step_{}.bin", step)));
            (quantised, out)
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

    let src_name = src_path.file_name().and_then(|s| s.to_str());
    if src_name != Some("quantised.bin") && src_name != Some("raw.bin") && src_path.extension().and_then(|s| s.to_str()) == Some("bin") {
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

    if let Err(e) = sprt::run_sprt_match(&cfg.sprt, &cand_path, &champ_path) {
        eprintln!("SPRT execution failed: {}", e);
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
    println!();
    println!("⚔️ SPRT VERIFICATION SETTINGS:");
    let (resolved_engine, _) = cfg.sprt.resolve_engine();
    println!("  - SPRT Enabled:        {}", cfg.sprt.enabled);
    println!("  - Cutechess CLI:       {}", cfg.sprt.cutechess_path);
    println!("  - Engine Name:         {}", cfg.sprt.engine_name);
    println!("  - Engine Executable:   {}", resolved_engine.display());
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
        .output_buckets(TotalPieceCount::<NUM_OUTPUT_BUCKETS>)
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
            l0.init_with_effective_input_size(32);

            let l1 = builder.new_affine("l1", PAIRWISE_SIZE, NUM_OUTPUT_BUCKETS * HIDDEN1_SIZE);
            let l2 = builder.new_affine("l2", HIDDEN1_SIZE, NUM_OUTPUT_BUCKETS * HIDDEN2_SIZE);
            let l3 = builder.new_affine("l3", HIDDEN2_SIZE, NUM_OUTPUT_BUCKETS);

            let ft = |input, start, end| l0.slice(start, end).forward(input).crelu();
            let stm_hidden = ft(stm_inputs, 0, ACCUM_SIZE / 2) * ft(stm_inputs, ACCUM_SIZE / 2, ACCUM_SIZE);
            let ntm_hidden = ft(ntm_inputs, 0, ACCUM_SIZE / 2) * ft(ntm_inputs, ACCUM_SIZE / 2, ACCUM_SIZE);

            let hl1 = stm_hidden.concat(ntm_hidden);
            let hl2 = l1.forward(hl1).select(output_buckets).screlu();
            let hl3 = l2.forward(hl2).select(output_buckets).screlu();
            l3.forward(hl3).select(output_buckets)
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

    let mut ordered_train_files = train_files.clone();
    if start_superbatch > 1 && ordered_train_files.len() > 1 {
        let rot = ((start_superbatch - 1) * ordered_train_files.len() / superbatches.max(1)) % ordered_train_files.len();
        ordered_train_files.rotate_left(rot);
    }
    let pack_strs: Vec<String> = ordered_train_files.iter().map(|p| p.to_string_lossy().to_string()).collect();
    let pack_refs: Vec<&str> = pack_strs.iter().map(|s| s.as_str()).collect();

    let direct_loader = if !is_sfbinpack {
        Some(loader::DirectSequentialDataLoader::new(&pack_refs))
    } else {
        None
    };

    let sf_loader = if is_sfbinpack {
        Some(convert::PersistentDataReader::new(
            loader::sfbinpack::SfBinpackLoader::new_concat_multiple(
                &pack_refs,
                512,
                8,
                |entry| !entry.pos.is_checked(entry.pos.side_to_move()) && entry.score.abs() <= 10000,
            ),
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
    println!("KittyEngineTrainer | Architecture: HalfKA-16 (12288->768)x2 -> 768 (Pairwise CReLU) -> [16->32->1]x8");
    println!("  Dataset:    {} files ({})", train_files.len(), total_pos_str);
    println!("  Schedule:   Steps {}..{} | {} batches/sb ({:.1}M pos/sb) | Batch: {}",
        start_superbatch, superbatches, batches_per_superbatch, positions_per_superbatch as f64 / 1_000_000.0, batch_size);
    println!("  Learning:   LR: {:.6} -> {:.6} (Cosine) | WDL: {:.2} | Save: Every {} sb ({:.1}M pos)",
        max_lr, min_lr, wdl_val, save_interval, (save_interval * positions_per_superbatch) as f64 / 1_000_000.0);
    println!("  Validation: {} pos/pack ({:.2}M total) | Factoriser: l0f (16 king buckets)",
        eval_positions_per_pack, total_eval_pos as f64 / 1_000_000.0);
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

        // Run 1 superbatch at a time so stop requests take effect immediately after the active superbatch
        let is_interval_step = (current_step % save_interval == 0) || (current_step == superbatches);

        let schedule = TrainingSchedule {
            net_id: net_id.clone(),
            eval_scale: SCALE,
            steps: TrainingSteps {
                batch_size,
                batches_per_superbatch,
                start_superbatch: current_step,
                end_superbatch: current_step,
            },
            wdl_scheduler: wdl::ConstantWDL { value: wdl_val },
            lr_scheduler: lr::CosineDecayLR {
                initial_lr: max_lr,
                final_lr: min_lr,
                final_superbatch: superbatches,
            },
            save_rate: 1, // save raw checkpoint at the end of each superbatch
        };

        if let Some(ref l) = sf_loader {
            trainer.run(&schedule, &settings, l);
        } else if let Some(ref l) = direct_loader {
            trainer.run(&schedule, &settings, l);
        }

        let stop_requested = signal::is_stop_requested();
        let checkpoint_dir = format!("checkpoints/{net_id}-{current_step}");
        let quantised_path = format!("{checkpoint_dir}/quantised.bin");
        let exported_bin = format!("checkpoints/weights_screlu_step{current_step}.bin");
        let periodic_bin = format!("checkpoints/weights_step_{current_step}.bin");

        if stop_requested || is_interval_step {
            if Path::new(&quantised_path).exists() {
                if let Err(e) = export::export_from_raw_checkpoint(Path::new(&quantised_path), Path::new(&periodic_bin)) {
                    eprintln!("Failed to export weights for step {}: {}", current_step, e);
                    if stop_requested {
                        signal::clear_stop_files();
                        return;
                    }
                    current_step += 1;
                    continue;
                }
                let _ = fs::copy(&periodic_bin, &exported_bin);
                println!("💾 [Step {}/{}] Checkpoint exported -> {}", current_step, superbatches, periodic_bin);
            }

            if stop_requested {
                println!("\n================================================================================");
                println!("🛑 SAFE MANUAL STOP COMPLETE: Stopped cleanly right after superbatch {}/{}", current_step, superbatches);
                println!("  Exported weights: {}", periodic_bin);
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
                        current_step, superbatches,
                        overall.mae(), diff_mae_str,
                        overall.smse(), diff_smse_str,
                        mate.mae(),
                        overall.count as f64 / 1_000_000.0
                    );

                    // SPRT Promotion Policy: SPRT is the EXCLUSIVE gate for champion promotion
                    let sync_champion_weights = |src: &str| {
                        let _ = fs::copy(src, CHAMPION_FILE);
                        let engine_weights = Path::new(ENGINE_WEIGHTS_FILE);
                        if let Some(parent) = engine_weights.parent() {
                            let _ = fs::create_dir_all(parent);
                        }
                        if let Err(e) = fs::copy(src, engine_weights) {
                            eprintln!("⚠️ Failed to sync champion weights to {}: {}", ENGINE_WEIGHTS_FILE, e);
                        } else {
                            println!("  📦 Synced champion weights -> {}", ENGINE_WEIGHTS_FILE);
                        }
                    };
                    let champion_file = Path::new(CHAMPION_FILE);
                    let (promoted, sprt_stats) = if !champion_file.exists() {
                        println!("👑 [Baseline Established] Step {} registered as active champion.", current_step);
                        sync_champion_weights(&exported_bin);
                        baseline_mae = overall.mae();
                        baseline_smse = overall.smse();
                        let base_entry = format!(
                            "[{}] BASELINE CHAMPION ESTABLISHED\n  Step:    {}\n  Weights: {}\n  MAE:     {:.6} | SMSE: {:.6}\n  --------------------------------------------------------------------------------",
                            sprt::current_timestamp(), current_step, CHAMPION_FILE, baseline_mae, baseline_smse
                        );
                        sprt::append_to_match_log(&base_entry);
                        prune_outdated_checkpoints(current_step);
                        (true, None)
                    } else if cfg.sprt.enabled {
                        match sprt::run_sprt_match(&cfg.sprt, Path::new(&exported_bin), champion_file) {
                            Ok(res) => {
                                let (passed, reason) = res.should_promote(cfg.sprt.elo1);
                                if passed {
                                    println!("  👑 [Promotion] Step {} crowned as new Champion! Outdated checkpoints cleaned.", current_step);
                                    sync_champion_weights(&exported_bin);
                                    baseline_mae = overall.mae();
                                    baseline_smse = overall.smse();
                                    let promo_entry = format!(
                                        "[{}] 👑 CHAMPION PROMOTED\n  Step:     {}\n  Reason:   {}\n  Score:    +{} -{} ={} | Elo: {:+.1} +/- {:.1} | LLR: {:.4}\n  Promoted: {}\n  --------------------------------------------------------------------------------",
                                        sprt::current_timestamp(), current_step, reason, res.wins, res.losses, res.draws,
                                        res.elo_diff.unwrap_or(0.0), res.elo_error.unwrap_or(0.0),
                                        res.llr.unwrap_or(0.0),
                                        CHAMPION_FILE
                                    );
                                    sprt::append_to_match_log(&promo_entry);
                                    prune_outdated_checkpoints(current_step);
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

                    let global_batch_end = current_step * batches_per_superbatch;
                    let epoch = global_batch_end / total_dataset_batches + 1;
                    if promoted {
                        let mut rec = serde_json::json!({
                            "step": current_step,
                            "pack": format!("unified_epoch_{}_step_{}", epoch, current_step),
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
                        current_step,
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
                    eprintln!("Evaluation failed for step {}: {}", current_step, e);
                }
            }
        } else {
            // Intermediate superbatch: clean up temporary raw checkpoint folder so disk space is not wasted
            if Path::new(&checkpoint_dir).exists() {
                let _ = fs::remove_dir_all(&checkpoint_dir);
            }
        }

        current_step += 1;
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
    println!("  Activation:          Pairwise CReLU + SCReLU head (NNUESGD8)");
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
    println!("  Activation:            Pairwise CReLU (clamp(lo,0,1) * clamp(hi,0,1))");
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

    let n = weights.accum_size;
    let half = n / 2;
    let h1_n = weights.hidden1_size;
    let h2_n = weights.hidden2_size;
    let num_buckets = weights.num_buckets;

    let mut accum_dead = Vec::new();
    let mut accum_low = Vec::new();
    let mut accum_rates = Vec::new();

    for i in 0..n {
        let comb = &diag.accum_combined.neurons[i];
        let rate = comb.firing_rate();
        accum_rates.push(rate);

        let p = i % half;
        let mut out_norm_sq = 0.0f32;
        for b in 0..num_buckets {
            for r in 0..h1_n {
                let row_base = (b * h1_n + r) * n;
                let w_stm = weights.l1w[row_base + p];
                let w_ntm = weights.l1w[row_base + half + p];
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
    let accum_avg_rate: f64 = accum_rates.iter().sum::<f64>() / n as f64;
    let accum_median_rate = accum_rates[n / 2];
    let accum_min_rate = accum_rates[0];
    let accum_max_rate = accum_rates[n - 1];

    let l1_total = num_buckets * h1_n;
    let l1_dead = diag.l1.neurons.iter().filter(|x| x.fires == 0).count();
    let mut l1_rates: Vec<f64> = diag.l1.neurons.iter().map(|x| x.firing_rate()).collect();
    l1_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let l2_total = num_buckets * h2_n;
    let l2_dead = diag.l2.neurons.iter().filter(|x| x.fires == 0).count();
    let mut l2_rates: Vec<f64> = diag.l2.neurons.iter().map(|x| x.firing_rate()).collect();
    l2_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("--------------------------------------------------------------------------------");
    println!("{:<24} | {:>7} | {:>10} | {:>6} | {:<32}",
        "Layer", "Neurons", "Active", "Dead", "Firing Rates (Min / Med / Mean / Max)");
    println!("{:-<24}-+-{:-<7}-+-{:-<10}-+-{:-<6}-+-{:-<32}", "", "", "", "", "");
    println!("{:<24} | {:>7} | {:>9.1}% | {:>6} | {:>5.1}% / {:>5.1}% / {:>5.1}% / {:>5.1}%",
        "Layer 0 (Accumulator)", n, (n - accum_dead.len()) as f64 / n as f64 * 100.0, accum_dead.len(),
        accum_min_rate, accum_median_rate, accum_avg_rate, accum_max_rate);
    println!("{:<24} | {:>7} | {:>9.1}% | {:>6} | {:>5.1}% / {:>5.1}% / {:>5.1}% / {:>5.1}%",
        "Layer 1 (16 x 8)", l1_total, (l1_total - l1_dead) as f64 / l1_total as f64 * 100.0, l1_dead,
        l1_rates[0], l1_rates[l1_total / 2], l1_rates.iter().sum::<f64>() / l1_total as f64, l1_rates[l1_total - 1]);
    println!("{:<24} | {:>7} | {:>9.1}% | {:>6} | {:>5.1}% / {:>5.1}% / {:>5.1}% / {:>5.1}%",
        "Layer 2 (32 x 8)", l2_total, (l2_total - l2_dead) as f64 / l2_total as f64 * 100.0, l2_dead,
        l2_rates[0], l2_rates[l2_total / 2], l2_rates.iter().sum::<f64>() / l2_total as f64, l2_rates[l2_total - 1]);
    println!("--------------------------------------------------------------------------------");

    if !accum_dead.is_empty() || !accum_low.is_empty() || verbose {
        let label = if verbose { "Neuron Details" } else { "Problematic / Inactive Neurons" };
        println!("\n⚠️ {label}:");
        println!("{:<8} | {:<10} | {:>11} | {:>10} | {:<24} | {:>8} | {:>8} | Status",
            "Neuron", "Layer", "Firing Rate", "Mean Act", "Pre-Act [Min, Max]", "Bias", "Out-Norm");
        println!("{:-<8}-+-{:-<10}-+-{:-<11}-+-{:-<10}-+-{:-<24}-+-{:-<8}-+-{:-<8}-+-------", "", "", "", "", "", "", "");
        for &(i, rate, mean_a, min_p, max_p, bias, out_n) in &accum_dead {
            println!("{:<8} | {:<10} | {:>10.2}% | {:>10.6} | [{:>10.4}, {:>10.4}] | {:>8.4} | {:>8.4} | ❌ DEAD",
                format!("#{i}"), "Layer 0", rate, mean_a, min_p, max_p, bias, out_n);
        }
        for &(i, rate, mean_a, min_p, max_p, bias, out_n) in &accum_low {
            println!("{:<8} | {:<10} | {:>10.2}% | {:>10.6} | [{:>10.4}, {:>10.4}] | {:>8.4} | {:>8.4} | ⚠️ LOW",
                format!("#{i}"), "Layer 0", rate, mean_a, min_p, max_p, bias, out_n);
        }
    }

    let total_neurons = n + l1_total + l2_total;
    let total_dead = accum_dead.len() + l1_dead + l2_dead;
    let total_active = total_neurons - total_dead;
    println!("\nVerdict: {}/{} neurons active ({:.1}% network utilization).{}",
        total_active, total_neurons, total_active as f64 / total_neurons as f64 * 100.0,
        if total_dead == 0 { " (All layers healthy)" } else { " (Dead neurons detected)" }
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
        .output_buckets(TotalPieceCount::<NUM_OUTPUT_BUCKETS>)
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
            l0.init_with_effective_input_size(32);

            let l1 = builder.new_affine("l1", PAIRWISE_SIZE, NUM_OUTPUT_BUCKETS * HIDDEN1_SIZE);
            let l2 = builder.new_affine("l2", HIDDEN1_SIZE, NUM_OUTPUT_BUCKETS * HIDDEN2_SIZE);
            let l3 = builder.new_affine("l3", HIDDEN2_SIZE, NUM_OUTPUT_BUCKETS);

            let ft = |input, start, end| l0.slice(start, end).forward(input).crelu();
            let stm_hidden = ft(stm_inputs, 0, ACCUM_SIZE / 2) * ft(stm_inputs, ACCUM_SIZE / 2, ACCUM_SIZE);
            let ntm_hidden = ft(ntm_inputs, 0, ACCUM_SIZE / 2) * ft(ntm_inputs, ACCUM_SIZE / 2, ACCUM_SIZE);

            let hl1 = stm_hidden.concat(ntm_hidden);
            let hl2 = l1.forward(hl1).select(output_buckets).screlu();
            let hl3 = l2.forward(hl2).select(output_buckets).screlu();
            l3.forward(hl3).select(output_buckets)
        });

    trainer.load_from_checkpoint(checkpoint_dir);
    println!("Successfully loaded Bullet checkpoint into GPU!");

    let compare_bin = Path::new("checkpoints/compare_tmp.bin");
    let quantised_path = Path::new(checkpoint_dir).join("quantised.bin");
    let cpu_weights = if quantised_path.exists() {
        export::export_from_raw_checkpoint(&quantised_path, compare_bin)
            .expect("Failed to export quantised.bin for comparison");
        let w = eval::NnueWeights::load_from_file(compare_bin).expect("Failed to load exported weights");
        let _ = fs::remove_file(compare_bin);
        w
    } else {
        eval::NnueWeights::load_from_file(Path::new(CHAMPION_FILE)).expect("Failed to load exported weights")
    };

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
        "spsa" => spsa::run_spsa_cli(&args),
        "check" => run_check(&args),
        "eval" => run_eval(&args),
        "export" => run_export(&args),
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
