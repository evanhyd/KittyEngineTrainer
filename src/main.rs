mod inputs;
mod export;
mod convert;
mod eval;
mod diagnostics;
mod config;
mod sprt;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use bullet_lib::{
    game::outputs::MaterialCount,
    nn::optimiser::{AdamW, AdamWParams},
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader, ValueTrainerBuilder},
};

use inputs::KittyHalfKA;
use config::{TrainerConfig, CHECKPOINTS_DIR, CHAMPION_FILE, DATA_DIR, SCRATCH_FILE};

const INPUT_SIZE: usize = 24_576;
const ACCUM_SIZE: usize = 256;
const HIDDEN1_SIZE: usize = 32;
const HIDDEN2_SIZE: usize = 32;
const NUM_OUTPUT_BUCKETS: usize = 8;
const SCALE: f32 = 400.0;
const DEFAULT_CHUNK_SIZE: usize = 50_000_000;

fn print_usage() {
    println!("KittyEngineTrainer - High-Performance NNUE Trainer via Bullet");
    println!("============================================================");
    println!("Architecture: (24576 -> 256) x 2 -> 32 -> 32 -> 1 x 8 (KittyEngine Layout)");
    println!();
    println!("CONFIGURATION:");
    println!("  Training & SPRT parameters are loaded from 'config.yaml'.");
    println!("  Data is loaded from 'data/' and checkpoints are saved to 'checkpoints/'.");
    println!();
    println!("CORE COMMANDS:");
    println!("  train [OPTIONS]             Start training from scratch (Option B: Positive Bias Init)");
    println!("  resume                      Resume training from the latest checkpoint in 'checkpoints/'");
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
    println!("  pack-chunks <json> <dir>    Convert evaluation JSON datasets into binary binpacks");
    println!("  import <in.bin> <out.bin>   Convert KittyEngine binary weights into Bullet format");
    println!("  help, --help, -h            Display this help and usage menu");
    println!();
}

fn run_info() {
    println!("KittyEngine NNUE Layout Specification:");
    println!("  Input Feature Space:   {} (HalfKA Folded, 32 king squares x 768 piece planes)", INPUT_SIZE);
    println!("  Dual Accumulator:      {} per perspective (512 combined)", ACCUM_SIZE);
    println!("  Hidden Layer 1:        512 -> {} (Dense + Activation)", HIDDEN1_SIZE);
    println!("  Hidden Layer 2:        32 -> {} (Dense + Activation)", HIDDEN2_SIZE);
    println!("  Output Layer:          32 -> 1 x {} Buckets (Material Count)", NUM_OUTPUT_BUCKETS);
    println!("  Centipawn Scale:       {:.1} cp", SCALE);
    println!("  Target Weight Size:    25,237,864 bytes (NNUESGD6 format)");
}

fn run_pack_chunks(args: &[String]) {
    if args.len() < 4 {
        println!("Error: missing arguments for pack-chunks command.");
        println!("Usage: cargo run --release -- pack-chunks <input_eval.json> <output_dir> [positions_per_chunk]");
        return;
    }

    let input_path = Path::new(&args[2]);
    let output_dir = Path::new(&args[3]);
    let chunk_size = args.get(4).and_then(|s| s.parse::<usize>().ok()).unwrap_or(DEFAULT_CHUNK_SIZE);

    match convert::convert_json_to_chunked_binpack(input_path, output_dir, "eval_chunk", chunk_size) {
        Ok(chunks) => {
            println!("Chunking finished successfully! Created {} chunk(s).", chunks.len());
        }
        Err(e) => {
            eprintln!("Failed to chunk dataset: {}", e);
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
            if res.decision == sprt::SprtDecision::AcceptedH1 {
                println!("Candidate statistically verified superior to Champion under SPRT.");
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
    println!("  - Time Control:        {}", cfg.sprt.time_control);
    println!("  - Concurrency:         {} threads", cfg.sprt.concurrency);
    println!("  - Max Rounds:          {}", cfg.sprt.max_rounds);
    println!("  - SPRT Bounds:         Elo [{}, {}] (alpha={}, beta={})",
        cfg.sprt.elo0, cfg.sprt.elo1, cfg.sprt.alpha, cfg.sprt.beta);
    println!();

    if baseline_path.exists() {
        if let Ok(content) = fs::read_to_string(baseline_path) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
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
        }
    } else {
        println!("👑 ACTIVE CHAMPION: None recorded in checkpoints/baseline.json");
    }

    if let Some((step, folder)) = config::detect_latest_checkpoint() {
        println!("\n📁 LATEST BULLET CHECKPOINT DETECTED:");
        println!("  - Step:             {}", step);
        println!("  - Folder:           {}", folder.display());
        println!("  - Ready to resume:  'cargo run --release -- resume'");
    }

    if history_path.exists() {
        if let Ok(content) = fs::read_to_string(history_path) {
            if let Ok(records) = serde_json::from_str::<Vec<serde_json::Value>>(&content) {
                println!("\n📜 RECENT TRAINING HISTORY (last 10 steps):");
                println!("{:<8} | {:<5} | {:<12} | {:<12} | {:<12} | {:<10} | {:<10}",
                    "Step", "Epoch", "Positions", "Overall MAE", "Overall SMSE", "Mate MAE", "Status");
                println!("{:-<8}-+-{:-<5}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<10}-+-{:-<10}", "", "", "", "", "", "", "");

                let count = records.len();
                let start = if count > 10 { count - 10 } else { 0 };
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

fn run_train(args: &[String], is_resume: bool) {
    let cfg = TrainerConfig::load_or_default();

    let mut target_dir_or_file = DATA_DIR.to_string();
    let mut weights_path: Option<String> = None;
    let mut superbatches: usize = cfg.superbatches;
    let mut start_superbatch: usize = 1;
    let mut schedule_start: Option<usize> = Some(1);
    let mut batch_size: usize = cfg.batch_size;
    let mut user_batches_per_sb: Option<usize> = None;
    let mut user_lr: Option<f32> = Some(cfg.lr);
    let mut user_min_lr: Option<f32> = Some(cfg.min_lr);
    let mut wdl_val: f32 = cfg.wdl;
    let mut eval_positions_per_pack: usize = cfg.eval_positions_per_pack;
    let save_interval: usize = cfg.save_interval.max(1);

    if is_resume {
        if let Some((step, folder)) = config::detect_latest_checkpoint() {
            println!("[resume] Detected latest checkpoint at step {} ({})", step, folder.display());
            start_superbatch = step + 1;
            schedule_start = Some(1);
            weights_path = Some(folder.to_string_lossy().to_string());
            if start_superbatch > superbatches {
                println!("Training is already complete up to target superbatches ({})!", superbatches);
                println!("To train further, increase 'superbatches' in config.yaml.");
                return;
            }
        } else {
            eprintln!("Error: No checkpoints found in '{}' to resume from.", CHECKPOINTS_DIR);
            return;
        }
    }

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" | "--resume" => {
                i += 1;
                if i < args.len() {
                    weights_path = Some(args[i].clone());
                } else if weights_path.is_none() {
                    if let Some((step, folder)) = config::detect_latest_checkpoint() {
                        println!("[resume] Detected latest checkpoint at step {} ({})", step, folder.display());
                        start_superbatch = step + 1;
                        schedule_start = Some(1);
                        weights_path = Some(folder.to_string_lossy().to_string());
                    }
                }
            }
            "--start-superbatch" => {
                i += 1;
                if i < args.len() {
                    if let Ok(v) = args[i].parse::<usize>() {
                        start_superbatch = v;
                    }
                }
            }
            "--schedule-start" | "--cycle-start" => {
                i += 1;
                if i < args.len() {
                    if let Ok(v) = args[i].parse::<usize>() {
                        schedule_start = Some(v);
                    }
                }
            }
            "--superbatches" => {
                i += 1;
                if i < args.len() {
                    if let Ok(v) = args[i].parse::<usize>() {
                        superbatches = v;
                    }
                }
            }
            "--batches-per-superbatch" => {
                i += 1;
                if i < args.len() {
                    if let Ok(v) = args[i].parse::<usize>() {
                        user_batches_per_sb = Some(v);
                    }
                }
            }
            "--batch-size" => {
                i += 1;
                if i < args.len() {
                    if let Ok(v) = args[i].parse::<usize>() {
                        batch_size = v;
                    }
                }
            }
            "--lr" => {
                i += 1;
                if i < args.len() {
                    if let Ok(v) = args[i].parse::<f32>() {
                        user_lr = Some(v);
                    }
                }
            }
            "--min-lr" => {
                i += 1;
                if i < args.len() {
                    if let Ok(v) = args[i].parse::<f32>() {
                        user_min_lr = Some(v);
                    }
                }
            }
            "--eval-per-pack" => {
                i += 1;
                if i < args.len() {
                    if let Ok(v) = args[i].parse::<usize>() {
                        eval_positions_per_pack = v;
                    }
                }
            }
            "--wdl" => {
                i += 1;
                if i < args.len() {
                    if let Ok(v) = args[i].parse::<f32>() {
                        wdl_val = v;
                    }
                }
            }
            other => {
                let p = Path::new(other);
                if p.is_dir() || p.extension().and_then(|s| s.to_str()) == Some("binpack") {
                    target_dir_or_file = other.to_string();
                } else if weights_path.is_none() && (other.ends_with(".bin") || p.is_dir()) {
                    weights_path = Some(other.to_string());
                } else if let Ok(val) = other.parse::<usize>() {
                    superbatches = val;
                }
            }
        }
        i += 1;
    }

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

    let total_packs = data_files.len();
    let mut train_files = data_files.clone();
    if train_files.len() == 9 {
        train_files = vec![
            data_files[0].clone(),
            data_files[1].clone(),
            data_files[7].clone(),
            data_files[2].clone(),
            data_files[3].clone(),
            data_files[8].clone(),
            data_files[4].clone(),
            data_files[5].clone(),
            data_files[6].clone(),
        ];
    }

    let total_positions: usize = train_files
        .iter()
        .map(|f| fs::metadata(f).map(|m| (m.len() / 32) as usize).unwrap_or(0))
        .sum();
    let total_dataset_batches = (total_positions / batch_size).max(1);
    let batches_per_superbatch = user_batches_per_sb.unwrap_or((total_dataset_batches / 9).max(1));
    let positions_per_superbatch = batches_per_superbatch * batch_size;

    println!("Initializing Bullet ValueTrainer for KittyEngine layout (Strategy 1: Global Unified Stream)...");
    println!("  Unified training dataset stream ({} files, {} positions, {} total batches):",
        train_files.len(), total_positions, total_dataset_batches);
    for (idx, f) in train_files.iter().enumerate() {
        let p_size = fs::metadata(f).map(|m| (m.len() / 32) as usize).unwrap_or(0);
        println!("    [{}] {} ({} pos)", idx + 1, f.display(), p_size);
    }
    println!("  Activation:       SCReLU");
    println!("  Batch Size:       {}", batch_size);
    println!("  Superbatch size:  {} batches ({} pos, ~{:.1} superbatches/epoch)",
        batches_per_superbatch, positions_per_superbatch, total_dataset_batches as f64 / batches_per_superbatch as f64);
    println!("  Superbatch steps: {}..{}", start_superbatch, superbatches);
    println!("  Eval per pack:    {} pos across ALL {} packs ({} total pos)",
        eval_positions_per_pack, total_packs, total_packs * eval_positions_per_pack);
    println!("  WDL lambda:       {:.2} (score distillation)", wdl_val);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(KittyHalfKA)
        .output_buckets(MaterialCount::<8>)
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
            let l1 = builder.new_affine("l1", 2 * ACCUM_SIZE, HIDDEN1_SIZE);
            let l2 = builder.new_affine("l2", HIDDEN1_SIZE, HIDDEN2_SIZE);
            let l3 = builder.new_affine("l3", HIDDEN2_SIZE, NUM_OUTPUT_BUCKETS);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_concat = stm_hidden.concat(ntm_hidden);

            let h1 = l1.forward(hidden_concat).screlu();
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
            println!("Resuming Bullet training from checkpoint directory {}...", p.display());
            trainer.load_from_checkpoint(p.to_str().unwrap());
        } else {
            let init_bullet_path = Path::new("checkpoints/init_bullet_weights.bin");
            println!("Converting KittyEngine weights from {}...", p.display());
            if let Err(e) = export::import_from_kitty_weights(p, init_bullet_path) {
                eprintln!("Failed to import KittyEngine weights: {}", e);
                return;
            }

            println!("Loading initial weights into GPU VRAM...");
            if let Err(e) = trainer.optimiser.load_weights_from_file(init_bullet_path.to_str().unwrap()) {
                eprintln!("Failed to load weights into trainer: {:?}", e);
                return;
            }
        }
        println!("Successfully loaded existing weights into trainer!");
    } else {
        println!("Option B: Initializing brand-new weights from scratch with positive bias initialization...");
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

        println!("Loading positive-bias scratch weights into GPU VRAM...");
        if let Err(e) = trainer.optimiser.load_weights_from_file(init_bullet_path.to_str().unwrap()) {
            eprintln!("Failed to load scratch weights into trainer: {:?}", e);
            return;
        }
        println!("Scratch weights successfully loaded into GPU VRAM! Positive biases active on all hidden layers.");
    }

    trainer.optimiser.set_params(AdamWParams {
        decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        min_weight: -1.98,
        max_weight: 1.98,
    });

    let max_lr = user_lr.unwrap_or(if weights_path.is_some() && !is_resume { 0.000015 } else { cfg.lr });
    let min_lr = user_min_lr.unwrap_or(cfg.min_lr);
    println!("  Learning Rate:    {} (peak) -> {} (min) [Cosine Annealing Decay]", max_lr, min_lr);

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

    if baseline_path.exists() {
        if let Ok(content) = fs::read_to_string(baseline_path) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                baseline_mae = v.get("mae").and_then(|x| x.as_f64()).unwrap_or(f64::MAX);
                baseline_smse = v.get("smse").and_then(|x| x.as_f64()).unwrap_or(f64::MAX);
            }
        }
    }

    let pack_strs: Vec<String> = train_files.iter().map(|p| p.to_string_lossy().to_string()).collect();
    let pack_refs: Vec<&str> = pack_strs.iter().map(|s| s.as_str()).collect();
    let data_loader = loader::DirectSequentialDataLoader::new(&pack_refs);

    println!("\nStarting GPU training across global unified dataset with automated SPRT verification...");

    for step in start_superbatch..=superbatches {
        let global_batch_start = (step - 1) * batches_per_superbatch;
        let epoch = global_batch_start / total_dataset_batches + 1;
        let epoch_progress_pos = (global_batch_start * batch_size) % total_positions;
        let epoch_progress_pct = (epoch_progress_pos as f64 / total_positions as f64) * 100.0;

        println!("\n================================================================================");
        println!("▶ [SUPERBATCH {}/{}] Epoch {} ({:.1}% of epoch) | {} batches ({} pos) | Global Batches {}..{}",
            step, superbatches, epoch, epoch_progress_pct, batches_per_superbatch, positions_per_superbatch,
            global_batch_start, global_batch_start + batches_per_superbatch
        );
        println!("================================================================================");

        let cycle_start = schedule_start.unwrap_or(start_superbatch);
        let progress = if superbatches > cycle_start {
            (step - cycle_start) as f32 / (superbatches - cycle_start) as f32
        } else {
            0.0
        };
        let step_lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos());
        println!("  Active Learning Rate: {:.7} (Cosine Annealing: {:.1}% progress)", step_lr, progress * 100.0);

        let schedule = TrainingSchedule {
            net_id: net_id.clone(),
            eval_scale: SCALE,
            steps: TrainingSteps {
                batch_size,
                batches_per_superbatch,
                start_superbatch: step,
                end_superbatch: step,
            },
            wdl_scheduler: wdl::ConstantWDL { value: wdl_val },
            lr_scheduler: lr::ConstantLR { value: step_lr },
            save_rate: 1,
        };

        trainer.run(&schedule, &settings, &data_loader);

        let checkpoint_dir = format!("checkpoints/{net_id}-{step}");
        let raw_path = format!("{checkpoint_dir}/raw.bin");
        let exported_bin = format!("checkpoints/weights_screlu_step{step}.bin");
        let periodic_bin = format!("checkpoints/weights_step_{step}.bin");

        if Path::new(&raw_path).exists() {
            println!("\nExporting step {} checkpoint to {}...", step, exported_bin);
            if let Err(e) = export::export_from_raw_checkpoint(Path::new(&raw_path), Path::new(&exported_bin)) {
                eprintln!("Failed to export weights for step {}: {}", step, e);
                continue;
            }

            let is_save_interval = step % save_interval == 0;
            if is_save_interval {
                let _ = fs::copy(&exported_bin, &periodic_bin);
                println!("💾 Preserved tournament checkpoint: {}", periodic_bin);
            }

            println!("\nEvaluating checkpoint across ALL {} packs ({} pos/pack)...", total_packs, eval_positions_per_pack);
            match eval::evaluate_checkpoint_packs(Path::new(&exported_bin), &data_files, eval_positions_per_pack, wdl_val) {
                Ok((overall, non_mate, mate)) => {
                    let diff_mae_str = if baseline_mae < f64::MAX {
                        format!(" ({:+.7} vs champion)", overall.mae() - baseline_mae)
                    } else {
                        String::new()
                    };
                    let diff_smse_str = if baseline_smse < f64::MAX {
                        format!(" ({:+.7} vs champion)", overall.smse() - baseline_smse)
                    } else {
                        String::new()
                    };

                    println!("\n================================================================================");
                    println!("Evaluation Results across ALL {} packs ({} positions):", total_packs, overall.count);
                    println!(
                        "  Overall:   MAE = {:.7}{}  |  SMSE = {:.7}{}  |  BCE = {:.7}  |  MSE = {:.7}  |  RMSE = {:.7}",
                        overall.mae(), diff_mae_str, overall.smse(), diff_smse_str, overall.bce(), overall.mse(), overall.rmse()
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

                    // SPRT Promotion Policy: SPRT is the EXCLUSIVE gate for champion promotion
                    let champion_file = Path::new(CHAMPION_FILE);
                    let (promoted, sprt_stats) = if !champion_file.exists() {
                        println!("\n👑 INITIAL BASELINE CHAMPION ESTABLISHED!");
                        println!("  Reason: No prior champion existed. Step {} registered as baseline champion.", step);
                        let _ = fs::copy(&exported_bin, CHAMPION_FILE);
                        baseline_mae = overall.mae();
                        baseline_smse = overall.smse();
                        let base_entry = format!(
                            "[{}] BASELINE CHAMPION ESTABLISHED\n  Step:    {}\n  Weights: {}\n  MAE:     {:.6} | SMSE: {:.6}\n  --------------------------------------------------------------------------------",
                            sprt::current_timestamp(), step, CHAMPION_FILE, baseline_mae, baseline_smse
                        );
                        sprt::append_to_match_log(&base_entry);
                        (true, None)
                    } else if cfg.sprt.enabled {
                        println!("\n⚔️ RUNNING AUTOMATED SPRT PROMOTION MATCH (Candidate Step {} vs Active Champion)...", step);
                        match sprt::run_sprt_match(&cfg.sprt, Path::new(&exported_bin), champion_file) {
                            Ok(res) => {
                                let passed = res.decision == sprt::SprtDecision::AcceptedH1;
                                if passed {
                                    println!("\n🏆 NEW ALL-TIME CHAMPION PROMOTED VIA SPRT!");
                                    println!("  Candidate defeated Champion: +{} -{} ={} (Elo {:+.1} +/- {:.1})",
                                        res.wins, res.losses, res.draws,
                                        res.elo_diff.unwrap_or(0.0), res.elo_error.unwrap_or(0.0));
                                    let _ = fs::copy(&exported_bin, CHAMPION_FILE);
                                    baseline_mae = overall.mae();
                                    baseline_smse = overall.smse();
                                    let promo_entry = format!(
                                        "[{}] 👑 CHAMPION PROMOTED\n  Step:     {}\n  Score:    +{} -{} ={} | Elo: {:+.1} +/- {:.1}\n  Promoted: {}\n  --------------------------------------------------------------------------------",
                                        sprt::current_timestamp(), step, res.wins, res.losses, res.draws,
                                        res.elo_diff.unwrap_or(0.0), res.elo_error.unwrap_or(0.0),
                                        CHAMPION_FILE
                                    );
                                    sprt::append_to_match_log(&promo_entry);
                                } else {
                                    println!("\n❌ Step {} did not beat active Champion under SPRT.", step);
                                    println!("  Score: Candidate +{} -{} ={}", res.wins, res.losses, res.draws);
                                    if !is_save_interval {
                                        let _ = fs::remove_file(&exported_bin);
                                    }
                                }
                                (passed, Some(res))
                            }
                            Err(e) => {
                                eprintln!("[sprt] SPRT execution error: {}", e);
                                if !is_save_interval {
                                    let _ = fs::remove_file(&exported_bin);
                                }
                                (false, None)
                            }
                        }
                    } else {
                        println!("\n⚠️ SPRT is disabled in config.yaml. Checkpoint preserved, but champion remains unchanged.");
                        println!("   (SPRT is the exclusive gate for champion promotion).");
                        (false, None)
                    };

                    if promoted {
                        let mut rec = serde_json::json!({
                            "step": step,
                            "pack": format!("unified_epoch_{}_step_{}", epoch, step),
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
                            rec["sprt"] = serde_json::json!({
                                "decision": format!("{:?}", s.decision),
                                "wins": s.wins,
                                "losses": s.losses,
                                "draws": s.draws,
                                "total_games": s.total_games,
                                "elo_diff": s.elo_diff,
                                "elo_error": s.elo_error,
                                "llr": s.llr,
                            });
                        }
                        if let Ok(json_str) = serde_json::to_string_pretty(&rec) {
                            let _ = fs::write(baseline_path, json_str);
                            println!("  Updated baseline record: {}", baseline_path.display());
                        }
                    }

                    record_training_history(
                        step,
                        epoch,
                        global_batch_start * batch_size + positions_per_superbatch,
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
                    eprintln!("Evaluation failed for step {}: {}", step, e);
                }
            }
        }
    }
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
        if a == "--wdl" && idx + 1 < args.len() {
            if let Ok(v) = args[idx + 1].parse::<f32>() {
                wdl_val = v;
            }
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
            println!("1. COMPLETE POLAR INVERSIONS (Model and Target disagree on sign at high cp)");
            println!("   Definition: Both |Target| >= T AND |Model Eval| >= T, with OPPOSITE signs!");
            println!("--------------------------------------------------------------------------------");
            println!("  Opposite sign at >= 200 cp (±2.0 pawns): {:>8} / {:>8} ({:6.4}%)",
                stats.inv_200, stats.total, pct(stats.inv_200));
            println!("  Opposite sign at >= 300 cp (±3.0 pawns): {:>8} / {:>8} ({:6.4}%)",
                stats.inv_300, stats.total, pct(stats.inv_300));
            println!("  Opposite sign at >= 500 cp (±5.0 pawns): {:>8} / {:>8} ({:6.4}%)",
                stats.inv_500, stats.total, pct(stats.inv_500));
            println!("  Opposite sign at >= 800 cp (±8.0 pawns): {:>8} / {:>8} ({:6.4}%)",
                stats.inv_800, stats.total, pct(stats.inv_800));
            println!("  Mate Inversions (|Target|>=2360, |Eval|>=200): {:>6} / {:>8} ({:6.4}%)",
                stats.inv_mate, stats.total, pct(stats.inv_mate));

            println!("\n--------------------------------------------------------------------------------");
            println!("2. ONE-WAY POLARITY DISAGREEMENTS (Decisive positions misjudged across 0.0 cp)");
            println!("--------------------------------------------------------------------------------");
            let pct_300_win = (stats.win_pred_loss_300 as f64 / stats.count_target_ge_300.max(1) as f64) * 100.0;
            let pct_300_loss = (stats.loss_pred_win_300 as f64 / stats.count_target_ge_300.max(1) as f64) * 100.0;
            println!("  Target >= +300 cp, but Model < 0 cp (False Loss): {:>8} ({:5.2}% of decisive +300)",
                stats.win_pred_loss_300, pct_300_win);
            println!("  Target <= -300 cp, but Model > 0 cp (False Win):  {:>8} ({:5.2}% of decisive -300)",
                stats.loss_pred_win_300, pct_300_loss);
            println!("  Target >= +500 cp, but Model < 0 cp (False Loss): {:>8}", stats.win_pred_loss_500);
            println!("  Target <= -500 cp, but Model > 0 cp (False Win):  {:>8}", stats.loss_pred_win_500);

            println!("\n--------------------------------------------------------------------------------");
            println!("3. ABSOLUTE CENTIPAWN ERROR SPREAD");
            println!("--------------------------------------------------------------------------------");
            println!("  Error >=  300 cp: {:>8} ({:5.2}%)", stats.err_ge_300, pct(stats.err_ge_300));
            println!("  Error >=  500 cp: {:>8} ({:5.2}%)", stats.err_ge_500, pct(stats.err_ge_500));
            println!("  Error >=  800 cp: {:>8} ({:5.2}%)", stats.err_ge_800, pct(stats.err_ge_800));
            println!("  Error >= 1200 cp: {:>8} ({:5.2}%)", stats.err_ge_1200, pct(stats.err_ge_1200));

            if !stats.worst_inversions.is_empty() {
                println!("\n--------------------------------------------------------------------------------");
                println!("4. TOP EXTREME INVERSION SAMPLES (|Target|>=300, |Eval|>=300, Opposite Sign)");
                println!("--------------------------------------------------------------------------------");
                for (idx, (tgt, pred, err, desc)) in stats.worst_inversions.iter().enumerate() {
                    println!("  #{}: Target = {:+7.1} cp | Model = {:+7.1} cp | Delta = {:6.1} cp",
                        idx + 1, tgt, pred, err);
                    println!("      Board: {}", desc);
                }
            }
            println!("================================================================================");
        }
        Err(e) => {
            eprintln!("Inversion analysis failed: {}", e);
        }
    }
}

fn run_dead_neurons(args: &[String]) {
    let default_weights = PathBuf::from(CHAMPION_FILE);
    let weights_path = args.get(2).map(Path::new).unwrap_or(&default_weights);
    let default_data = PathBuf::from(DATA_DIR);
    let target_data = args.get(3).map(Path::new).unwrap_or(&default_data);
    let positions_per_pack: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(50_000);

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
    println!("--------------------------------------------------------------------------------");
    println!("Layer 0: Accumulator (Feature Transformer -> 256 Neurons per Perspective)");
    println!("--------------------------------------------------------------------------------");
    let mut accum_dead = Vec::new();
    let mut accum_low = Vec::new();
    let mut accum_rates = Vec::new();

    for i in 0..256 {
        let comb = &diag.accum_combined.neurons[i];
        let rate = comb.firing_rate();
        accum_rates.push(rate);

        let mut out_norm_sq = 0.0f32;
        for h in 0..32 {
            let w_stm = weights.l1w[h * 512 + i];
            let w_ntm = weights.l1w[h * 512 + 256 + i];
            out_norm_sq += w_stm * w_stm + w_ntm * w_ntm;
        }
        let out_norm = out_norm_sq.sqrt();

        if comb.fires == 0 || out_norm < 1e-7 {
            accum_dead.push((i, rate, comb.min_pre, comb.max_pre, weights.l0b[i], out_norm));
        } else if rate < 1.0 {
            accum_low.push((i, rate, comb.min_pre, comb.max_pre, weights.l0b[i], out_norm));
        }
    }

    accum_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let accum_avg_rate: f64 = accum_rates.iter().sum::<f64>() / 256.0;
    let accum_median_rate = accum_rates[128];
    let accum_min_rate = accum_rates[0];
    let accum_max_rate = accum_rates[255];

    println!("  Total Accumulator Neurons: 256");
    println!("  Dead Neurons (0.0% firing or 0 outgoing weight): {}", accum_dead.len());
    println!("  Low Activity Neurons (< 1.0% firing):             {}", accum_low.len());
    println!("  Firing Rate Stats: Min = {:.2}%, Median = {:.2}%, Mean = {:.2}%, Max = {:.2}%",
        accum_min_rate, accum_median_rate, accum_avg_rate, accum_max_rate);

    if !accum_dead.is_empty() {
        println!("\n  ⚠️ DEAD ACCUMULATOR NEURONS DETECTED:");
        for (idx, rate, min_p, max_p, bias, out_norm) in &accum_dead {
            println!("    - Neuron #{:3}: Firing={:.2}%, Pre-Act=[{:.4}, {:.4}], Bias={:.4}, Outgoing Norm={:.6}",
                idx, rate, min_p, max_p, bias, out_norm);
        }
    } else {
        println!("  ✅ ZERO Dead Accumulator Neurons: All 256 accumulator neurons actively fire!");
    }

    // 2. Hidden Layer 1 (32 neurons)
    println!("\n--------------------------------------------------------------------------------");
    println!("Layer 1: Hidden Dense Layer 1 (512 -> 32 Neurons)");
    println!("--------------------------------------------------------------------------------");
    println!("{:<6} | {:<12} | {:<12} | {:<22} | {:<10} | {:<10} | {:<10}",
        "Neuron", "Firing Rate", "Mean Act", "Pre-Activation [Min, Max]", "Bias", "In-Norm", "Out-Norm");
    println!("{:-<6}-+-{:-<12}-+-{:-<12}-+-{:-<22}-+-{:-<10}-+-{:-<10}-+-{:-<10}", "", "", "", "", "", "", "");

    let mut l1_dead = Vec::new();
    for i in 0..32 {
        let n = &diag.l1.neurons[i];
        let rate = n.firing_rate();

        let in_norm: f32 = weights.l1w[i * 512..(i + 1) * 512].iter().map(|w| w * w).sum::<f32>().sqrt();
        let out_norm: f32 = (0..32).map(|j| {
            let w = weights.l2w[j * 32 + i];
            w * w
        }).sum::<f32>().sqrt();

        let is_dead = n.fires == 0 || in_norm < 1e-7 || out_norm < 1e-7;
        if is_dead {
            l1_dead.push(i);
        }

        let dead_flag = if is_dead { " ❌ DEAD" } else { "" };
        println!("{:<6} | {:>10.2}% | {:>12.6} | [{:>9.4}, {:>9.4}] | {:>10.4} | {:>10.4} | {:>10.4}{}",
            format!("#{}", i), rate, n.mean_activation(), n.min_pre, n.max_pre, weights.l1b[i], in_norm, out_norm, dead_flag);
    }
    if l1_dead.is_empty() {
        println!("  ✅ ZERO Dead Neurons in Layer 1: All 32 neurons active and propagating gradients!");
    } else {
        println!("  ⚠️ DEAD NEURONS IN LAYER 1: {:?}", l1_dead);
    }

    // 3. Hidden Layer 2 (32 neurons)
    println!("\n--------------------------------------------------------------------------------");
    println!("Layer 2: Hidden Dense Layer 2 (32 -> 32 Neurons)");
    println!("--------------------------------------------------------------------------------");
    println!("{:<6} | {:<12} | {:<12} | {:<22} | {:<10} | {:<10} | {:<10}",
        "Neuron", "Firing Rate", "Mean Act", "Pre-Activation [Min, Max]", "Bias", "In-Norm", "Out-Norm");
    println!("{:-<6}-+-{:-<12}-+-{:-<12}-+-{:-<22}-+-{:-<10}-+-{:-<10}-+-{:-<10}", "", "", "", "", "", "", "");

    let mut l2_dead = Vec::new();
    for j in 0..32 {
        let n = &diag.l2.neurons[j];
        let rate = n.firing_rate();

        let in_norm: f32 = weights.l2w[j * 32..(j + 1) * 32].iter().map(|w| w * w).sum::<f32>().sqrt();
        let out_norm: f32 = (0..8).map(|b| {
            let w = weights.l3w[b * 32 + j];
            w * w
        }).sum::<f32>().sqrt();

        let is_dead = n.fires == 0 || in_norm < 1e-7 || out_norm < 1e-7;
        if is_dead {
            l2_dead.push(j);
        }

        let dead_flag = if is_dead { " ❌ DEAD" } else { "" };
        println!("{:<6} | {:>10.2}% | {:>12.6} | [{:>9.4}, {:>9.4}] | {:>10.4} | {:>10.4} | {:>10.4}{}",
            format!("#{}", j), rate, n.mean_activation(), n.min_pre, n.max_pre, weights.l2b[j], in_norm, out_norm, dead_flag);
    }
    if l2_dead.is_empty() {
        println!("  ✅ ZERO Dead Neurons in Layer 2: All 32 neurons active and contributing to output!");
    } else {
        println!("  ⚠️ DEAD NEURONS IN LAYER 2: {:?}", l2_dead);
    }

    // Summary verdict
    println!("\n================================================================================");
    println!("FINAL VERDICT:");
    let total_dead = accum_dead.len() + l1_dead.len() + l2_dead.len();
    if total_dead == 0 {
        println!("  🎉 NO DEAD NEURONS DETECTED ACROSS THE ENTIRE NETWORK!");
        println!("  - Layer 0 (Accumulator): 256/256 active (100.0% utilization)");
        println!("  - Layer 1 (Hidden 1):    32/32 active (100.0% utilization)");
        println!("  - Layer 2 (Hidden 2):    32/32 active (100.0% utilization)");
        println!("  - Output Layer (L3):     8 material buckets");
    } else {
        println!("  ⚠️ FOUND {} DEAD NEURON(S):", total_dead);
        println!("  - Layer 0: {} dead", accum_dead.len());
        println!("  - Layer 1: {} dead ({:?})", l1_dead.len(), l1_dead);
        println!("  - Layer 2: {} dead ({:?})", l2_dead.len(), l2_dead);
    }
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
        .inputs(KittyHalfKA)
        .output_buckets(MaterialCount::<8>)
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
            let l1 = builder.new_affine("l1", 2 * ACCUM_SIZE, HIDDEN1_SIZE);
            let l2 = builder.new_affine("l2", HIDDEN1_SIZE, HIDDEN2_SIZE);
            let l3 = builder.new_affine("l3", HIDDEN2_SIZE, NUM_OUTPUT_BUCKETS);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_concat = stm_hidden.concat(ntm_hidden);

            let h1 = l1.forward(hidden_concat).screlu();
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
        "train" => run_train(&args, false),
        "resume" => run_train(&args, true),
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
        "pack-chunks" => run_pack_chunks(&args),
        "help" | "--help" | "-h" => print_usage(),
        other => {
            if other != "help" {
                eprintln!("Unknown command: '{}'\n", other);
            }
            print_usage();
        }
    }
}
