mod inputs;

mod export;

mod convert;

mod eval;

mod diagnostics;



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



use inputs::KittyHalfKA;



const INPUT_SIZE: usize = 49_152;

const ACCUM_SIZE: usize = 256;

const HIDDEN1_SIZE: usize = 32;

const HIDDEN2_SIZE: usize = 32;

const SCALE: f32 = 400.0;

const DEFAULT_CHUNK_SIZE: usize = 50_000_000;



fn print_usage() {

    println!("KittyEngineTrainer - High-Performance NNUE Trainer via Bullet");

    println!("============================================================");

    println!("Architecture: (49152 -> 256) x 2 -> 32 -> 32 -> 1 (KittyEngine Layout)");

    println!();

    println!("USAGE:");

    println!("  cargo run --release -- <COMMAND> [OPTIONS...]");

    println!();

    println!("COMMANDS:");

    println!("  train <data_dir_or_file> [OPTIONS]");

    println!("      Trains across the entire dataset with multi-pack interleaved stream (Strategy 1),");

    println!("      running an automated evaluation across ALL packs after every superbatch. Automatically");

    println!("      promotes qualifying weights to Desktop and KittyEngine, and purges outdated checkpoints.");

    println!("      Options:");

    println!("        --weights, --resume <path>  Start from weights.bin or checkpoint folder");



    println!("        --start-superbatch <N>      Starting superbatch index (default: 1)");

    println!("        --superbatches <N>          End superbatch index (default: 36)");

    println!("        --batches-per-superbatch <N> Batches per superbatch (default: total_batches/9 = 2823)");

    println!("        --batch-size <N>            Batch size in positions (default: 16384)");

    println!("        --lr <val>                  Peak learning rate (default: 0.000015 with Cosine Annealing decay)");

    println!("        --min-lr <val>              Minimum learning rate at cycle end (default: 0.2 * lr, e.g. 0.000003)");

    println!("        --eval-per-pack <N>         Validation positions per pack during eval (default: 2,000,000)");

    println!("        --wdl <val>                 WDL score distillation lambda [0.0..1.0] (default: 0.20)");

    println!();

    println!("  eval <weights.bin> [data_dir_or_file] [positions_per_pack]");

    println!("      Streams evaluation across all dataset packs and outputs the 6 core metrics:");

    println!("      Overall MAE, Overall SMSE, MSE, RMSE, Non-mate MAE/SMSE, and Mate MAE/SMSE.");

    println!("      Default positions per pack: 2,000,000 (18M positions across 9 packs).");

    println!();

    println!("  dead-neurons [weights.bin] [data_dir] [positions_per_pack]");

    println!("      Performs deep empirical activation telemetry across Layer 0 (Accumulator),");

    println!("      Layer 1 (Hidden 1), and Layer 2 (Hidden 2) to detect dead or saturated neurons");

    println!("      and verify bidirectional gradient flow.");

    println!();

    println!("  inversions [weights.bin] [data_dir] [positions_per_pack]");

    println!("      Scans dataset positions to detect complete polar inversions where the model");

    println!("      evaluates opposite to the ground truth at high centipawn thresholds.");

    println!();

    println!("  compare [checkpoint_dir]");

    println!("      Validates numerical parity between Bullet's GPU forward pass and the CPU");

    println!("      NNUE evaluation on standard benchmark FEN positions.");

    println!();

    println!("  info");

    println!("      Displays the neural network architecture dimensions and layout specifications.");

    println!();

    println!("  import <input_weights.bin> <output_bullet.bin>");

    println!("      Converts KittyEngine exact NNUESGD binary weights into Bullet format.");

    println!();

    println!("  export <checkpoints/raw.bin> <output_weights.bin>");

    println!("      Converts a raw Bullet checkpoint directory or raw.bin into KittyEngine format.");

    println!();

    println!("  pack-chunks <input_eval.json> <output_dir> [positions_per_chunk]");

    println!("      Converts large JSON evaluation files into memory-mapped binary packs (default: 50M each).");

    println!();

    println!("  help, --help, -h");

    println!("      Displays this help and usage menu.");

}



fn run_info() {

    println!("KittyEngine NNUE Layout Specification:");

    println!("  Input Feature Space:   {} (HalfKA, 64 king squares x 768 piece planes)", INPUT_SIZE);

    println!("  Dual Accumulator:      {} per perspective (512 combined)", ACCUM_SIZE);

    println!("  Hidden Layer 1:        512 -> {} (Dense + Activation)", HIDDEN1_SIZE);

    println!("  Hidden Layer 2:        32 -> {} (Dense + Activation)", HIDDEN2_SIZE);

    println!("  Output Layer:          32 -> 1 (Scalar evaluation score)");

    println!("  Centipawn Scale:       {:.1} cp", SCALE);

    println!("  Target Weight Size:    50,402,756 bytes (NNUESGD5 format)");

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

    if args.len() < 4 {

        println!("Error: missing arguments for export command.");

        println!("Usage: cargo run --release -- export <checkpoints/raw.bin> <output_weights.bin>");

        return;

    }



    let raw_bin_path = Path::new(&args[2]);

    let output_path = Path::new(&args[3]);

    match export::export_from_raw_checkpoint(raw_bin_path, output_path) {

        Ok(()) => {

            println!("Export completed successfully.");

        }

        Err(e) => {

            eprintln!("Failed to export weights: {}", e);

        }

    }

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



fn cleanup_outdated_weights(checkpoints_dir: &Path, keep_dir: Option<&Path>, keep_file: Option<&Path>) {

    println!("Cleaning up outdated weight data in {}...", checkpoints_dir.display());

    let mut deleted_dirs = 0;

    let mut deleted_files = 0;



    let keep_dir_canon = keep_dir.and_then(|p| fs::canonicalize(p).ok());

    let keep_file_canon = keep_file.and_then(|p| fs::canonicalize(p).ok());



    if let Ok(entries) = fs::read_dir(checkpoints_dir) {

        for entry in entries.flatten() {

            let path = entry.path();

            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();



            if path.is_dir() {

                let is_keep = if let (Some(c), Ok(p_canon)) = (&keep_dir_canon, fs::canonicalize(&path)) {

                    c == &p_canon

                } else {

                    false

                };

                if !is_keep && (name.starts_with("kitty_bullet_") || name.starts_with("checkpoint_")) {

                    if fs::remove_dir_all(&path).is_ok() {

                        deleted_dirs += 1;

                    }

                }

            } else if path.is_file() {

                let is_keep = if let (Some(c), Ok(p_canon)) = (&keep_file_canon, fs::canonicalize(&path)) {

                    c == &p_canon

                } else {

                    false

                };

                let is_protected = name == "baseline.json" || name == "init_bullet_weights.bin" || name == "champion_weights.bin";

                if !is_keep && !is_protected && name.ends_with(".bin") {

                    if fs::remove_file(&path).is_ok() {

                        deleted_files += 1;

                    }

                }

            }

        }

    }

    println!("  Cleanup complete: removed {} old checkpoint folder(s) and {} old weight file(s).", deleted_dirs, deleted_files);

}



fn run_train(args: &[String]) {

    if args.len() < 3 {

        println!("Error: missing data binpack file or directory for train command.");

        println!("Usage: cargo run --release -- train <data_dir_or_file> [--weights <weights.bin>] [--superbatches <N>] [--batch-size <N>] [--lr <val>] [--min-lr <val>] [--eval-per-pack <N>] [--wdl <val>]");

        return;

    }



    let mut target_dir_or_file = String::new();

    let mut weights_path: Option<String> = None;

    let mut superbatches: usize = 36;

    let mut start_superbatch: usize = 1;

    let mut schedule_start: Option<usize> = None;

    let mut batch_size: usize = 16_384;

    let mut user_batches_per_sb: Option<usize> = None;

    let mut user_lr: Option<f32> = None;

    let mut user_min_lr: Option<f32> = None;

    let mut wdl_val: f32 = 0.20;

    let mut eval_positions_per_pack: usize = 2_000_000;



    let mut i = 2;

    while i < args.len() {

        match args[i].as_str() {

            "--weights" | "--resume" => {

                i += 1;

                if i < args.len() {

                    weights_path = Some(args[i].clone());

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

                if target_dir_or_file.is_empty() {

                    target_dir_or_file = other.to_string();

                } else if weights_path.is_none() && (other.ends_with(".bin") || Path::new(other).is_dir()) {

                    weights_path = Some(other.to_string());

                } else if let Ok(val) = other.parse::<usize>() {

                    if superbatches == 40 {

                        superbatches = val;

                    } else {

                        batch_size = val;

                    }

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

            data_files[0].clone(), // eval_chunk_001.binpack (50M pos, standard)

            data_files[1].clone(), // eval_chunk_002.binpack (50M pos, standard)

            data_files[7].clone(), // eval_chunk_008.binpack (50M pos, heavy tactics/mates)

            data_files[2].clone(), // eval_chunk_003.binpack (50M pos, standard)

            data_files[3].clone(), // eval_chunk_004.binpack (50M pos, standard)

            data_files[8].clone(), // eval_chunk_009.binpack (16.37M pos, heavy tactics/mates)

            data_files[4].clone(), // eval_chunk_005.binpack (50M pos, standard)

            data_files[5].clone(), // eval_chunk_006.binpack (50M pos, standard)

            data_files[6].clone(), // eval_chunk_007.binpack (50M pos, standard)

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

        .build(|builder, stm_inputs, ntm_inputs| {

            let l0 = builder.new_affine("l0", INPUT_SIZE, ACCUM_SIZE);

            let l1 = builder.new_affine("l1", 2 * ACCUM_SIZE, HIDDEN1_SIZE);

            let l2 = builder.new_affine("l2", HIDDEN1_SIZE, HIDDEN2_SIZE);

            let l3 = builder.new_affine("l3", HIDDEN2_SIZE, 1);



            let stm_hidden = l0.forward(stm_inputs).screlu();

            let ntm_hidden = l0.forward(ntm_inputs).screlu();

            let hidden_concat = stm_hidden.concat(ntm_hidden);



            let h1 = l1.forward(hidden_concat).screlu();

            let h2 = l2.forward(h1).screlu();

            l3.forward(h2)

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

        println!("Starting training from scratch with random weight initialization.");

    }



    trainer.optimiser.set_params(AdamWParams {

        decay: 0.01,

        beta1: 0.9,

        beta2: 0.999,

        min_weight: -1.98,

        max_weight: 1.98,

    });



    let max_lr = user_lr.unwrap_or(if weights_path.is_some() { 0.000015 } else { 0.0025 });

    let min_lr = user_min_lr.unwrap_or(max_lr * 0.2);

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

    let mut baseline_wdl: Option<f32> = None;



    if baseline_path.exists() {

        if let Ok(content) = fs::read_to_string(baseline_path) {

            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {

                baseline_mae = v.get("mae").and_then(|x| x.as_f64()).unwrap_or(f64::MAX);

                baseline_smse = v.get("smse").and_then(|x| x.as_f64()).unwrap_or(f64::MAX);

                baseline_wdl = v.get("wdl").and_then(|x| x.as_f64()).map(|x| x as f32);

            }

        }

    }



    let need_baseline_eval = !baseline_path.exists()

        || baseline_mae == f64::MAX

        || baseline_wdl.map(|w| (w - wdl_val).abs() > 1e-4).unwrap_or(true);



    if let Some(ref w_path) = weights_path {

        let p = Path::new(w_path);

        if p.exists() && p.is_file() {

            if need_baseline_eval {

                println!("\nEvaluating initial champion weights under WDL={:.2} across ALL {} packs to establish baseline...", wdl_val, total_packs);

                if let Ok((b_overall, b_non_mate, b_mate)) = eval::evaluate_checkpoint_packs(p, &data_files, eval_positions_per_pack, wdl_val) {

                    baseline_mae = b_overall.mae();

                    baseline_smse = b_overall.smse();

                    let champion_bin = "checkpoints/champion_weights.bin";

                    let _ = fs::copy(p, champion_bin);

                    let initial_record = serde_json::json!({

                        "step": 0,

                        "pack": "champion_weights",

                        "epoch": 0,

                        "wdl": wdl_val,

                        "mae": baseline_mae,

                        "smse": baseline_smse,

                        "mse": b_overall.mse(),

                        "rmse": b_overall.rmse(),

                        "non_mate_mae": b_non_mate.mae(),

                        "non_mate_smse": b_non_mate.smse(),

                        "mate_mae": b_mate.mae(),

                        "mate_smse": b_mate.smse(),

                        "positions": b_overall.count,

                        "weights_file": champion_bin,

                    });

                    if let Ok(json_str) = serde_json::to_string_pretty(&initial_record) {

                        let _ = fs::write(baseline_path, json_str);

                    }

                }

            }

        }

    }



    println!("\n================================================================================");

    println!("👑 ACTIVE BASELINE CHAMPION (BENCHMARK TO BEAT BEFORE TRAINING):");

    if baseline_mae < f64::MAX && baseline_smse < f64::MAX {

        println!("  Baseline Overall MAE:  {:.7} (WDL={:.2})", baseline_mae, wdl_val);

        println!("  Baseline Overall SMSE: {:.7}", baseline_smse);

        println!("  Promotion Policy:      Normalized Composite Score");

        println!("    - Target:            0.5 * (MAE / {:.7}) + 0.5 * (SMSE / {:.7}) < 1.00000", baseline_mae, baseline_smse);

        println!("    - Safety bounds:     MAE <= {:.7} (+1.5%), SMSE <= {:.7} (+1.5%)", baseline_mae * 1.015, baseline_smse * 1.015);

    } else {

        println!("  Baseline: None established yet (first pack will set initial baseline)");

    }

    println!("================================================================================");



    let pack_strs: Vec<String> = train_files.iter().map(|p| p.to_string_lossy().to_string()).collect();

    let pack_refs: Vec<&str> = pack_strs.iter().map(|s| s.as_str()).collect();

    let data_loader = loader::DirectSequentialDataLoader::new(&pack_refs);



    println!("\nStarting GPU training across global unified dataset with automated evaluation after every superbatch...");



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



        if Path::new(&raw_path).exists() {

            println!("\nExporting step {} checkpoint to {}...", step, exported_bin);

            if let Err(e) = export::export_from_raw_checkpoint(Path::new(&raw_path), Path::new(&exported_bin)) {

                eprintln!("Failed to export weights for step {}: {}", step, e);

                continue;

            }



            println!("\nEvaluating checkpoint across ALL {} packs ({} pos/pack)...", total_packs, eval_positions_per_pack);

            match eval::evaluate_checkpoint_packs(Path::new(&exported_bin), &data_files, eval_positions_per_pack, wdl_val) {

                Ok((overall, non_mate, mate)) => {

                    println!("\n================================================================================");

                    println!("Evaluation Results across ALL {} packs ({} positions):", total_packs, overall.count);

                    let diff_mae = if baseline_mae < f64::MAX {

                        format!(" ({:+.7} vs baseline)", overall.mae() - baseline_mae)

                    } else {

                        String::new()

                    };

                    let diff_smse = if baseline_smse < f64::MAX {

                        format!(" ({:+.7} vs baseline)", overall.smse() - baseline_smse)

                    } else {

                        String::new()

                    };

                    println!(

                        "  Overall:   MAE = {:.7}{}  |  SMSE = {:.7}{}  |  BCE = {:.7}  |  MSE = {:.7}  |  RMSE = {:.7}",

                        overall.mae(), diff_mae, overall.smse(), diff_smse, overall.bce(), overall.mse(), overall.rmse()

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



                    let (b_mae, b_smse) = if baseline_path.exists() {

                        if let Ok(content) = fs::read_to_string(baseline_path) {

                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {

                                let m = v.get("mae").and_then(|x| x.as_f64()).unwrap_or(baseline_mae);

                                let s = v.get("smse").and_then(|x| x.as_f64()).unwrap_or(baseline_smse);

                                (m, s)

                            } else {

                                (baseline_mae, baseline_smse)

                            }

                        } else {

                            (baseline_mae, baseline_smse)

                        }

                    } else {

                        (baseline_mae, baseline_smse)

                    };

                    baseline_mae = b_mae;

                    baseline_smse = b_smse;



                    let composite = if baseline_mae < f64::MAX && baseline_smse < f64::MAX {

                        0.5 * (overall.mae() / baseline_mae) + 0.5 * (overall.smse() / baseline_smse)

                    } else {

                        0.0

                    };



                    let qualifies = if baseline_smse == f64::MAX || baseline_mae == f64::MAX {

                        true

                    } else {

                        // Composite improvement, with a safety cap preventing >1.5% degradation in either metric

                        composite < 1.0 && overall.mae() <= baseline_mae * 1.015 && overall.smse() <= baseline_smse * 1.015

                    };



                    if qualifies {

                        println!("\n🏆 NEW ALL-TIME CHAMPION PROMOTED TO BASELINE!");

                        if baseline_smse == f64::MAX || baseline_mae == f64::MAX {

                            println!("  Reason: Initial baseline established.");

                        } else {

                            let mae_pct = ((overall.mae() / baseline_mae) - 1.0) * 100.0;

                            let smse_pct = ((overall.smse() / baseline_smse) - 1.0) * 100.0;

                            let comp_pct = (composite - 1.0) * 100.0;

                            println!("  Reason: Normalized Composite Score improved to {:.5} ({:+.2}% net improvement)", composite, comp_pct);

                            println!("    - MAE:  {:.7} -> {:.7} ({:+.2}%)", baseline_mae, overall.mae(), mae_pct);

                            println!("    - SMSE: {:.7} -> {:.7} ({:+.2}%)", baseline_smse, overall.smse(), smse_pct);

                            println!("    - BCE:  {:.7} (Validation Log-Loss)", overall.bce());

                        }

                        println!("  Previous Baseline: MAE={:.7}, SMSE={:.7}", baseline_mae, baseline_smse);

                        println!("  New Baseline:      MAE={:.7}, SMSE={:.7}", overall.mae(), overall.smse());

                        baseline_mae = overall.mae();

                        baseline_smse = overall.smse();



                        let champion_bin = "checkpoints/champion_weights.bin";

                        let _ = fs::copy(&exported_bin, champion_bin);



                        let new_record = serde_json::json!({

                            "step": step,

                            "pack": format!("unified_epoch_{}_step_{}", epoch, step),

                            "epoch": epoch,

                            "wdl": wdl_val,

                            "composite": composite,

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

                            "weights_file": champion_bin,

                        });

                        if let Ok(json_str) = serde_json::to_string_pretty(&new_record) {

                            let _ = fs::write(baseline_path, json_str);

                            println!("  Updated baseline record: {}", baseline_path.display());

                        }



                        let targets = [

                            "C:\\Users\\evanh\\Desktop\\weights.bin",

                            "C:\\Users\\evanh\\Desktop\\KittyEngineV5\\KittyEngineV5\\weights.bin",

                        ];

                        for t in targets {

                            if let Err(e) = std::fs::copy(&exported_bin, t) {

                                eprintln!("Failed to copy to {}: {}", t, e);

                            } else {

                                println!("  Successfully updated: {}", t);

                            }

                        }



                        // User directive: "Once promote, always clean up the outdaetd weight data."

                        cleanup_outdated_weights(

                            Path::new("checkpoints"),

                            Some(Path::new(&checkpoint_dir)),

                            Some(Path::new(&exported_bin)),

                        );

                    } else {

                        let mae_pct = if baseline_mae < f64::MAX { ((overall.mae() / baseline_mae) - 1.0) * 100.0 } else { 0.0 };

                        let smse_pct = if baseline_smse < f64::MAX { ((overall.smse() / baseline_smse) - 1.0) * 100.0 } else { 0.0 };

                        println!("\nDid not beat active baseline champion under Normalized Composite Promotion Rule.");

                        println!("  Composite Score: {:.5} (Target: < 1.00000)", composite);

                        println!("  Current:   MAE={:.7} ({:+.2}%), SMSE={:.7} ({:+.2}%), BCE={:.7}", overall.mae(), mae_pct, overall.smse(), smse_pct, overall.bce());

                        println!("  Baseline:  MAE={:.7}, SMSE={:.7}", baseline_mae, baseline_smse);

                        // Clean up un-promoted checkpoint folder and rejected bin file

                        let _ = fs::remove_dir_all(&checkpoint_dir);

                        let is_baseline = fs::read_to_string(baseline_path)

                            .map(|c| c.contains(&exported_bin))

                            .unwrap_or(false);

                        if !is_baseline {

                            let _ = fs::remove_file(&exported_bin);

                        }

                    }

                }

                Err(e) => {

                    eprintln!("Evaluation failed for step {}: {}", step, e);

                }

            }

        }

    }

}



fn run_eval(args: &[String]) {

    if args.len() < 3 {


        return;

    }



    let weights_path = Path::new(&args[2]);

    let default_data = PathBuf::from("data");

    let target_data = args.get(3).map(Path::new).unwrap_or(&default_data);

    let positions_per_pack: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2_000_000);



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

    println!("  Activation:          SCReLU (NNUESGD5)");

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

                overall.mae(),

                overall.smse(),

                overall.mse(),

                overall.rmse()

            );

            println!(

                "  Non-mate:  MAE = {:.7}  |  SMSE = {:.7}  |  MSE = {:.7}  ({} pos)",

                non_mate.mae(),

                non_mate.smse(),

                non_mate.mse(),

                non_mate.count

            );

            println!(

                "  Mate:      MAE = {:.7}  |  SMSE = {:.7}  |  MSE = {:.7}  ({} pos)",

                mate.mae(),

                mate.smse(),

                mate.mse(),

                mate.count

            );

            println!("================================================================================");

        }

        Err(e) => {

            eprintln!("Evaluation failed: {}", e);

        }

    }

}



fn run_inversions(args: &[String]) {

    let default_weights = PathBuf::from("C:\\Users\\evanh\\Desktop\\weights.bin");

    let weights_path = args.get(2).map(Path::new).unwrap_or(&default_weights);

    let default_data = PathBuf::from("data");

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

    let default_weights = PathBuf::from("C:\\Users\\evanh\\Desktop\\weights.bin");

    let weights_path = args.get(2).map(Path::new).unwrap_or(&default_weights);

    let default_data = PathBuf::from("data");

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

        "Neuron", "Firing Rate", "Mean Act", "Pre-Activation [Min, Max]", "Bias", "In-Norm", "Out-Weight");

    println!("{:-<6}-+-{:-<12}-+-{:-<12}-+-{:-<22}-+-{:-<10}-+-{:-<10}-+-{:-<10}", "", "", "", "", "", "", "");



    let mut l2_dead = Vec::new();

    for j in 0..32 {

        let n = &diag.l2.neurons[j];

        let rate = n.firing_rate();



        let in_norm: f32 = weights.l2w[j * 32..(j + 1) * 32].iter().map(|w| w * w).sum::<f32>().sqrt();

        let out_w = weights.l3w[j];



        let is_dead = n.fires == 0 || in_norm < 1e-7 || out_w.abs() < 1e-7;

        if is_dead {

            l2_dead.push(j);

        }



        let dead_flag = if is_dead { " ❌ DEAD" } else { "" };

        println!("{:<6} | {:>10.2}% | {:>12.6} | [{:>9.4}, {:>9.4}] | {:>10.4} | {:>10.4} | {:>10.4}{}",

            format!("#{}", j), rate, n.mean_activation(), n.min_pre, n.max_pre, weights.l2b[j], in_norm, out_w, dead_flag);

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

        println!("  - Output Layer (L3):     Scalar bias = {:.4}", weights.l3b);

    } else {

        println!("  ⚠️ FOUND {} DEAD NEURON(S):", total_dead);

        println!("  - Layer 0: {} dead", accum_dead.len());

        println!("  - Layer 1: {} dead ({:?})", l1_dead.len(), l1_dead);

        println!("  - Layer 2: {} dead ({:?})", l2_dead.len(), l2_dead);

    }

    println!("================================================================================\n");

}



fn run_compare(args: &[String]) {

    let checkpoint_dir = args.get(2).map(|s| s.as_str()).unwrap_or("checkpoints/kitty_bullet_screlu-40");

    println!("Loading Bullet model from {}...", checkpoint_dir);



    let mut trainer = ValueTrainerBuilder::default()

        .dual_perspective()

        .optimiser(AdamW)

        .inputs(KittyHalfKA)

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

        .build(|builder, stm_inputs, ntm_inputs| {

            let l0 = builder.new_affine("l0", INPUT_SIZE, ACCUM_SIZE);

            let l1 = builder.new_affine("l1", 2 * ACCUM_SIZE, HIDDEN1_SIZE);

            let l2 = builder.new_affine("l2", HIDDEN1_SIZE, HIDDEN2_SIZE);

            let l3 = builder.new_affine("l3", HIDDEN2_SIZE, 1);



            let stm_hidden = l0.forward(stm_inputs).screlu();

            let ntm_hidden = l0.forward(ntm_inputs).screlu();

            let hidden_concat = stm_hidden.concat(ntm_hidden);



            let h1 = l1.forward(hidden_concat).screlu();

            let h2 = l2.forward(h1).screlu();

            l3.forward(h2)

        });



    trainer.load_from_checkpoint(checkpoint_dir);

    println!("Successfully loaded Bullet checkpoint into GPU!");



    let exported_path = Path::new("checkpoints/weights_screlu_scratch_pass40.bin");

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

        "pack-chunks" | "convert" => run_pack_chunks(&args),

        "train" => run_train(&args),

        "import" => run_import(&args),

        "export" => run_export(&args),

        "eval" => run_eval(&args),

        "inversions" | "eval-inversions" | "outliers" => run_inversions(&args),

        "dead-neurons" | "dead" | "check-neurons" => run_dead_neurons(&args),

        "compare" => run_compare(&args),

        "info" => run_info(),

        "help" | "--help" | "-h" => print_usage(),

        other => {

            if other != "help" {

                eprintln!("Unknown command: '{}'\n", other);

            }

            print_usage();

        }

    }

}

