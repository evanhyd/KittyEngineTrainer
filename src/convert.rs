use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;
use rayon::prelude::*;
use serde::Deserialize;
use bulletformat::ChessBoard;

#[derive(Deserialize)]
struct EvalRecord {
    fen: String,
    evals: Vec<EvalEntry>,
}

#[derive(Deserialize)]
struct EvalEntry {
    pvs: Vec<PvEntry>,
}

#[derive(Deserialize)]
struct PvEntry {
    cp: Option<i32>,
    mate: Option<i32>,
}

fn parse_line_to_board(line: &str) -> Option<ChessBoard> {
    if line.trim().is_empty() {
        return None;
    }

    let record: EvalRecord = serde_json::from_str(line).ok()?;
    let first_pv = record.evals.first()?.pvs.first()?;

    // Extract score in centipawns (applying Pass 11 mating policy)
    let cp_score = if let Some(cp) = first_pv.cp {
        cp.clamp(-3000, 3000)
    } else if let Some(mate) = first_pv.mate {
        let d = mate.abs().min(290);
        if mate > 0 {
            (3000 - 10 * d).max(0)
        } else {
            (-3000 + 10 * d).min(0)
        }
    } else {
        return None;
    };

    // Approximate WDL game result from centipawns (realistic chess thresholds: >= +100 cp -> Win, <= -100 cp -> Loss, else Draw)
    let wdl_str = if cp_score >= 100 {
        "1.0"
    } else if cp_score <= -100 {
        "0.0"
    } else {
        "0.5"
    };

    let entry_str = format!("{} | {} | {}", record.fen, cp_score, wdl_str);
    entry_str.parse::<ChessBoard>().ok()
}

/// Converts JSON lines from eval.json into multiple 50M position binary chunks (1.6 GB each).
pub fn convert_json_to_chunked_binpack(
    json_path: &Path,
    output_dir: &Path,
    prefix: &str,
    positions_per_chunk: usize,
) -> std::io::Result<Vec<(String, usize)>> {
    fs::create_dir_all(output_dir)?;

    let input_file = File::open(json_path)?;
    let mut reader = BufReader::with_capacity(32 * 1024 * 1024, input_file);

    let mut chunk_idx = 1usize;
    let mut chunk_count = 0usize;
    let mut total_count = 0usize;
    let mut completed_chunks = Vec::new();

    let mut chunk_file_path = output_dir.join(format!("{}_{:03}.binpack", prefix, chunk_idx));
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, File::create(&chunk_file_path)?);

    println!("Starting multi-threaded chunked conversion from: {}", json_path.display());
    println!("Output directory:     {}", output_dir.display());
    println!("Chunk size:           {} positions (~{:.2} GB per chunk)", 
        positions_per_chunk, (positions_per_chunk * 32) as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("Active chunk:         {}", chunk_file_path.display());

    let start_time = Instant::now();
    let mut last_log_time = Instant::now();
    let batch_read_size = 65_536; // Read 64K lines at a time for parallel parsing
    let mut line_buffer = Vec::with_capacity(batch_read_size);

    loop {
        line_buffer.clear();
        for _ in 0..batch_read_size {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            line_buffer.push(line);
        }

        if line_buffer.is_empty() {
            break;
        }

        // Parallel parse across all CPU threads via Rayon
        let boards: Vec<ChessBoard> = line_buffer
            .par_iter()
            .filter_map(|l| parse_line_to_board(l))
            .collect();

        for board in boards {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &board as *const ChessBoard as *const u8,
                    std::mem::size_of::<ChessBoard>(),
                )
            };
            writer.write_all(bytes)?;
            chunk_count += 1;
            total_count += 1;

            if chunk_count >= positions_per_chunk {
                writer.flush()?;
                let chunk_name = chunk_file_path.file_name().unwrap().to_string_lossy().to_string();
                let chunk_size_mb = (chunk_count * 32) as f64 / (1024.0 * 1024.0);
                println!(
                    "Finished chunk #{}: '{}' ({} positions, {:.2} MB)",
                    chunk_idx, chunk_name, chunk_count, chunk_size_mb
                );
                completed_chunks.push((chunk_name, chunk_count));

                chunk_idx += 1;
                chunk_count = 0;
                chunk_file_path = output_dir.join(format!("{}_{:03}.binpack", prefix, chunk_idx));
                writer = BufWriter::with_capacity(16 * 1024 * 1024, File::create(&chunk_file_path)?);
                println!("Opened next chunk: {}", chunk_file_path.display());
            }
        }

        if last_log_time.elapsed().as_secs() >= 5 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = total_count as f64 / elapsed;
            println!(
                "Progress: {:.2}M positions total | Chunk #{}: {:.2}M / {:.0}M | Rate: {:.0} pos/sec",
                total_count as f64 / 1_000_000.0,
                chunk_idx,
                chunk_count as f64 / 1_000_000.0,
                positions_per_chunk as f64 / 1_000_000.0,
                rate
            );
            last_log_time = Instant::now();
        }
    }

    writer.flush()?;
    if chunk_count > 0 {
        let chunk_name = chunk_file_path.file_name().unwrap().to_string_lossy().to_string();
        let chunk_size_mb = (chunk_count * 32) as f64 / (1024.0 * 1024.0);
        println!(
            "Finished final chunk #{}: '{}' ({} positions, {:.2} MB)",
            chunk_idx, chunk_name, chunk_count, chunk_size_mb
        );
        completed_chunks.push((chunk_name, chunk_count));
    } else {
        // Remove empty placeholder file if loop ended exactly on boundary
        let _ = fs::remove_file(&chunk_file_path);
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    println!("\n========================================================");
    println!("All chunks successfully packed!");
    println!("  Total positions:   {}", total_count);
    println!("  Total chunks:      {}", completed_chunks.len());
    println!("  Total size:        {:.2} GB", (total_count * 32) as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("  Total elapsed:     {:.1}s ({:.0} pos/sec)", elapsed, total_count as f64 / elapsed);
    println!("========================================================");

    Ok(completed_chunks)
}

