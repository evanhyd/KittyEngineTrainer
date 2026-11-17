use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use crate::config::{SprtConfig, SPRT_GAMES_PGN, SPRT_MATCH_LOG};

/// Returns the current UTC timestamp formatted as YYYY-MM-DD HH:MM:SS UTC.
pub fn current_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let sec = now % 60;
    let min = (now / 60) % 60;
    let hour = (now / 3600) % 24;
    let days = now / 86400;

    let mut year = 1970;
    let mut d = days;
    loop {
        let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        let days_in_year = if leap { 366 } else { 365 };
        if d < days_in_year {
            let month_days = if leap {
                [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            } else {
                [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            };
            let mut month = 1;
            for &md in &month_days {
                if d < md {
                    break;
                }
                d -= md;
                month += 1;
            }
            let day = d + 1;
            return format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC", year, month, day, hour, min, sec);
        }
        d -= days_in_year;
        year += 1;
    }
}

/// Appends a log entry to the standard SPRT match audit log (`checkpoints/sprt_matches.log`).
pub fn append_to_match_log(msg: &str) {
    let log_path = Path::new(SPRT_MATCH_LOG);
    if let Some(parent) = log_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(log_path) {
        let _ = writeln!(file, "{}", msg);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SprtDecision {
    AcceptedH1,   // Candidate is statistically stronger (PROMOTE!)
    AcceptedH0,   // Candidate failed to beat Champion (REJECT)
    Inconclusive, // Max rounds reached without crossing bounds
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SprtMatchResult {
    pub decision: SprtDecision,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub total_games: usize,
    pub elo_diff: Option<f64>,
    pub elo_error: Option<f64>,
    pub llr: Option<f64>,
    pub lbound: Option<f64>,
    pub ubound: Option<f64>,
}

struct SandboxGuard {
    sandbox_dir: PathBuf,
}

impl Drop for SandboxGuard {
    fn drop(&mut self) {
        if self.sandbox_dir.exists() {
            let _ = fs::remove_dir_all(&self.sandbox_dir);
        }
    }
}

/// Discovers `search_config.yaml` near the engine binary or in known project paths.
fn locate_search_config(engine_path: &Path) -> Option<PathBuf> {
    // 1. Same directory as engine executable
    if let Some(parent) = engine_path.parent() {
        let p = parent.join("search_config.yaml");
        if p.exists() {
            return Some(p);
        }
    }

    // 2. Current working directory
    let cwd_cfg = PathBuf::from("search_config.yaml");
    if cwd_cfg.exists() {
        return Some(cwd_cfg);
    }

    // 3. Known KittyEngine repository path
    let known = PathBuf::from("C:/Users/evanh/Desktop/KittyEngineV5/x64/Release/search_config.yaml");
    if known.exists() {
        return Some(known);
    }

    None
}

/// Runs an automated SPRT match between candidate and champion weights using cutechess-cli.
pub fn run_sprt_match(
    config: &SprtConfig,
    candidate_weights: &Path,
    champion_weights: &Path,
) -> Result<SprtMatchResult, String> {
    if !config.enabled {
        return Err("SPRT verification is disabled in config.yaml".to_string());
    }

    let cutechess_path = Path::new(&config.cutechess_path);
    if !cutechess_path.exists() {
        return Err(format!("Cutechess CLI binary not found at: {}", config.cutechess_path));
    }

    let engine_path = Path::new(&config.engine_path);
    if !engine_path.exists() {
        return Err(format!("KittyEngine binary not found at: {}", config.engine_path));
    }

    if !candidate_weights.exists() {
        return Err(format!("Candidate weights file not found: {}", candidate_weights.display()));
    }

    if !champion_weights.exists() {
        return Err(format!("Champion weights file not found: {}", champion_weights.display()));
    }

    let book_path = Path::new(&config.book_path);
    if !book_path.exists() {
        return Err(format!("Opening book not found at: {}", config.book_path));
    }

    // 1. Set up sandboxes
    let sandbox_base = PathBuf::from("checkpoints/sprt_sandbox");
    let candidate_dir = sandbox_base.join("candidate");
    let champion_dir = sandbox_base.join("champion");

    if sandbox_base.exists() {
        let _ = fs::remove_dir_all(&sandbox_base);
    }
    fs::create_dir_all(&candidate_dir)
        .map_err(|e| format!("Failed to create candidate sandbox: {}", e))?;
    fs::create_dir_all(&champion_dir)
        .map_err(|e| format!("Failed to create champion sandbox: {}", e))?;

    let _guard = SandboxGuard { sandbox_dir: sandbox_base.clone() };

    // Copy weights into their respective sandboxes as 'weights.bin'
    fs::copy(candidate_weights, candidate_dir.join("weights.bin"))
        .map_err(|e| format!("Failed to stage candidate weights: {}", e))?;
    fs::copy(champion_weights, champion_dir.join("weights.bin"))
        .map_err(|e| format!("Failed to stage champion weights: {}", e))?;

    // Copy search_config.yaml into both sandboxes so the engine has valid search heuristics
    if let Some(search_cfg) = locate_search_config(engine_path) {
        let _ = fs::copy(&search_cfg, candidate_dir.join("search_config.yaml"));
        let _ = fs::copy(&search_cfg, champion_dir.join("search_config.yaml"));
    }

    println!("================================================================================");
    println!("⚔️ INITIATING AUTOMATED SPRT VERIFICATION (Cutechess-CLI)");
    println!("================================================================================");
    println!("  Candidate:      {}", candidate_weights.display());
    println!("  Champion:       {}", champion_weights.display());
    println!("  Engine Binary:  {}", config.engine_path);
    println!("  Opening Book:   {} (depth: {})", config.book_path, config.book_depth);
    println!("  Time Control:   {} | Concurrency: {} | Max Rounds: {}",
        config.time_control, config.concurrency, config.max_rounds);
    println!("  SPRT Bounds:    Elo [{}, {}] | Alpha: {} | Beta: {}",
        config.elo0, config.elo1, config.alpha, config.beta);
    println!("  Adjudication:   Draw moves >= {} (score <= {} cp), Resign moves >= {} (score <= -{} cp)",
        config.draw_movenumber, config.draw_score, config.resign_movecount, config.resign_score);
    println!("--------------------------------------------------------------------------------");

    // 2. Build cutechess command
    let mut cmd = Command::new(&config.cutechess_path);
    cmd.arg("-engine")
        .arg("name=Candidate")
        .arg(format!("cmd={}", config.engine_path))
        .arg(format!("dir={}", candidate_dir.display()))
        .arg("proto=uci")
        .arg("-engine")
        .arg("name=Champion")
        .arg(format!("cmd={}", config.engine_path))
        .arg(format!("dir={}", champion_dir.display()))
        .arg("proto=uci")
        .arg("-each")
        .arg(format!("tc={}", config.time_control))
        .arg(format!("book={}", config.book_path))
        .arg(format!("bookdepth={}", config.book_depth))
        .arg("-games")
        .arg("2")
        .arg("-rounds")
        .arg(config.max_rounds.to_string())
        .arg("-repeat")
        .arg("-concurrency")
        .arg(config.concurrency.to_string())
        .arg("-draw")
        .arg(format!("movenumber={}", config.draw_movenumber))
        .arg(format!("movecount={}", config.draw_movecount))
        .arg(format!("score={}", config.draw_score))
        .arg("-resign")
        .arg(format!("movecount={}", config.resign_movecount))
        .arg(format!("score={}", config.resign_score))
        .arg("-sprt")
        .arg(format!("elo0={}", config.elo0))
        .arg(format!("elo1={}", config.elo1))
        .arg(format!("alpha={}", config.alpha))
        .arg(format!("beta={}", config.beta));

    // Record full games to standard default PGN file
    let pgn_path = Path::new(SPRT_GAMES_PGN);
    if let Some(parent) = pgn_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    cmd.arg("-pgnout")
        .arg(pgn_path)
        .arg("fi");

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn Cutechess: {}", e))?;
    let stdout = child.stdout.take().ok_or("Failed to capture Cutechess stdout")?;
    let reader = BufReader::new(stdout);

    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut draws = 0usize;
    let mut total_games = 0usize;
    let mut elo_diff = None;
    let mut elo_error = None;
    let mut llr = None;
    let mut lbound = None;
    let mut ubound = None;
    let mut explicit_decision = None;

    for line_res in reader.lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => break,
        };

        // Parse finished game
        if line.starts_with("Finished game") {
            println!("  {}", line);
        }

        // Parse live score: "Score of Candidate vs Champion: 10 - 5 - 15  [0.583] 30"
        if line.starts_with("Score of Candidate vs Champion:") {
            let parts: Vec<&str> = line.split(':').collect();
            if parts.len() > 1 {
                let tokens: Vec<&str> = parts[1].split_whitespace().collect();
                if tokens.len() >= 5 {
                    if let (Ok(w), Ok(l), Ok(d)) = (
                        tokens[0].parse::<usize>(),
                        tokens[2].parse::<usize>(),
                        tokens[4].parse::<usize>(),
                    ) {
                        wins = w;
                        losses = l;
                        draws = d;
                        total_games = w + l + d;
                    }
                }
            }
            println!("  📊 {}", line);
        }

        // Parse Elo difference: "Elo difference: 14.5 +/- 8.2, LOS: ..."
        if line.contains("Elo difference:") {
            if let Some(pos) = line.find("Elo difference:") {
                let rest = &line[pos + 15..];
                let tokens: Vec<&str> = rest.split_whitespace().collect();
                if !tokens.is_empty() {
                    if let Ok(val) = tokens[0].parse::<f64>() {
                        elo_diff = Some(val);
                    }
                }
                if tokens.len() >= 3 && tokens[1] == "+/-" {
                    let err_str = tokens[2].trim_end_matches(',');
                    if let Ok(err) = err_str.parse::<f64>() {
                        elo_error = Some(err);
                    }
                }
            }
        }

        // Parse SPRT stats: "SPRT: llr 0.0269 (0.9%), lbound -2.94, ubound 2.94"
        if line.contains("SPRT:") {
            println!("  {}", line);
            // Parse LLR
            if let Some(llr_pos) = line.find("llr ") {
                let sub = &line[llr_pos + 4..];
                if let Some(end) = sub.find(|c: char| c == ' ' || c == '(') {
                    if let Ok(v) = sub[..end].parse::<f64>() {
                        llr = Some(v);
                    }
                }
            }
            // Parse lbound
            if let Some(lb_pos) = line.find("lbound ") {
                let sub = &line[lb_pos + 7..];
                let end = sub.find(',').unwrap_or(sub.len());
                if let Ok(v) = sub[..end].trim().parse::<f64>() {
                    lbound = Some(v);
                }
            }
            // Parse ubound
            if let Some(ub_pos) = line.find("ubound ") {
                let sub = &line[ub_pos + 7..];
                let end = sub.find(',').unwrap_or(sub.len());
                if let Ok(v) = sub[..end].trim().parse::<f64>() {
                    ubound = Some(v);
                }
            }
        }

        // Check for hypothesis acceptance
        let lower = line.to_lowercase();
        if lower.contains("accept h1") || lower.contains("candidate is stronger") {
            explicit_decision = Some(SprtDecision::AcceptedH1);
        } else if lower.contains("accept h0") || lower.contains("candidate is not stronger") || lower.contains("champion is stronger") {
            explicit_decision = Some(SprtDecision::AcceptedH0);
        }
    }

    let status = child.wait().map_err(|e| format!("Cutechess process wait error: {}", e))?;
    if !status.success() {
        eprintln!("[sprt] Warning: Cutechess exited with status: {}", status);
    }

    // Determine final decision
    let decision = if let Some(dec) = explicit_decision {
        dec
    } else if let (Some(l), Some(ub), Some(lb)) = (llr, ubound, lbound) {
        if l >= ub {
            SprtDecision::AcceptedH1
        } else if l <= lb {
            SprtDecision::AcceptedH0
        } else {
            SprtDecision::Inconclusive
        }
    } else {
        SprtDecision::Inconclusive
    };

    println!("--------------------------------------------------------------------------------");
    println!("SPRT Match Result:");
    println!("  Total Games: {}", total_games);
    println!("  Score:       Candidate +{} -{} ={} ({:.1}%)",
        wins, losses, draws, if total_games > 0 { (wins as f64 + 0.5 * draws as f64) / total_games as f64 * 100.0 } else { 0.0 });
    if let (Some(diff), Some(err)) = (elo_diff, elo_error) {
        println!("  Elo Diff:    {:+.1} +/- {:.1}", diff, err);
    } else if let Some(diff) = elo_diff {
        println!("  Elo Diff:    {:+.1}", diff);
    }
    if let (Some(l), Some(lb), Some(ub)) = (llr, lbound, ubound) {
        println!("  SPRT LLR:    {:.4} [Bounds: {:.2}, {:.2}]", l, lb, ub);
    }
    match &decision {
        SprtDecision::AcceptedH1 => println!("  🏆 VERDICT: ACCEPT H1 (Candidate is statistically superior -> Promoted to Champion!)"),
        SprtDecision::AcceptedH0 => println!("  ❌ VERDICT: ACCEPT H0 (Candidate failed SPRT -> Champion retained)"),
        SprtDecision::Inconclusive => println!("  ⚖️ VERDICT: INCONCLUSIVE (Max rounds reached without hypothesis confirmation)"),
    }
    println!("================================================================================\n");

    let verdict_str = match &decision {
        SprtDecision::AcceptedH1 => "ACCEPTED H1 (Candidate statistically superior)",
        SprtDecision::AcceptedH0 => "ACCEPTED H0 (Candidate failed to beat Champion)",
        SprtDecision::Inconclusive => "INCONCLUSIVE (Max rounds reached)",
    };
    let elo_str = if let (Some(diff), Some(err)) = (elo_diff, elo_error) {
        format!("{:+.1} +/- {:.1}", diff, err)
    } else if let Some(diff) = elo_diff {
        format!("{:+.1}", diff)
    } else {
        "N/A".to_string()
    };
    let llr_str = if let (Some(l), Some(lb), Some(ub)) = (llr, lbound, ubound) {
        format!("{:.4} [Bounds: {:.2}, {:.2}]", l, lb, ub)
    } else {
        "N/A".to_string()
    };
    let win_pct = if total_games > 0 {
        (wins as f64 + 0.5 * draws as f64) / total_games as f64 * 100.0
    } else {
        0.0
    };

    let log_entry = format!(
        "[{}] SPRT MATCH\n  Candidate: {}\n  Champion:  {}\n  Verdict:   {}\n  Score:     +{} -{} ={} (total: {}) ({:.1}%)\n  Elo Diff:  {}\n  SPRT LLR:  {}\n  TC: {} | Concurrency: {} | Book: {} (depth {})\n  --------------------------------------------------------------------------------",
        current_timestamp(),
        candidate_weights.display(),
        champion_weights.display(),
        verdict_str,
        wins, losses, draws, total_games, win_pct,
        elo_str,
        llr_str,
        config.time_control, config.concurrency, config.book_path, config.book_depth
    );
    append_to_match_log(&log_entry);

    Ok(SprtMatchResult {
        decision,
        wins,
        losses,
        draws,
        total_games,
        elo_diff,
        elo_error,
        llr,
        lbound,
        ubound,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sprt_config_defaults() {
        let cfg = SprtConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.concurrency, 2);
        assert_eq!(cfg.max_rounds, 200);
        assert_eq!(cfg.time_control, "5+0.1");
        assert_eq!(cfg.book_depth, 8);
        assert_eq!(cfg.elo0, 0.0);
        assert_eq!(cfg.elo1, 5.0);
        assert_eq!(cfg.alpha, 0.05);
        assert_eq!(cfg.beta, 0.05);
        assert_eq!(cfg.draw_movenumber, 40);
        assert_eq!(cfg.draw_movecount, 8);
        assert_eq!(cfg.draw_score, 10);
        assert_eq!(cfg.resign_movecount, 4);
        assert_eq!(cfg.resign_score, 550);
    }

    #[test]
    fn test_sprt_decision_bounds() {
        let lbound = -2.94;
        let ubound = 2.94;

        // Above ubound -> H1 accepted
        let dec_h1 = if 3.01 >= ubound { SprtDecision::AcceptedH1 } else { SprtDecision::Inconclusive };
        assert_eq!(dec_h1, SprtDecision::AcceptedH1);

        // Below lbound -> H0 accepted
        let dec_h0 = if -3.05 <= lbound { SprtDecision::AcceptedH0 } else { SprtDecision::Inconclusive };
        assert_eq!(dec_h0, SprtDecision::AcceptedH0);

        // In between -> Inconclusive
        let dec_inc = if 0.5 >= ubound {
            SprtDecision::AcceptedH1
        } else if 0.5 <= lbound {
            SprtDecision::AcceptedH0
        } else {
            SprtDecision::Inconclusive
        };
        assert_eq!(dec_inc, SprtDecision::Inconclusive);
    }

    #[test]
    fn test_current_timestamp() {
        let ts = current_timestamp();
        assert!(ts.ends_with(" UTC"));
        assert_eq!(ts.len(), 23); // "YYYY-MM-DD HH:MM:SS UTC"
    }
}
