use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub total_games: usize,
    pub score: f64,
    pub score_diff: f64,
    pub elo_diff: f64,
    pub elo_error: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SprtDecisionType {
    AcceptedH1,
    AcceptedH0,
    Inconclusive,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SprtResult {
    pub decision: SprtDecisionType,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub total_games: usize,
    pub llr: f64,
    pub lbound: f64,
    pub ubound: f64,
    pub elo_diff: f64,
    pub elo_error: f64,
}

use crate::sprt::{self, SprtDecision};

/// Minimum score fraction in (0, 1) required to achieve `elo` Elo difference.
pub fn required_score_fraction(elo: f64) -> f64 {
    sprt::required_score_fraction(elo)
}

/// Evaluates whether an SPRT match is already mathematically decided against
/// the configured `elo1` target given `wins`, `losses`, `draws`, and `max_games`.
pub fn check_sprt_early_cutoff(
    wins: usize,
    losses: usize,
    draws: usize,
    max_games: usize,
    elo1: f64,
) -> Option<SprtDecisionType> {
    sprt::check_sprt_early_cutoff(wins, losses, draws, max_games, elo1).map(|d| match d {
        SprtDecision::AcceptedH1 => SprtDecisionType::AcceptedH1,
        SprtDecision::AcceptedH0 => SprtDecisionType::AcceptedH0,
        SprtDecision::Inconclusive => SprtDecisionType::Inconclusive,
    })
}

fn to_clean_path<P: AsRef<Path>>(p: P) -> PathBuf {
    let p = p.as_ref();
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(p)
    };
    let s = abs.to_string_lossy();
    if let Some(stripped) = s.strip_prefix(r"\\?\") {
        PathBuf::from(stripped)
    } else {
        abs
    }
}

pub struct CutechessRunnerConfig {
    pub cutechess_path: PathBuf,
    pub engine_path: PathBuf,
    pub engine_dir: PathBuf,
    pub engine_cli_args: Vec<String>,
    pub book_path: PathBuf,
    pub book_depth: usize,
    pub time_control: String,
    pub concurrency: usize,
    pub draw_movenumber: usize,
    pub draw_movecount: usize,
    pub draw_score: i32,
    pub resign_movecount: usize,
    pub resign_score: i32,
    pub max_moves: usize,
}

pub struct CutechessRunner {
    pub config: CutechessRunnerConfig,
}

impl CutechessRunner {
    pub fn new(config: CutechessRunnerConfig) -> Self {
        Self { config }
    }

    fn build_engine_args(&self, name: &str, config_path: &Path) -> Vec<String> {
        let abs_config = to_clean_path(config_path);
        let abs_engine = to_clean_path(&self.config.engine_path);
        let abs_dir = to_clean_path(&self.config.engine_dir);

        let mut args = vec![
            "-engine".to_string(),
            format!("name={}", name),
            format!("cmd={}", abs_engine.display()),
            "arg=--search_config".to_string(),
            format!("arg={}", abs_config.display()),
        ];

        for custom_arg in &self.config.engine_cli_args {
            args.push(format!("arg={}", custom_arg));
        }

        args.push(format!("dir={}", abs_dir.display()));
        args.push("proto=uci".to_string());
        args
    }

    /// Runs an SPSA gradient match between Theta+ and Theta- in paired games.
    pub fn run_spsa_match(
        &self,
        label: &str,
        name_plus: &str,
        config_plus_path: &Path,
        name_minus: &str,
        config_minus_path: &Path,
        total_games: usize,
        pgn_path: Option<&Path>,
    ) -> Result<MatchResult, String> {
        let rounds = 1.max(total_games / 2);

        if !self.config.engine_path.exists() {
            return Err(format!(
                "KittyEngine binary not found at: {}",
                self.config.engine_path.display()
            ));
        }

        let clean_cutechess = to_clean_path(&self.config.cutechess_path);
        let clean_book = to_clean_path(&self.config.book_path);

        let mut cmd = Command::new(clean_cutechess);
        let plus_args = self.build_engine_args(name_plus, config_plus_path);
        for a in plus_args {
            cmd.arg(a);
        }
        let minus_args = self.build_engine_args(name_minus, config_minus_path);
        for a in minus_args {
            cmd.arg(a);
        }

        cmd.arg("-each")
            .arg(format!("tc={}", self.config.time_control))
            .arg("proto=uci")
            .arg(format!("book={}", clean_book.display()))
            .arg(format!("bookdepth={}", self.config.book_depth))
            .arg("-concurrency")
            .arg(self.config.concurrency.to_string())
            .arg("-rounds")
            .arg(rounds.to_string())
            .arg("-games")
            .arg("2")
            .arg("-repeat")
            .arg("-recover")
            .arg("-draw")
            .arg(format!("movenumber={}", self.config.draw_movenumber))
            .arg(format!("movecount={}", self.config.draw_movecount))
            .arg(format!("score={}", self.config.draw_score))
            .arg("-resign")
            .arg(format!("movecount={}", self.config.resign_movecount))
            .arg(format!("score={}", self.config.resign_score))
            .arg("twosided=true")
            .arg("-maxmoves")
            .arg(self.config.max_moves.to_string())
            .arg("-ratinginterval")
            .arg("1");

        if let Some(pgn) = pgn_path {
            let clean_pgn = to_clean_path(pgn);
            if let Some(parent) = clean_pgn.parent() {
                let _ = fs::create_dir_all(parent);
            }
            cmd.arg("-pgnout").arg(clean_pgn);
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::inherit());

        let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn Cutechess: {}", e))?;
        let stdout = child.stdout.take().ok_or("Failed to capture Cutechess stdout")?;
        let reader = BufReader::new(stdout);

        let mut wins = 0usize;
        let mut losses = 0usize;
        let mut draws = 0usize;
        let mut elo_diff = 0.0f64;
        let mut elo_error = 0.0f64;

        let score_prefix = format!("Score of {} vs {}:", name_plus, name_minus);

        for line_res in reader.lines() {
            let line = match line_res {
                Ok(l) => l,
                Err(_) => break,
            };

            let mut updated = false;

            // Score: "Score of Theta_Plus vs Theta_Minus: 10 - 5 - 15  [0.583] 30"
            if line.starts_with(&score_prefix) {
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() > 1 {
                    let tokens: Vec<&str> = parts[1].split_whitespace().collect();
                    if tokens.len() >= 5
                        && let (Ok(w), Ok(l), Ok(d)) = (
                            tokens[0].parse::<usize>(),
                            tokens[2].parse::<usize>(),
                            tokens[4].parse::<usize>(),
                        ) {
                            wins = w;
                            losses = l;
                            draws = d;
                            updated = true;
                        }
                }
            }

            // Elo: "Elo difference: 14.5 +/- 8.2"
            if let Some(pos) = line.find("Elo difference:") {
                let rest = &line[pos + 15..];
                let tokens: Vec<&str> = rest.split_whitespace().collect();
                if !tokens.is_empty()
                    && let Ok(v) = tokens[0].trim_end_matches(',').parse::<f64>()
                    && v.is_finite()
                {
                    elo_diff = v;
                    updated = true;
                }
                if tokens.len() >= 3 && tokens[1] == "+/-" {
                    let err_str = tokens[2].trim_end_matches(',');
                    if let Ok(err) = err_str.parse::<f64>()
                        && err.is_finite()
                    {
                        elo_error = err;
                    }
                }
            }

            if updated {
                let total = wins + losses + draws;
                let pct = if total > 0 {
                    (wins as f64 + 0.5 * draws as f64) / total as f64 * 100.0
                } else {
                    0.0
                };
                print!(
                    "\r[{}] {:>2}/{} (+{} -{} ={}) {:>5.1}% | Elo: {:>+5.1} +/- {:<4.1}   ",
                    label, total, total_games, wins, losses, draws, pct, elo_diff, elo_error
                );
                let _ = std::io::stdout().flush();

                // Early Mathematical Cutoff for SPSA iteration:
                let remaining = total_games.saturating_sub(total);
                let min_check = (total_games / 3).max(10);
                if remaining > 0 && total >= min_check && (wins > losses + remaining || losses > wins + remaining) {
                    print!(
                        "\r[{}] {:>2}/{} (+{} -{} ={}) {:>5.1}% | Elo: {:>+5.1} +/- {:<4.1} [cutoff]",
                        label, total, total_games, wins, losses, draws, pct, elo_diff, elo_error
                    );
                    let _ = std::io::stdout().flush();
                    let _ = child.kill();
                    break;
                }
            }
        }

        let status = child.wait().map_err(|e| format!("Cutechess process wait error: {}", e))?;
        println!();

        let total = wins + losses + draws;
        if total == 0 {
            return Err(format!(
                "Cutechess exited with status {} and 0 games played. Check Cutechess path and configuration.",
                status
            ));
        }

        let score = wins as f64 + 0.5 * (draws as f64);
        let score_diff = if total > 0 {
            (wins as f64 - losses as f64) / (total as f64)
        } else {
            0.0
        };

        Ok(MatchResult {
            wins,
            losses,
            draws,
            total_games: total,
            score,
            score_diff,
            elo_diff,
            elo_error,
        })
    }

    /// Runs an SPRT match to determine whether candidate is statistically better than baseline.
    pub fn run_sprt_verification(
        &self,
        candidate_name: &str,
        candidate_config_path: &Path,
        baseline_name: &str,
        baseline_config_path: &Path,
        elo0: f64,
        elo1: f64,
        alpha: f64,
        beta: f64,
        max_rounds: usize,
        pgn_path: Option<&Path>,
    ) -> Result<SprtResult, String> {
        if !self.config.engine_path.exists() {
            return Err(format!(
                "KittyEngine binary not found at: {}",
                self.config.engine_path.display()
            ));
        }

        let clean_cutechess = to_clean_path(&self.config.cutechess_path);
        let clean_book = to_clean_path(&self.config.book_path);

        let mut cmd = Command::new(clean_cutechess);
        let cand_args = self.build_engine_args(candidate_name, candidate_config_path);
        for a in cand_args {
            cmd.arg(a);
        }
        let base_args = self.build_engine_args(baseline_name, baseline_config_path);
        for a in base_args {
            cmd.arg(a);
        }

        cmd.arg("-each")
            .arg(format!("tc={}", self.config.time_control))
            .arg("proto=uci")
            .arg(format!("book={}", clean_book.display()))
            .arg(format!("bookdepth={}", self.config.book_depth))
            .arg("-concurrency")
            .arg(self.config.concurrency.to_string())
            .arg("-rounds")
            .arg(max_rounds.to_string())
            .arg("-games")
            .arg("2")
            .arg("-repeat")
            .arg("-recover")
            .arg("-sprt")
            .arg(format!("elo0={}", elo0))
            .arg(format!("elo1={}", elo1))
            .arg(format!("alpha={}", alpha))
            .arg(format!("beta={}", beta))
            .arg("-draw")
            .arg(format!("movenumber={}", self.config.draw_movenumber))
            .arg(format!("movecount={}", self.config.draw_movecount))
            .arg(format!("score={}", self.config.draw_score))
            .arg("-resign")
            .arg(format!("movecount={}", self.config.resign_movecount))
            .arg(format!("score={}", self.config.resign_score))
            .arg("twosided=true")
            .arg("-maxmoves")
            .arg(self.config.max_moves.to_string())
            .arg("-ratinginterval")
            .arg("1");

        if let Some(pgn) = pgn_path {
            let clean_pgn = to_clean_path(pgn);
            if let Some(parent) = clean_pgn.parent() {
                let _ = fs::create_dir_all(parent);
            }
            cmd.arg("-pgnout").arg(clean_pgn);
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::inherit());

        let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn Cutechess: {}", e))?;
        let stdout = child.stdout.take().ok_or("Failed to capture Cutechess stdout")?;
        let reader = BufReader::new(stdout);

        let mut wins = 0usize;
        let mut losses = 0usize;
        let mut draws = 0usize;
        let mut llr = 0.0f64;
        let mut lbound = 0.0f64;
        let mut ubound = 0.0f64;
        let mut elo_diff = 0.0f64;
        let mut elo_error = 0.0f64;
        let mut explicit_decision = None;

        let score_prefix = format!("Score of {} vs {}:", candidate_name, baseline_name);

        for line_res in reader.lines() {
            let line = match line_res {
                Ok(l) => l,
                Err(_) => break,
            };

            if line.contains("Error:") {
                eprintln!("\n  [Cutechess Error] {}", line.trim());
            }

            let mut updated = false;

            if line.starts_with(&score_prefix) {
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() > 1 {
                    let tokens: Vec<&str> = parts[1].split_whitespace().collect();
                    if tokens.len() >= 5
                        && let (Ok(w), Ok(l), Ok(d)) = (
                            tokens[0].parse::<usize>(),
                            tokens[2].parse::<usize>(),
                            tokens[4].parse::<usize>(),
                        ) {
                            wins = w;
                            losses = l;
                            draws = d;
                            updated = true;
                        }
                }
            }

            if let Some(pos) = line.find("Elo difference:") {
                let rest = &line[pos + 15..];
                let tokens: Vec<&str> = rest.split_whitespace().collect();
                if !tokens.is_empty()
                    && let Ok(v) = tokens[0].trim_end_matches(',').parse::<f64>()
                    && v.is_finite()
                {
                    elo_diff = v;
                    updated = true;
                }
                if tokens.len() >= 3 && tokens[1] == "+/-" {
                    let err_str = tokens[2].trim_end_matches(',');
                    if let Ok(err) = err_str.parse::<f64>()
                        && err.is_finite()
                    {
                        elo_error = err;
                    }
                }
            }

            if line.contains("SPRT:") {
                if let Some(pos) = line.find("llr ") {
                    let sub = &line[pos + 4..];
                    if let Some(end) = sub.find([' ', '('])
                        && let Ok(v) = sub[..end].parse::<f64>() {
                            llr = v;
                            updated = true;
                        }
                }
                if let Some(pos) = line.find("lbound ") {
                    let sub = &line[pos + 7..];
                    let end = sub.find(',').unwrap_or(sub.len());
                    if let Ok(v) = sub[..end].trim().parse::<f64>() {
                        lbound = v;
                    }
                }
                if let Some(pos) = line.find("ubound ") {
                    let sub = &line[pos + 7..];
                    let end = sub.find(',').unwrap_or(sub.len());
                    if let Ok(v) = sub[..end].trim().parse::<f64>() {
                        ubound = v;
                    }
                }
            }

            let lower = line.to_lowercase();
            if lower.contains("accept h1") || lower.contains("h1 was accepted") {
                explicit_decision = Some(SprtDecisionType::AcceptedH1);
            } else if lower.contains("accept h0") || lower.contains("h0 was accepted") {
                explicit_decision = Some(SprtDecisionType::AcceptedH0);
            }

            if updated {
                let total = wins + losses + draws;
                let pct = if total > 0 {
                    (wins as f64 + 0.5 * draws as f64) / total as f64 * 100.0
                } else {
                    0.0
                };
                print!(
                    "\r  [SPRT] {:>3}/{} (+{} -{} ={}) {:>5.1}% | Elo: {:>+5.1} +/- {:<4.1} | LLR: {:>+5.2} [{:.2}, {:.2}]   ",
                    total, max_rounds * 2, wins, losses, draws, pct, elo_diff, elo_error, llr, lbound, ubound
                );
                let _ = std::io::stdout().flush();

                // Early Mathematical Cutoff respecting configured Elo demand (elo1):
                let max_games = max_rounds * 2;
                if line.contains("SPRT:")
                    && let Some(cutoff_decision) = check_sprt_early_cutoff(wins, losses, draws, max_games, elo1)
                {
                    explicit_decision = Some(cutoff_decision);
                    let _ = child.kill();
                    break;
                }
            }
        }

        let status = child.wait().map_err(|e| format!("Cutechess process wait error: {}", e))?;
        print!("\r{: <96}\r", "");
        let _ = std::io::stdout().flush();

        let total = wins + losses + draws;
        if total == 0 {
            return Err(format!(
                "Cutechess SPRT exited with status {} and 0 games played. Check Cutechess path and configuration.",
                status
            ));
        }

        let decision = if let Some(dec) = explicit_decision {
            dec
        } else if ubound > 0.0 && llr >= ubound {
            SprtDecisionType::AcceptedH1
        } else if lbound < 0.0 && llr <= lbound {
            SprtDecisionType::AcceptedH0
        } else {
            SprtDecisionType::Inconclusive
        };

        Ok(SprtResult {
            decision,
            wins,
            losses,
            draws,
            total_games: wins + losses + draws,
            llr,
            lbound,
            ubound,
            elo_diff,
            elo_error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_required_score_fraction() {
        assert!((required_score_fraction(0.0) - 0.5).abs() < 1e-9);
        // +20 Elo requires ~52.875%
        let p20 = required_score_fraction(20.0);
        assert!(p20 > 0.528 && p20 < 0.529);
    }

    #[test]
    fn test_sprt_early_cutoff_respects_elo_bound() {
        // 24-game match with +20 Elo demand -> requires 12.69 pts
        // 2W 2L 19D (23 games, 1 left): even winning the last game gives 3W 2L 19D = 12.5 pts (+14.5 Elo < 20.0) -> AcceptedH0
        assert_eq!(
            check_sprt_early_cutoff(2, 2, 19, 24, 20.0),
            Some(SprtDecisionType::AcceptedH0)
        );

        // 100-game match with +20 Elo demand -> requires 52.875 pts
        // +10 -8 =79 (97 games, 3 left): positive Elo, but max possible is 49.5 + 3 = 52.5 < 52.875 -> AcceptedH0
        assert_eq!(
            check_sprt_early_cutoff(10, 8, 79, 100, 20.0),
            Some(SprtDecisionType::AcceptedH0)
        );

        // +10 -8 =78 (96 games, 4 left): max possible is 49.0 + 4 = 53.0 >= 52.875 -> still open
        assert_eq!(check_sprt_early_cutoff(10, 8, 78, 100, 20.0), None);

        // +45 -5 =20 (70 games, 30 left): current score 55.0 >= 52.875 -> guaranteed >= +20 Elo -> AcceptedH1
        assert_eq!(
            check_sprt_early_cutoff(45, 5, 20, 100, 20.0),
            Some(SprtDecisionType::AcceptedH1)
        );

        // Finished match (remaining == 0) does not trigger early cutoff
        assert_eq!(check_sprt_early_cutoff(3, 2, 19, 24, 20.0), None);
    }
}
