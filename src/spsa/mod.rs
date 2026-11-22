pub mod params;
pub mod optimizer;
pub mod runner;
pub mod deployer;

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

use params::{
    format_discrete_param, load_config_yaml, write_search_config,
    ParamType, SEARCH_PARAMS,
};
use optimizer::SpsaOptimizer;
use runner::{CutechessRunner, CutechessRunnerConfig, SprtDecisionType};
use deployer::Deployer;

fn default_cutechess_path() -> String {
    "C:/Users/evanh/works/software/cutechess/cutechess-cli.exe".to_string()
}
fn default_engine_path() -> String {
    "engine/KittyEngineV5.exe".to_string()
}
fn default_engine_dir() -> String {
    "engine".to_string()
}
fn default_engine_cli_flags() -> String {
    "--mode bullet".to_string()
}
fn default_book_path() -> String {
    "C:/Users/evanh/works/software/cutechess/book/book.bin".to_string()
}
fn default_desktop_config() -> String {
    "C:/Users/evanh/Desktop/search_config.yaml".to_string()
}
fn default_local_config() -> String {
    "engine/search_config.yaml".to_string()
}
fn default_tc() -> String {
    "10+1".to_string()
}
fn default_concurrency() -> usize {
    12
}
fn default_games_per_iter() -> usize {
    80
}
fn default_book_depth() -> usize {
    1
}
fn default_draw_movenumber() -> usize {
    40
}
fn default_draw_movecount() -> usize {
    20
}
fn default_draw_score() -> i32 {
    10
}
fn default_resign_movecount() -> usize {
    10
}
fn default_resign_score() -> i32 {
    500
}
fn default_max_moves() -> usize {
    200
}
fn default_sprt_elo0() -> f64 {
    0.0
}
fn default_sprt_elo1() -> f64 {
    20.0
}
fn default_sprt_alpha() -> f64 {
    0.15
}
fn default_sprt_beta() -> f64 {
    0.15
}
fn default_sprt_max_rounds() -> usize {
    150
}
fn default_verify_interval() -> usize {
    50
}
fn default_verify_win_rate() -> f64 {
    0.58
}
fn default_verify_elo() -> f64 {
    20.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpsaSettings {
    #[serde(default = "default_cutechess_path")]
    pub cutechess_path: String,
    #[serde(default = "default_engine_path")]
    pub engine_path: String,
    #[serde(default = "default_engine_dir")]
    pub engine_dir: String,
    #[serde(default = "default_engine_cli_flags")]
    pub engine_cli_flags: String,
    #[serde(default = "default_book_path")]
    pub book_path: String,
    #[serde(default = "default_desktop_config")]
    pub desktop_config_path: String,
    #[serde(default = "default_local_config")]
    pub local_config_path: String,
    #[serde(default = "default_tc")]
    pub time_control: String,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    #[serde(default = "default_games_per_iter")]
    pub games_per_iteration: usize,
    #[serde(default = "default_book_depth")]
    pub book_depth: usize,
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
    #[serde(default = "default_max_moves")]
    pub max_moves: usize,
    #[serde(default = "default_sprt_elo0")]
    pub sprt_elo0: f64,
    #[serde(default = "default_sprt_elo1")]
    pub sprt_elo1: f64,
    #[serde(default = "default_sprt_alpha")]
    pub sprt_alpha: f64,
    #[serde(default = "default_sprt_beta")]
    pub sprt_beta: f64,
    #[serde(default = "default_sprt_max_rounds")]
    pub sprt_max_rounds: usize,
    #[serde(default = "default_verify_interval")]
    pub verify_interval_iterations: usize,
    #[serde(default = "default_verify_win_rate")]
    pub verify_win_rate_threshold: f64,
    #[serde(default = "default_verify_elo")]
    pub verify_elo_threshold: f64,
}

impl Default for SpsaSettings {
    fn default() -> Self {
        Self {
            cutechess_path: default_cutechess_path(),
            engine_path: default_engine_path(),
            engine_dir: default_engine_dir(),
            engine_cli_flags: default_engine_cli_flags(),
            book_path: default_book_path(),
            desktop_config_path: default_desktop_config(),
            local_config_path: default_local_config(),
            time_control: default_tc(),
            concurrency: default_concurrency(),
            games_per_iteration: default_games_per_iter(),
            book_depth: default_book_depth(),
            draw_movenumber: default_draw_movenumber(),
            draw_movecount: default_draw_movecount(),
            draw_score: default_draw_score(),
            resign_movecount: default_resign_movecount(),
            resign_score: default_resign_score(),
            max_moves: default_max_moves(),
            sprt_elo0: default_sprt_elo0(),
            sprt_elo1: default_sprt_elo1(),
            sprt_alpha: default_sprt_alpha(),
            sprt_beta: default_sprt_beta(),
            sprt_max_rounds: default_sprt_max_rounds(),
            verify_interval_iterations: default_verify_interval(),
            verify_win_rate_threshold: default_verify_win_rate(),
            verify_elo_threshold: default_verify_elo(),
        }
    }
}

impl SpsaSettings {
    pub fn load_or_default() -> Self {
        let json_paths = ["spsa_settings.json", "state/spsa_settings.json"];
        for p in json_paths {
            let path = Path::new(p);
            if path.exists() {
                if let Ok(content) = fs::read_to_string(path) {
                    if let Ok(s) = serde_json::from_str::<Self>(&content) {
                        return s;
                    }
                }
            }
        }
        Self::default()
    }
}

pub struct SPSATuner {
    pub settings: SpsaSettings,
    pub runner: CutechessRunner,
    pub optimizer: SpsaOptimizer,
    pub deployer: Deployer,
    pub state_dir: PathBuf,
    pub logs_dir: PathBuf,
}

impl SPSATuner {
    pub fn new() -> Self {
        let settings = SpsaSettings::load_or_default();
        let state_dir = PathBuf::from("state");
        let logs_dir = PathBuf::from("logs");
        let history_dir = PathBuf::from("history");

        let _ = fs::create_dir_all(&state_dir);
        let _ = fs::create_dir_all(&logs_dir);
        let _ = fs::create_dir_all(&history_dir);

        // Auto-detect engine executable if needed
        let mut engine_path = PathBuf::from(&settings.engine_path);
        let engine_dir = PathBuf::from(&settings.engine_dir);
        if !engine_path.exists() && engine_dir.is_dir() {
            if let Ok(entries) = fs::read_dir(&engine_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().map(|e| e == "exe").unwrap_or(false) {
                        engine_path = path;
                        break;
                    }
                }
            }
        }

        // Parse engine CLI args (filtering out --search_config to prevent collisions)
        let mut cli_args = Vec::new();
        let mut skip_next = false;
        for token in settings.engine_cli_flags.split_whitespace() {
            if skip_next {
                skip_next = false;
                continue;
            }
            if token == "--search_config" || token == "-search_config" {
                skip_next = true;
                continue;
            }
            if token.starts_with("--search_config=") {
                continue;
            }
            cli_args.push(token.to_string());
        }

        let runner_cfg = CutechessRunnerConfig {
            cutechess_path: PathBuf::from(&settings.cutechess_path),
            engine_path,
            engine_dir,
            engine_cli_args: cli_args,
            book_path: PathBuf::from(&settings.book_path),
            book_depth: settings.book_depth,
            time_control: settings.time_control.clone(),
            concurrency: settings.concurrency,
            draw_movenumber: settings.draw_movenumber,
            draw_movecount: settings.draw_movecount,
            draw_score: settings.draw_score,
            resign_movecount: settings.resign_movecount,
            resign_score: settings.resign_score,
            max_moves: settings.max_moves,
        };

        let runner = CutechessRunner::new(runner_cfg);
        let deployer = Deployer::new(
            &settings.desktop_config_path,
            Some(&settings.local_config_path),
            &history_dir,
        );

        // Load initial theta from state, or local config, or desktop config
        let initial_cfg = if Path::new(&settings.local_config_path).exists() {
            load_config_yaml(&settings.local_config_path).unwrap_or_default()
        } else {
            load_config_yaml(&settings.desktop_config_path).unwrap_or_default()
        };

        let state_path = state_dir.join("spsa_state.json");
        let optimizer = SpsaOptimizer::new(
            initial_cfg,
            0.602,
            0.101,
            1.0,
            1.0,
            20.0,
            state_path,
        );

        Self {
            settings,
            runner,
            optimizer,
            deployer,
            state_dir,
            logs_dir,
        }
    }

    pub fn run_iteration(&mut self) -> Result<(), String> {
        let k = self.optimizer.iteration;
        let (ak, ck) = self.optimizer.get_step_sizes(k);

        println!("--------------------------------------------------------------------------------");
        println!(
            "🎯 SPSA ITERATION {} (Games: {}) | ak={:.5}, ck={:.5} | TC: {} | Concurrency: {} | Batch: {}",
            k + 1, self.optimizer.total_games, ak, ck, self.runner.config.time_control, self.runner.config.concurrency, self.settings.games_per_iteration
        );

        // 1. Generate perturbations
        let (deltas, theta_plus, theta_minus) = self.optimizer.generate_perturbations();

        let plus_yaml = self.state_dir.join("theta_plus.yaml");
        let minus_yaml = self.state_dir.join("theta_minus.yaml");
        write_search_config(&theta_plus, &plus_yaml)?;
        write_search_config(&theta_minus, &minus_yaml)?;

        // 2. Run match between Theta+ and Theta-
        let pgn_file = self.logs_dir.join(format!("iter_{}_plus_vs_minus.pgn", k + 1));
        let match_res = self.runner.run_spsa_match(
            "Theta_Plus",
            &plus_yaml,
            "Theta_Minus",
            &minus_yaml,
            self.settings.games_per_iteration,
            Some(&pgn_file),
        )?;

        println!(
            "  🏁 Match Result: Theta+ scored {}W / {}L / {}D ({:.1}/{}) | Diff: {:+.3} | Elo: {:+.1} +/- {:.1}",
            match_res.wins, match_res.losses, match_res.draws, match_res.score, match_res.total_games, match_res.score_diff, match_res.elo_diff, match_res.elo_error
        );

        // 3. Update parameters
        self.optimizer.update(&deltas, match_res.score_diff, match_res.total_games, match_res.elo_diff);

        // 4. Save current theta
        let current_theta_path = self.state_dir.join("current_theta.yaml");
        let _ = write_search_config(&self.optimizer.theta, &current_theta_path);

        // 5. Verification check
        let deployed_theta = self.deployer.get_deployed_config();
        let win_rate = if match_res.total_games > 0 {
            match_res.score / (match_res.total_games as f64)
        } else {
            0.5
        };
        let top_rate = win_rate.max(1.0 - win_rate);
        let elo_gain = match_res.elo_diff.abs();

        let mut should_verify = false;
        let mut trigger_reason = "";

        if (k + 1) % self.settings.verify_interval_iterations == 0 {
            should_verify = true;
            trigger_reason = "Periodic verification interval reached";
        } else if match_res.total_games > 0 && (elo_gain >= self.settings.verify_elo_threshold || top_rate >= self.settings.verify_win_rate_threshold) {
            should_verify = true;
            trigger_reason = "Decisive iteration performance threshold exceeded";
        }

        if should_verify {
            let diffs = self.deployer.get_parameter_diff(&deployed_theta, &self.optimizer.theta);
            if diffs.is_empty() {
                println!("  [Verify] Candidate parameters are identical to baseline after rounding. Skipping SPRT.");
            } else {
                println!("  ⚡ Trigger: {} ({} discrete parameters differ). Starting SPRT verification...", trigger_reason, diffs.len());
                self.verify_and_deploy_if_better(&deployed_theta)?;
            }
        }

        Ok(())
    }

    pub fn verify_and_deploy_if_better(
        &mut self,
        deployed_theta: &HashMap<String, f64>,
    ) -> Result<bool, String> {
        let diffs = self.deployer.get_parameter_diff(deployed_theta, &self.optimizer.theta);
        if diffs.is_empty() {
            println!("  [SPRT Verification] Candidate discretized parameters are identical to baseline. Skipping.");
            return Ok(false);
        }

        let cand_yaml = self.state_dir.join("candidate.yaml");
        let base_yaml = self.state_dir.join("deployed_baseline.yaml");
        write_search_config(&self.optimizer.theta, &cand_yaml)?;
        write_search_config(deployed_theta, &base_yaml)?;

        println!("--------------------------------------------------------------------------------");
        println!("⚔️ INITIATING SPRT STATISTICAL VERIFICATION (Candidate vs Deployed Baseline)");
        println!(
            "  Bounds: Elo [{}, {}] | Alpha: {} | Beta: {} | Max Rounds: {}",
            self.settings.sprt_elo0, self.settings.sprt_elo1, self.settings.sprt_alpha, self.settings.sprt_beta, self.settings.sprt_max_rounds
        );
        println!("  Divergent Parameters ({}):", diffs.len());
        for (k, old_val, new_val) in diffs.iter().take(8) {
            println!("    - {:<38} : {:>8} -> {:<8}", k, old_val, new_val);
        }
        if diffs.len() > 8 {
            println!("    ... and {} more parameters", diffs.len() - 8);
        }

        let sprt_pgn = self.logs_dir.join(format!("sprt_verify_iter_{}.pgn", self.optimizer.iteration));
        let result = self.runner.run_sprt_verification(
            "Candidate_SPSA",
            &cand_yaml,
            "Deployed_Baseline",
            &base_yaml,
            self.settings.sprt_elo0,
            self.settings.sprt_elo1,
            self.settings.sprt_alpha,
            self.settings.sprt_beta,
            self.settings.sprt_max_rounds,
            Some(&sprt_pgn),
        )?;

        println!(
            "  🏁 SPRT Finished: Decision = {:?} | Score: +{} -{} ={} ({} games) | Elo: {:+.1} +/- {:.1} | LLR: {:.2} [{:.2}, {:.2}]",
            result.decision, result.wins, result.losses, result.draws, result.total_games, result.elo_diff, result.elo_error, result.llr, result.lbound, result.ubound
        );

        let is_promoted = if result.decision == SprtDecisionType::AcceptedH1 {
            true
        } else {
            // Net positive promotion
            result.total_games > 0 && (result.wins > result.losses || (result.wins >= result.losses && result.elo_diff > 0.0))
        };

        if is_promoted {
            let stats_summary = format!(
                "+{} -{} ={} ({} games) | Elo: {:+.1} +/- {:.1} | LLR: {:.2}",
                result.wins, result.losses, result.draws, result.total_games, result.elo_diff, result.elo_error, result.llr
            );
            let reason = if result.decision == SprtDecisionType::AcceptedH1 {
                "SPRT Accepted H1 (Statistically superior search parameters)"
            } else {
                "Decisive net positive edge over deployed baseline"
            };
            self.deployer.deploy(&self.optimizer.theta, reason, self.optimizer.iteration, Some(&stats_summary))?;
            Ok(true)
        } else {
            println!("  [Verification] Candidate did not outperform baseline. Retaining current deployed configuration.");
            Ok(false)
        }
    }

    pub fn run_loop(&mut self, max_iterations: usize) {
        println!("================================================================================");
        println!("🚀 KITTYENGINE SPSA SEARCH OPTIMIZER (Native Rust)");
        println!("================================================================================");
        println!("  Target Iterations : {}", max_iterations);
        println!("  Batch Size        : {} games/iteration", self.settings.games_per_iteration);
        println!("  Time Control      : {}", self.runner.config.time_control);
        println!("  Concurrency       : {} threads", self.runner.config.concurrency);
        println!("  Desktop Config    : {}", self.settings.desktop_config_path);
        println!("  Current Iteration : {}", self.optimizer.iteration);
        println!("  Total Games Played: {}", self.optimizer.total_games);
        println!("  (Press Ctrl+C to stop cleanly anytime with saved state)");
        println!("================================================================================");

        crate::signal::init();

        while self.optimizer.iteration < max_iterations {
            if crate::signal::is_stop_requested() {
                println!("\n🛑 Graceful SPSA stop requested. Saving state...");
                let _ = self.optimizer.save_state();
                break;
            }

            if let Err(e) = self.run_iteration() {
                eprintln!("\n⚠️ SPSA iteration failed: {}", e);
                break;
            }
        }

        println!("\nSPSA Tuner stopped. State cleanly saved at iteration {}.", self.optimizer.iteration);
    }

    pub fn status(&self) {
        println!("================================================================================");
        println!("KITTYENGINE SPSA TUNER STATUS");
        println!("================================================================================");
        println!("  Current Iteration : {}", self.optimizer.iteration);
        println!("  Total Games Played: {}", self.optimizer.total_games);
        println!("  Engine Binary     : {}", self.runner.config.engine_path.display());
        println!("  Engine Directory  : {}", self.runner.config.engine_dir.display());
        println!("  Engine CLI Args   : {:?}", self.runner.config.engine_cli_args);
        println!("  Desktop Config    : {}", self.settings.desktop_config_path);
        println!("  Time Control      : {}", self.runner.config.time_control);
        println!("  Concurrency       : {} parallel games", self.runner.config.concurrency);
        println!();

        let deployed_theta = self.deployer.get_deployed_config();
        let current_theta = &self.optimizer.theta;

        println!("{:<36} | {:<12} | {:<10} | {:<10} | {:<8} | {:<6}", "Parameter", "Current SPSA", "Discrete", "Deployed", "Status", "Tuning");
        println!("{:-<36}-+-{:-<12}-+-{:-<10}-+-{:-<10}-+-{:-<8}-+-{:-<6}", "", "", "", "", "", "");

        for p in SEARCH_PARAMS {
            let k = p.full_key();
            let cur = current_theta.get(&k).copied().unwrap_or(p.default_val);
            let dep = deployed_theta.get(&k).copied().unwrap_or(p.default_val);

            let cur_disc = format_discrete_param(&k, cur);
            let dep_disc = format_discrete_param(&k, dep);

            let status = if cur_disc == dep_disc { "MATCH" } else { "MODIFIED" };
            let tuning = if p.active { "ACTIVE" } else { "FIXED" };

            let cur_str = match p.param_type {
                ParamType::Int => format!("{:.4}", cur),
                ParamType::Float => format!("{:.5}", cur),
            };

            println!("{:<36} | {:<12} | {:<10} | {:<10} | {:<8} | {:<6}", k, cur_str, cur_disc, dep_disc, status, tuning);
        }

        if self.deployer.deploy_log_path.exists() {
            if let Ok(log_text) = fs::read_to_string(&self.deployer.deploy_log_path) {
                let lines: Vec<&str> = log_text.lines().collect();
                let tail = if lines.len() > 15 { &lines[lines.len() - 15..] } else { &lines[..] };
                println!("\nRecent Deployments:\n{}", tail.join("\n"));
            }
        }
    }

    pub fn verify(&mut self) -> Result<(), String> {
        println!("Running manual SPRT verification of current SPSA parameters vs deployed desktop config...");
        let deployed_theta = self.deployer.get_deployed_config();
        self.verify_and_deploy_if_better(&deployed_theta)?;
        Ok(())
    }

    pub fn deploy_now(&mut self) -> Result<(), String> {
        println!("Force deploying current SPSA parameters to desktop config...");
        self.deployer.deploy(
            &self.optimizer.theta,
            "Manual force deployment via CLI",
            self.optimizer.iteration,
            None,
        )?;
        Ok(())
    }

    pub fn reset(&mut self, force: bool) -> Result<(), String> {
        if !force {
            print!("Are you sure you want to reset SPSA state to desktop search_config.yaml? [y/N]: ");
            let _ = std::io::Write::flush(&mut std::io::stdout());
            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_ok() {
                if !input.trim().eq_ignore_ascii_case("y") {
                    println!("Reset aborted.");
                    return Ok(());
                }
            }
        }

        let deployed_theta = self.deployer.get_deployed_config();
        self.optimizer.theta.clear();
        for p in SEARCH_PARAMS {
            let key = p.full_key();
            let val = deployed_theta.get(&key).copied().unwrap_or(p.default_val);
            self.optimizer.theta.insert(key, val);
        }
        self.optimizer.iteration = 0;
        self.optimizer.total_games = 0;
        self.optimizer.history.clear();
        self.optimizer.save_state()?;

        println!("SPSA state reset cleanly to deployed desktop configuration.");
        Ok(())
    }
}

pub fn run_spsa_cli(args: &[String]) {
    let subcmd = args.get(2).map(|s| s.as_str()).unwrap_or("run");

    let mut tuner = SPSATuner::new();

    match subcmd {
        "run" => {
            let mut iterations = 1000;
            let mut games = None;
            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--iterations" | "-i" => {
                        if let Some(v) = args.get(i + 1).and_then(|s| s.parse::<usize>().ok()) {
                            iterations = v;
                            i += 1;
                        }
                    }
                    "--games" | "-g" => {
                        if let Some(v) = args.get(i + 1).and_then(|s| s.parse::<usize>().ok()) {
                            games = Some(v);
                            i += 1;
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            if let Some(g) = games {
                tuner.settings.games_per_iteration = g;
            }
            tuner.run_loop(iterations);
        }
        "status" => {
            tuner.status();
        }
        "verify" => {
            if let Err(e) = tuner.verify() {
                eprintln!("Verification failed: {}", e);
            }
        }
        "deploy" => {
            if let Err(e) = tuner.deploy_now() {
                eprintln!("Deployment failed: {}", e);
            }
        }
        "reset" => {
            let force = args.iter().any(|a| a == "--force" || a == "-f");
            if let Err(e) = tuner.reset(force) {
                eprintln!("Reset failed: {}", e);
            }
        }
        "help" | "--help" | "-h" => {
            println!("KittyEngine SPSA Parameter Optimizer & Auto-Deployer");
            println!("Usage: cargo run --release -- spsa <command> [OPTIONS]");
            println!();
            println!("Commands:");
            println!("  run [--iterations N] [--games G]   Run continuous SPSA optimization loop");
            println!("  status                             Display parameter comparison & tuning telemetry");
            println!("  verify                             Run SPRT verification against desktop config and deploy if superior");
            println!("  deploy                             Force deploy current SPSA parameters to desktop");
            println!("  reset [--force]                    Reset SPSA parameters to currently deployed desktop config");
        }
        other => {
            eprintln!("Unknown SPSA subcommand: '{}'", other);
            println!("Available subcommands: run, status, verify, deploy, reset");
        }
    }
}
