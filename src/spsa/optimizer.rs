use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use super::params::{
    find_param, clamp_param_float, compute_discrete_perturbation, get_discrete_float, SEARCH_PARAMS,
};

pub struct SimpleRng(u64);

impl SimpleRng {
    pub fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self(if seed == 0 { 0x853c49e6748fea9b } else { seed })
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn gen_bernoulli(&mut self) -> i32 {
        if self.next_u64() & 1 == 0 { 1 } else { -1 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpsaIterationRecord {
    pub iteration: usize,
    pub score_diff: f64,
    pub games: usize,
    pub elo_diff: f64,
    pub ak: f64,
    pub ck: f64,
    pub changes: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct SpsaStateFile {
    pub iteration: usize,
    pub total_games: usize,
    pub alpha: f64,
    pub gamma: f64,
    pub a: f64,
    pub c: f64,
    pub A: f64,
    pub theta: HashMap<String, f64>,
    pub history_tail: Vec<SpsaIterationRecord>,
}

#[allow(non_snake_case)]
pub struct SpsaOptimizer {
    pub alpha: f64,
    pub gamma: f64,
    pub a: f64,
    pub c: f64,
    pub A: f64,
    pub state_path: PathBuf,
    pub iteration: usize,
    pub total_games: usize,
    pub theta: HashMap<String, f64>,
    pub history: Vec<SpsaIterationRecord>,
    rng: SimpleRng,
}

impl SpsaOptimizer {
    #[allow(non_snake_case)]
    pub fn new(
        initial_theta: HashMap<String, f64>,
        alpha: f64,
        gamma: f64,
        a: f64,
        c: f64,
        A: f64,
        state_path: PathBuf,
    ) -> Self {
        let mut theta = HashMap::new();
        for p in SEARCH_PARAMS {
            let key = p.full_key();
            let val = initial_theta.get(&key).copied().unwrap_or(p.default_val);
            theta.insert(key.clone(), clamp_param_float(&key, val));
        }

        let mut opt = Self {
            alpha,
            gamma,
            a,
            c,
            A,
            state_path,
            iteration: 0,
            total_games: 0,
            theta,
            history: Vec::new(),
            rng: SimpleRng::new(),
        };

        if opt.state_path.exists() {
            let _ = opt.load_state();
        }

        opt
    }

    pub fn get_step_sizes(&self, k: usize) -> (f64, f64) {
        let ak = self.a / (self.A + (k as f64) + 1.0).powf(self.alpha);
        let ck = self.c / ((k as f64) + 1.0).powf(self.gamma);
        (ak, ck)
    }

    pub fn generate_perturbations(
        &mut self,
    ) -> (HashMap<String, i32>, HashMap<String, f64>, HashMap<String, f64>) {
        let (_ak, ck) = self.get_step_sizes(self.iteration);
        let mut deltas = HashMap::new();
        let mut theta_plus = HashMap::new();
        let mut theta_minus = HashMap::new();

        for p in SEARCH_PARAMS {
            let key = p.full_key();
            let current_val = self.theta.get(&key).copied().unwrap_or(p.default_val);

            if !p.active {
                deltas.insert(key.clone(), 0);
                let disc = get_discrete_float(&key, current_val);
                theta_plus.insert(key.clone(), disc);
                theta_minus.insert(key.clone(), disc);
                continue;
            }

            let delta = self.rng.gen_bernoulli();
            deltas.insert(key.clone(), delta);

            let (plus_val, minus_val) = compute_discrete_perturbation(&key, current_val, delta, ck);
            theta_plus.insert(key.clone(), plus_val);
            theta_minus.insert(key.clone(), minus_val);
        }

        (deltas, theta_plus, theta_minus)
    }

    pub fn update(
        &mut self,
        deltas: &HashMap<String, i32>,
        score_diff: f64,
        games: usize,
        elo_diff: f64,
    ) {
        let (ak, ck) = self.get_step_sizes(self.iteration);
        self.total_games += games;

        let mut new_theta = HashMap::new();
        let mut param_changes = HashMap::new();

        for p in SEARCH_PARAMS {
            let key = p.full_key();
            let current_val = self.theta.get(&key).copied().unwrap_or(p.default_val);
            let delta = deltas.get(&key).copied().unwrap_or(0);

            if !p.active || delta == 0 {
                new_theta.insert(key.clone(), current_val);
                continue;
            }

            // SPSA gradient step: delta_theta = ak * a_step * (score_diff * delta) / (2.0 * ck)
            let step = ak * p.a_step * (score_diff * (delta as f64)) / (2.0 * ck);
            let updated_val = clamp_param_float(&key, current_val + step);
            let rounded_change = ((updated_val - current_val) * 100_000.0).round() / 100_000.0;

            new_theta.insert(key.clone(), updated_val);
            param_changes.insert(key.clone(), rounded_change);
        }

        self.history.push(SpsaIterationRecord {
            iteration: self.iteration,
            score_diff,
            games,
            elo_diff,
            ak,
            ck,
            changes: param_changes,
        });

        if self.history.len() > 50 {
            let excess = self.history.len() - 50;
            self.history.drain(0..excess);
        }

        self.theta = new_theta;
        self.iteration += 1;
        let _ = self.save_state();
    }

    pub fn save_state(&self) -> Result<(), String> {
        if let Some(parent) = self.state_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create state directory: {}", e))?;
        }

        let state_data = SpsaStateFile {
            iteration: self.iteration,
            total_games: self.total_games,
            alpha: self.alpha,
            gamma: self.gamma,
            a: self.a,
            c: self.c,
            A: self.A,
            theta: self.theta.clone(),
            history_tail: self.history.clone(),
        };

        let json_str = serde_json::to_string_pretty(&state_data)
            .map_err(|e| format!("Failed to serialize SPSA state: {}", e))?;

        fs::write(&self.state_path, json_str)
            .map_err(|e| format!("Failed to write SPSA state to {}: {}", self.state_path.display(), e))
    }

    pub fn load_state(&mut self) -> Result<(), String> {
        let content = fs::read_to_string(&self.state_path)
            .map_err(|e| format!("Failed to read {}: {}", self.state_path.display(), e))?;

        let data: SpsaStateFile = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse SPSA state from {}: {}", self.state_path.display(), e))?;

        self.iteration = data.iteration;
        self.total_games = data.total_games;
        self.alpha = data.alpha;
        self.gamma = data.gamma;
        self.a = data.a;
        self.c = data.c;
        self.A = data.A;

        for (k, v) in data.theta {
            if find_param(&k).is_some() {
                self.theta.insert(k.clone(), clamp_param_float(&k, v));
            }
        }
        self.history = data.history_tail;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_step_sizes_decay() {
        let opt = SpsaOptimizer::new(
            HashMap::new(),
            0.602,
            0.101,
            1.0,
            1.0,
            20.0,
            PathBuf::from("state/non_existent_test_state.json"),
        );

        let (a0, c0) = opt.get_step_sizes(0);
        let (a10, c10) = opt.get_step_sizes(10);
        let (a100, c100) = opt.get_step_sizes(100);

        assert!(a0 > a10 && a10 > a100, "Learning rate ak must decay monotonically!");
        assert!(c0 > c10 && c10 > c100, "Perturbation ck must decay monotonically!");
    }

    #[test]
    fn test_optimizer_update_step() {
        let mut opt = SpsaOptimizer::new(
            HashMap::new(),
            0.602,
            0.101,
            1.0,
            1.0,
            20.0,
            PathBuf::from("state/test_spsa_state_temp.json"),
        );

        let (deltas, _, _) = opt.generate_perturbations();
        let initial_aspiration = opt.theta["aspiration_window"];
        let delta_asp = deltas["aspiration_window"];

        // Suppose score_diff = +0.5 (Theta+ won decisively)
        opt.update(&deltas, 0.5, 40, 30.0);

        assert_eq!(opt.iteration, 1);
        assert_eq!(opt.total_games, 40);

        let new_aspiration = opt.theta["aspiration_window"];
        if delta_asp > 0 {
            assert!(new_aspiration > initial_aspiration, "Positive score and delta should increase parameter");
        } else if delta_asp < 0 {
            assert!(new_aspiration < initial_aspiration, "Positive score and negative delta should decrease parameter");
        }

        // Clean up temp file
        let _ = fs::remove_file("state/test_spsa_state_temp.json");
    }
}
