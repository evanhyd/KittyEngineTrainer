use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use super::params::{
    format_discrete_param, load_config_yaml, write_search_config, SEARCH_PARAMS,
};

pub struct Deployer {
    pub engine_config_path: PathBuf,
    pub history_dir: PathBuf,
    pub deploy_log_path: PathBuf,
}

impl Deployer {
    pub fn new<P: AsRef<Path>, R: AsRef<Path>>(
        engine_config_path: P,
        history_dir: R,
    ) -> Self {
        let h_dir = history_dir.as_ref().to_path_buf();
        let log_path = h_dir.join("deploy_log.txt");
        Self {
            engine_config_path: engine_config_path.as_ref().to_path_buf(),
            history_dir: h_dir,
            deploy_log_path: log_path,
        }
    }

    pub fn get_deployed_config(&self) -> HashMap<String, f64> {
        if self.engine_config_path.exists() {
            if let Ok(map) = load_config_yaml(&self.engine_config_path) {
                return map;
            }
        }
        HashMap::new()
    }

    pub fn get_parameter_diff(
        &self,
        old_cfg: &HashMap<String, f64>,
        new_cfg: &HashMap<String, f64>,
    ) -> Vec<(String, String, String)> {
        let mut diffs = Vec::new();
        for p in SEARCH_PARAMS {
            let key = p.full_key();
            let v_old = old_cfg.get(&key).copied().unwrap_or(p.default_val);
            let v_new = new_cfg.get(&key).copied().unwrap_or(p.default_val);

            let old_str = format_discrete_param(&key, v_old);
            let new_str = format_discrete_param(&key, v_new);

            if old_str != new_str {
                diffs.push((key, old_str, new_str));
            }
        }
        diffs
    }

    pub fn deploy(
        &self,
        new_theta: &HashMap<String, f64>,
        reason: &str,
        iteration: usize,
        stats_str: Option<&str>,
    ) -> Result<bool, String> {
        fs::create_dir_all(&self.history_dir)
            .map_err(|e| format!("Failed to create history dir: {}", e))?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let old_cfg = self.get_deployed_config();

        // 1. Backup existing engine config if present
        if self.engine_config_path.exists() {
            let backup_name = format!("search_config_backup_{}.yaml", now);
            let backup_path = self.history_dir.join(backup_name);
            let _ = fs::copy(&self.engine_config_path, &backup_path);
        }

        // 2. Write new config to engine directory only
        if let Some(parent) = self.engine_config_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        write_search_config(new_theta, &self.engine_config_path)?;

        let diffs = self.get_parameter_diff(&old_cfg, new_theta);

        println!(
            "  🚀 Deployed iter {} -> {} ({} params changed)",
            iteration,
            self.engine_config_path.display(),
            diffs.len()
        );

        // 3. Append to log
        use std::io::Write;
        if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(&self.deploy_log_path) {
            let _ = writeln!(f, "[Epoch {}] Deployment at iteration {}", now, iteration);
            let _ = writeln!(f, "Reason: {}", reason);
            if let Some(s) = stats_str {
                let _ = writeln!(f, "Stats: {}", s);
            }
            let _ = writeln!(f, "Parameter Changes ({}):", diffs.len());
            for (k, old_val, new_val) in &diffs {
                let _ = writeln!(f, "  {}: {} -> {}", k, old_val, new_val);
            }
            let _ = writeln!(f, "------------------------------------------------------------");
        }

        Ok(true)
    }
}
