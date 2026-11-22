use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use super::params::{
    format_discrete_param, load_config_yaml, write_search_config, SEARCH_PARAMS,
};

pub struct Deployer {
    pub desktop_config_path: PathBuf,
    pub local_config_path: Option<PathBuf>,
    pub history_dir: PathBuf,
    pub deploy_log_path: PathBuf,
}

impl Deployer {
    pub fn new<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
        desktop_config_path: P,
        local_config_path: Option<Q>,
        history_dir: R,
    ) -> Self {
        let h_dir = history_dir.as_ref().to_path_buf();
        let log_path = h_dir.join("deploy_log.txt");
        Self {
            desktop_config_path: desktop_config_path.as_ref().to_path_buf(),
            local_config_path: local_config_path.map(|p| p.as_ref().to_path_buf()),
            history_dir: h_dir,
            deploy_log_path: log_path,
        }
    }

    pub fn get_deployed_config(&self) -> HashMap<String, f64> {
        if self.desktop_config_path.exists() {
            if let Ok(map) = load_config_yaml(&self.desktop_config_path) {
                return map;
            }
        }
        if let Some(local) = &self.local_config_path {
            if local.exists() {
                if let Ok(map) = load_config_yaml(local) {
                    return map;
                }
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

        // 1. Backup existing desktop config if present
        if self.desktop_config_path.exists() {
            let backup_name = format!("search_config_backup_{}.yaml", now);
            let backup_path = self.history_dir.join(backup_name);
            let _ = fs::copy(&self.desktop_config_path, &backup_path);
            println!("  [Deployer] Backed up old config: {}", backup_path.display());
        }

        // 2. Write new config to desktop & local engine folder
        write_search_config(new_theta, &self.desktop_config_path)?;
        if let Some(local) = &self.local_config_path {
            let _ = write_search_config(new_theta, local);
        }

        let diffs = self.get_parameter_diff(&old_cfg, new_theta);

        println!("================================================================================");
        println!("🚀 DEPLOYED NEW SEARCH CONFIGURATION (Iteration {})", iteration);
        println!("  Target: {}", self.desktop_config_path.display());
        println!("  Reason: {}", reason);
        if let Some(s) = stats_str {
            println!("  Stats:  {}", s);
        }
        println!("  Changed Parameters ({}/{}):", diffs.len(), SEARCH_PARAMS.len());
        for (k, old_val, new_val) in &diffs {
            println!("    - {:<38} : {:>8} -> {:<8}", k, old_val, new_val);
        }
        println!("================================================================================");

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
