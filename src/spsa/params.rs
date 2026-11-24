use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamType {
    Int,
    Float,
}

#[derive(Debug, Clone, Copy)]
pub struct ParamDef {
    pub name: &'static str,
    pub section: &'static str,
    pub param_type: ParamType,
    pub min_val: f64,
    pub max_val: f64,
    pub c_step: f64,
    pub a_step: f64,
    pub active: bool,
    pub default_val: f64,
}

impl ParamDef {
    pub const fn new(
        name: &'static str,
        section: &'static str,
        param_type: ParamType,
        min_val: f64,
        max_val: f64,
        c_step: f64,
        a_step: f64,
        active: bool,
        default_val: f64,
    ) -> Self {
        Self {
            name,
            section,
            param_type,
            min_val,
            max_val,
            c_step,
            a_step,
            active,
            default_val,
        }
    }

    pub fn full_key(&self) -> String {
        if self.section.is_empty() {
            self.name.to_string()
        } else {
            format!("{}.{}", self.section, self.name)
        }
    }
}

pub const SEARCH_PARAMS: &[ParamDef] = &[
    // Top-level (Aspiration window fixed as requested; improving_threshold active)
    ParamDef::new("aspiration_window", "", ParamType::Int, 20.0, 250.0, 10.0, 100.0, false, 100.0),
    ParamDef::new("aspiration_window_divisor", "", ParamType::Int, 2.0, 16.0, 1.0, 20.0, false, 5.0),
    ParamDef::new("improving_threshold", "", ParamType::Int, 0.0, 150.0, 6.0, 60.0, true, 40.0),

    // Reverse Futility Pruning (Static Null Move Pruning)
    ParamDef::new("depth", "reverse_futility_pruning", ParamType::Int, 1.0, 8.0, 1.0, 15.0, true, 3.0),
    ParamDef::new("multiplier", "reverse_futility_pruning", ParamType::Int, 30.0, 250.0, 15.0, 120.0, true, 100.0),
    ParamDef::new("base", "reverse_futility_pruning", ParamType::Int, -50.0, 100.0, 10.0, 80.0, true, 0.0),
    ParamDef::new("improving_bonus", "reverse_futility_pruning", ParamType::Int, 0.0, 100.0, 8.0, 60.0, true, 30.0),

    // Null Move Pruning
    ParamDef::new("depth", "null_move_pruning", ParamType::Int, 1.0, 8.0, 1.0, 15.0, true, 4.0),
    ParamDef::new("base", "null_move_pruning", ParamType::Int, 1.0, 6.0, 1.0, 20.0, true, 3.0),
    ParamDef::new("depth_divisor", "null_move_pruning", ParamType::Int, 2.0, 12.0, 1.0, 20.0, true, 6.0),
    ParamDef::new("eval_divisor", "null_move_pruning", ParamType::Int, 50.0, 500.0, 25.0, 200.0, true, 200.0),
    ParamDef::new("max_eval_bonus", "null_move_pruning", ParamType::Int, 0.0, 6.0, 1.0, 16.0, true, 2.0),

    // Futility Pruning
    ParamDef::new("depth", "futility_pruning", ParamType::Int, 1.0, 8.0, 1.0, 15.0, true, 3.0),
    ParamDef::new("multiplier", "futility_pruning", ParamType::Int, 30.0, 250.0, 15.0, 120.0, true, 100.0),
    ParamDef::new("base", "futility_pruning", ParamType::Int, -50.0, 100.0, 10.0, 80.0, true, 0.0),
    ParamDef::new("improving_bonus", "futility_pruning", ParamType::Int, 0.0, 80.0, 8.0, 60.0, true, 20.0),

    // SEE Pruning
    ParamDef::new("depth", "see_pruning", ParamType::Int, 1.0, 8.0, 1.0, 15.0, true, 3.0),
    ParamDef::new("multiplier", "see_pruning", ParamType::Int, -250.0, -30.0, 15.0, 120.0, true, -100.0),
    ParamDef::new("base", "see_pruning", ParamType::Int, -100.0, 50.0, 10.0, 80.0, true, 0.0),

    // Late Move Pruning
    ParamDef::new("depth", "late_move_pruning", ParamType::Int, 1.0, 10.0, 1.0, 15.0, true, 4.0),
    ParamDef::new("multiplier", "late_move_pruning", ParamType::Int, 1.0, 6.0, 1.0, 20.0, true, 2.0),
    ParamDef::new("base", "late_move_pruning", ParamType::Int, 1.0, 16.0, 1.0, 20.0, true, 6.0),

    // History Pruning
    ParamDef::new("depth", "history_pruning", ParamType::Int, 1.0, 8.0, 1.0, 15.0, true, 3.0),
    ParamDef::new("multiplier", "history_pruning", ParamType::Int, -600.0, -50.0, 30.0, 250.0, true, -300.0),
    ParamDef::new("base", "history_pruning", ParamType::Int, -1000.0, -100.0, 50.0, 375.0, true, -500.0),

    // Late Move Reductions (LMR)
    ParamDef::new("depth", "late_move_reductions", ParamType::Int, 1.0, 6.0, 1.0, 15.0, true, 3.0),
    ParamDef::new("base", "late_move_reductions", ParamType::Float, 0.2, 2.0, 0.06, 0.50, true, 0.7844),
    ParamDef::new("divisor", "late_move_reductions", ParamType::Float, 1.0, 5.0, 0.15, 1.25, true, 2.69),
    ParamDef::new("history_k", "late_move_reductions", ParamType::Float, 0.0001, 0.01, 0.0003, 0.0025, true, 0.001),
    ParamDef::new("history_b", "late_move_reductions", ParamType::Float, -1.0, 1.0, 0.08, 0.60, true, 0.0),

    // Internal Iterative Deepening (IID)
    ParamDef::new("depth", "internal_iterative_deepening", ParamType::Int, 1.0, 8.0, 1.0, 15.0, true, 4.0),

    // Singular Extension
    ParamDef::new("depth", "singular_extension", ParamType::Int, 4.0, 12.0, 1.0, 15.0, true, 8.0),
    ParamDef::new("depth_margin", "singular_extension", ParamType::Int, 1.0, 6.0, 1.0, 20.0, true, 3.0),
    ParamDef::new("multiplier", "singular_extension", ParamType::Int, 0.0, 10.0, 1.0, 20.0, true, 2.0),
    ParamDef::new("base", "singular_extension", ParamType::Int, 0.0, 80.0, 4.0, 50.0, true, 30.0),

    // History Heuristics
    ParamDef::new("max_quiet_history", "history_heuristics", ParamType::Int, 200.0, 2500.0, 80.0, 500.0, true, 770.0),
    ParamDef::new("max_refutation_history", "history_heuristics", ParamType::Int, 300.0, 3000.0, 120.0, 750.0, true, 1230.0),
    ParamDef::new("max_followup_history", "history_heuristics", ParamType::Int, 100.0, 2000.0, 60.0, 375.0, true, 500.0),
    ParamDef::new("max_capture_history", "history_heuristics", ParamType::Int, 100.0, 2000.0, 50.0, 300.0, true, 400.0),
];

pub fn find_param(key: &str) -> Option<&'static ParamDef> {
    SEARCH_PARAMS.iter().find(|p| p.full_key() == key)
}

pub fn clamp_param_float(key: &str, val: f64) -> f64 {
    if let Some(p) = find_param(key) {
        val.clamp(p.min_val, p.max_val)
    } else {
        val
    }
}

pub fn format_discrete_param(key: &str, val: f64) -> String {
    if let Some(p) = find_param(key) {
        let clamped = val.clamp(p.min_val, p.max_val);
        match p.param_type {
            ParamType::Int => format!("{}", clamped.round() as i64),
            ParamType::Float => {
                // Round float to 5 decimal places
                let rounded = (clamped * 100_000.0).round() / 100_000.0;
                format!("{:.5}", rounded)
            }
        }
    } else {
        format!("{:.5}", val)
    }
}

pub fn get_discrete_float(key: &str, val: f64) -> f64 {
    if let Some(p) = find_param(key) {
        let clamped = val.clamp(p.min_val, p.max_val);
        match p.param_type {
            ParamType::Int => clamped.round(),
            ParamType::Float => (clamped * 100_000.0).round() / 100_000.0,
        }
    } else {
        val
    }
}

pub fn compute_discrete_perturbation(
    key: &str,
    current_val: f64,
    delta: i32,
    ck: f64,
) -> (f64, f64) {
    let p = match find_param(key) {
        Some(p) => p,
        None => return (current_val, current_val),
    };

    match p.param_type {
        ParamType::Int => {
            let pert = 1.0f64.max((ck * p.c_step).round());
            let cur_disc = current_val.clamp(p.min_val, p.max_val).round();
            let plus = cur_disc + (delta as f64) * pert;
            let minus = cur_disc - (delta as f64) * pert;

            let mut plus_clamped = plus.clamp(p.min_val, p.max_val).round();
            let mut minus_clamped = minus.clamp(p.min_val, p.max_val).round();

            // If boundary caused values to collapse to the same integer, shift
            if (plus_clamped - minus_clamped).abs() < 0.1 {
                if plus_clamped >= p.max_val {
                    minus_clamped = p.min_val.max(plus_clamped - pert).round();
                } else if plus_clamped <= p.min_val {
                    plus_clamped = p.max_val.min(minus_clamped + pert).round();
                }
            }

            // Enforce strict inequality based on delta
            if delta > 0 && plus_clamped <= minus_clamped {
                if plus_clamped < p.max_val {
                    plus_clamped = p.max_val.min(minus_clamped + 1.0);
                } else if minus_clamped > p.min_val {
                    minus_clamped = p.min_val.max(plus_clamped - 1.0);
                }
            } else if delta < 0 && plus_clamped >= minus_clamped {
                if minus_clamped < p.max_val {
                    minus_clamped = p.max_val.min(plus_clamped + 1.0);
                } else if plus_clamped > p.min_val {
                    plus_clamped = p.min_val.max(minus_clamped - 1.0);
                }
            }

            (plus_clamped, minus_clamped)
        }
        ParamType::Float => {
            let pert = 1e-5f64.max(ck * p.c_step);
            let mut plus = (current_val + (delta as f64) * pert).clamp(p.min_val, p.max_val);
            let mut minus = (current_val - (delta as f64) * pert).clamp(p.min_val, p.max_val);
            plus = (plus * 100_000.0).round() / 100_000.0;
            minus = (minus * 100_000.0).round() / 100_000.0;

            if (plus - minus).abs() < 1e-6 {
                if plus >= p.max_val {
                    minus = (p.min_val.max(plus - pert) * 100_000.0).round() / 100_000.0;
                } else if plus <= p.min_val {
                    plus = (p.max_val.min(minus + pert) * 100_000.0).round() / 100_000.0;
                }
            }
            (plus, minus)
        }
    }
}

pub fn load_config_yaml<P: AsRef<Path>>(yaml_path: P) -> Result<HashMap<String, f64>, String> {
    let mut flat = HashMap::new();
    // Default initialization
    for p in SEARCH_PARAMS {
        flat.insert(p.full_key(), p.default_val);
    }

    let path = yaml_path.as_ref();
    if !path.exists() {
        return Ok(flat);
    }

    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read YAML at {}: {}", path.display(), e))?;

    let val: serde_yaml::Value = serde_yaml::from_str(&content)
        .map_err(|e| format!("YAML parsing error in {}: {}", path.display(), e))?;

    for p in SEARCH_PARAMS {
        let key = p.full_key();
        let parsed_val = if p.section.is_empty() {
            val.get(p.name)
        } else {
            val.get(p.section).and_then(|sec| sec.get(p.name))
        };

        if let Some(v) = parsed_val {
            if let Some(f) = v.as_f64() {
                flat.insert(key, clamp_param_float(&p.full_key(), f));
            } else if let Some(i) = v.as_i64() {
                flat.insert(key, clamp_param_float(&p.full_key(), i as f64));
            }
        }
    }

    Ok(flat)
}

pub fn serialize_search_config(values: &HashMap<String, f64>) -> String {
    let get_val = |full_key: &str| -> String {
        let val = values.get(full_key).copied().unwrap_or_else(|| {
            find_param(full_key).map(|p| p.default_val).unwrap_or(0.0)
        });
        format_discrete_param(full_key, val)
    };

    let mut out = String::new();
    out.push_str("# KittyEngine Negamax Search Configuration\n");
    out.push_str(&format!("aspiration_window: {}\n", get_val("aspiration_window")));
    out.push_str(&format!("aspiration_window_divisor: {}\n", get_val("aspiration_window_divisor")));
    out.push_str(&format!("improving_threshold: {}\n\n", get_val("improving_threshold")));

    out.push_str("reverse_futility_pruning:\n");
    out.push_str(&format!("  depth: {}\n", get_val("reverse_futility_pruning.depth")));
    out.push_str(&format!("  multiplier: {}\n", get_val("reverse_futility_pruning.multiplier")));
    out.push_str(&format!("  base: {}\n", get_val("reverse_futility_pruning.base")));
    out.push_str(&format!("  improving_bonus: {}\n\n", get_val("reverse_futility_pruning.improving_bonus")));

    out.push_str("null_move_pruning:\n");
    out.push_str(&format!("  depth: {}\n", get_val("null_move_pruning.depth")));
    out.push_str(&format!("  base: {}\n", get_val("null_move_pruning.base")));
    out.push_str(&format!("  depth_divisor: {}\n", get_val("null_move_pruning.depth_divisor")));
    out.push_str(&format!("  eval_divisor: {}\n", get_val("null_move_pruning.eval_divisor")));
    out.push_str(&format!("  max_eval_bonus: {}\n\n", get_val("null_move_pruning.max_eval_bonus")));

    out.push_str("futility_pruning:\n");
    out.push_str(&format!("  depth: {}\n", get_val("futility_pruning.depth")));
    out.push_str(&format!("  multiplier: {}\n", get_val("futility_pruning.multiplier")));
    out.push_str(&format!("  base: {}\n", get_val("futility_pruning.base")));
    out.push_str(&format!("  improving_bonus: {}\n\n", get_val("futility_pruning.improving_bonus")));

    out.push_str("see_pruning:\n");
    out.push_str(&format!("  depth: {}\n", get_val("see_pruning.depth")));
    out.push_str(&format!("  multiplier: {}\n", get_val("see_pruning.multiplier")));
    out.push_str(&format!("  base: {}\n\n", get_val("see_pruning.base")));

    out.push_str("late_move_pruning:\n");
    out.push_str(&format!("  depth: {}\n", get_val("late_move_pruning.depth")));
    out.push_str(&format!("  multiplier: {}\n", get_val("late_move_pruning.multiplier")));
    out.push_str(&format!("  base: {}\n\n", get_val("late_move_pruning.base")));

    out.push_str("history_pruning:\n");
    out.push_str(&format!("  depth: {}\n", get_val("history_pruning.depth")));
    out.push_str(&format!("  multiplier: {}\n", get_val("history_pruning.multiplier")));
    out.push_str(&format!("  base: {}\n\n", get_val("history_pruning.base")));

    out.push_str("late_move_reductions:\n");
    out.push_str(&format!("  depth: {}\n", get_val("late_move_reductions.depth")));
    out.push_str(&format!("  base: {}\n", get_val("late_move_reductions.base")));
    out.push_str(&format!("  divisor: {}\n", get_val("late_move_reductions.divisor")));
    out.push_str(&format!("  history_k: {}\n", get_val("late_move_reductions.history_k")));
    out.push_str(&format!("  history_b: {}\n\n", get_val("late_move_reductions.history_b")));

    out.push_str("internal_iterative_deepening:\n");
    out.push_str(&format!("  depth: {}\n\n", get_val("internal_iterative_deepening.depth")));

    out.push_str("singular_extension:\n");
    out.push_str(&format!("  depth: {}\n", get_val("singular_extension.depth")));
    out.push_str(&format!("  depth_margin: {}\n", get_val("singular_extension.depth_margin")));
    out.push_str(&format!("  multiplier: {}\n", get_val("singular_extension.multiplier")));
    out.push_str(&format!("  base: {}\n\n", get_val("singular_extension.base")));

    out.push_str("history_heuristics:\n");
    out.push_str(&format!("  max_quiet_history: {}\n", get_val("history_heuristics.max_quiet_history")));
    out.push_str(&format!("  max_refutation_history: {}\n", get_val("history_heuristics.max_refutation_history")));
    out.push_str(&format!("  max_followup_history: {}\n", get_val("history_heuristics.max_followup_history")));
    out.push_str(&format!("  max_capture_history: {}\n", get_val("history_heuristics.max_capture_history")));

    out
}

pub fn write_search_config<P: AsRef<Path>>(values: &HashMap<String, f64>, output_path: P) -> Result<(), String> {
    let content = serialize_search_config(values);
    let p = output_path.as_ref();
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create parent directory for {}: {}", p.display(), e))?;
    }
    fs::write(p, content)
        .map_err(|e| format!("Failed to write search config to {}: {}", p.display(), e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_params_count_and_keys() {
        assert_eq!(SEARCH_PARAMS.len(), 39);
        let mut keys = std::collections::HashSet::new();
        for p in SEARCH_PARAMS {
            let k = p.full_key();
            assert!(keys.insert(k.clone()), "Duplicate parameter key: {}", k);
            assert!(p.min_val <= p.max_val, "Invalid bounds for {}", k);
            assert!(p.default_val >= p.min_val && p.default_val <= p.max_val, "Default out of bounds for {}", k);
        }
    }

    #[test]
    fn test_compute_discrete_perturbation_distinctness() {
        let ck = 0.5;
        for p in SEARCH_PARAMS {
            if !p.active {
                continue;
            }
            let key = p.full_key();
            for delta in [-1, 1] {
                let (plus, minus) = compute_discrete_perturbation(&key, p.default_val, delta, ck);
                assert!(plus >= p.min_val && plus <= p.max_val, "{}: plus {} out of bounds [{}, {}]", key, plus, p.min_val, p.max_val);
                assert!(minus >= p.min_val && minus <= p.max_val, "{}: minus {} out of bounds [{}, {}]", key, minus, p.min_val, p.max_val);
                assert!((plus - minus).abs() > 1e-6, "{}: plus and minus must not collapse to identical value!", key);
            }
        }
    }

    #[test]
    fn test_yaml_serialization_roundtrip() {
        let mut map = HashMap::new();
        for p in SEARCH_PARAMS {
            map.insert(p.full_key(), p.default_val);
        }
        let yaml_str = serialize_search_config(&map);
        let parsed: serde_yaml::Value = serde_yaml::from_str(&yaml_str).unwrap();
        assert!(parsed.get("aspiration_window").is_some());
        assert!(parsed.get("reverse_futility_pruning").is_some());
    }
}
