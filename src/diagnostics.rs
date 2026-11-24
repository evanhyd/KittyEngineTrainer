use std::io;
use std::path::{Path, PathBuf};
use rayon::prelude::*;
use crate::eval::NnueWeights;

#[derive(Clone, Debug)]
pub struct NeuronActivity {
    pub count: u64,
    pub fires: u64,
    pub sum_act: f64,
    pub min_pre: f32,
    pub max_pre: f32,
    pub min_act: f32,
    pub max_act: f32,
}

impl Default for NeuronActivity {
    fn default() -> Self {
        Self {
            count: 0,
            fires: 0,
            sum_act: 0.0,
            min_pre: f32::MAX,
            max_pre: f32::MIN,
            min_act: f32::MAX,
            max_act: f32::MIN,
        }
    }
}

impl NeuronActivity {
    pub fn record(&mut self, pre: f32, act: f32) {
        self.count += 1;
        if act > 0.0 {
            self.fires += 1;
        }
        self.sum_act += act as f64;
        if pre < self.min_pre { self.min_pre = pre; }
        if pre > self.max_pre { self.max_pre = pre; }
        if act < self.min_act { self.min_act = act; }
        if act > self.max_act { self.max_act = act; }
    }

    pub fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.fires += other.fires;
        self.sum_act += other.sum_act;
        if other.min_pre < self.min_pre { self.min_pre = other.min_pre; }
        if other.max_pre > self.max_pre { self.max_pre = other.max_pre; }
        if other.min_act < self.min_act { self.min_act = other.min_act; }
        if other.max_act > self.max_act { self.max_act = other.max_act; }
    }

    pub fn firing_rate(&self) -> f64 {
        if self.count > 0 { (self.fires as f64 / self.count as f64) * 100.0 } else { 0.0 }
    }

    pub fn mean_activation(&self) -> f64 {
        if self.count > 0 { self.sum_act / self.count as f64 } else { 0.0 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LayerDiagnostics {
    pub neurons: Vec<NeuronActivity>,
}

impl LayerDiagnostics {
    pub fn new(size: usize) -> Self {
        Self { neurons: vec![NeuronActivity::default(); size] }
    }

    pub fn merge(&mut self, other: &Self) {
        for (a, b) in self.neurons.iter_mut().zip(&other.neurons) {
            a.merge(b);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NetworkDiagnostics {
    pub total_positions: u64,
    pub accum_combined: LayerDiagnostics,
    pub l1: LayerDiagnostics,
    pub l2: LayerDiagnostics,
}

impl NetworkDiagnostics {
    pub fn new(accum_size: usize, l1_total: usize, l2_total: usize) -> Self {
        Self {
            total_positions: 0,
            accum_combined: LayerDiagnostics::new(accum_size),
            l1: LayerDiagnostics::new(l1_total),
            l2: LayerDiagnostics::new(l2_total),
        }
    }

    pub fn merge(&mut self, other: &Self) {
        self.total_positions += other.total_positions;
        self.accum_combined.merge(&other.accum_combined);
        self.l1.merge(&other.l1);
        self.l2.merge(&other.l2);
    }
}

pub fn analyze_weights(
    weights_path: &Path,
    pack_paths: &[PathBuf],
    positions_per_pack: usize,
) -> io::Result<NetworkDiagnostics> {
    let weights = NnueWeights::load_from_file(weights_path)?;
    let n = weights.accum_size;
    let half = n / 2;
    let h1_n = weights.hidden1_size;
    let h2_n = weights.hidden2_size;
    let b_n = weights.num_buckets;

    let mut full_diag = NetworkDiagnostics::new(n, b_n * h1_n, b_n * h2_n);
    let batch_size = 65_536;

    for (pack_idx, pack_path) in pack_paths.iter().enumerate() {
        if !pack_path.exists() {
            continue;
        }

        print!("\rSampling eval packs: {}/{}...", pack_idx + 1, pack_paths.len());
        use std::io::Write;
        let _ = std::io::stdout().flush();

        crate::convert::for_each_board_in_pack(pack_path, batch_size, positions_per_pack, |batch| {
            let batch_diag = batch
                .par_chunks(4096)
                .map(|chunk| {
                    let mut chunk_diag = NetworkDiagnostics::new(n, b_n * h1_n, b_n * h2_n);
                    chunk_diag.total_positions = chunk.len() as u64;

                    let mut pairwise = vec![0.0f32; n];
                    let mut h1 = vec![0.0f32; h1_n];

                    for board in chunk {
                        let mut stm_features = Vec::with_capacity(32);
                        let mut ntm_features = Vec::with_capacity(32);
                        crate::inputs::map_halfka_hm(board, |stm, ntm| {
                            stm_features.push(stm);
                            ntm_features.push(ntm);
                        });

                        let mut stm_acc = weights.l0b.clone();
                        for &feat in &stm_features {
                            let row = &weights.l0w[feat * n..(feat + 1) * n];
                            for (acc, &w) in stm_acc.iter_mut().zip(row) {
                                *acc += w;
                            }
                        }

                        let mut ntm_acc = weights.l0b.clone();
                        for &feat in &ntm_features {
                            let row = &weights.l0w[feat * n..(feat + 1) * n];
                            for (acc, &w) in ntm_acc.iter_mut().zip(row) {
                                *acc += w;
                            }
                        }

                        for i in 0..n {
                            let s_act = stm_acc[i].clamp(0.0, 1.0);
                            let n_act = ntm_acc[i].clamp(0.0, 1.0);
                            chunk_diag.accum_combined.neurons[i].record(stm_acc[i].max(ntm_acc[i]), s_act.max(n_act));
                        }

                        for i in 0..half {
                            pairwise[i] = stm_acc[i].clamp(0.0, 1.0) * stm_acc[half + i].clamp(0.0, 1.0);
                            pairwise[half + i] = ntm_acc[i].clamp(0.0, 1.0) * ntm_acc[half + i].clamp(0.0, 1.0);
                        }

                        let occ = board.occ().count_ones() as usize;
                        let bucket = ((occ.saturating_sub(2)) / 4).min(b_n - 1);

                        let l1b_b = &weights.l1b[bucket * h1_n..(bucket + 1) * h1_n];
                        let l1w_b = &weights.l1w[bucket * h1_n * n..(bucket + 1) * h1_n * n];
                        for r in 0..h1_n {
                            let row = &l1w_b[r * n..(r + 1) * n];
                            let mut sum = l1b_b[r];
                            for (&p, &w) in pairwise.iter().zip(row) {
                                sum += p * w;
                            }
                            let c = sum.clamp(0.0, 1.0);
                            let act = c * c;
                            h1[r] = act;
                            chunk_diag.l1.neurons[bucket * h1_n + r].record(sum, act);
                        }

                        let l2b_b = &weights.l2b[bucket * h2_n..(bucket + 1) * h2_n];
                        let l2w_b = &weights.l2w[bucket * h2_n * h1_n..(bucket + 1) * h2_n * h1_n];
                        for r in 0..h2_n {
                            let row = &l2w_b[r * h1_n..(r + 1) * h1_n];
                            let mut sum = l2b_b[r];
                            for (&v, &w) in h1.iter().zip(row) {
                                sum += v * w;
                            }
                            let c = sum.clamp(0.0, 1.0);
                            let act = c * c;
                            chunk_diag.l2.neurons[bucket * h2_n + r].record(sum, act);
                        }
                    }

                    chunk_diag
                })
                .reduce(|| NetworkDiagnostics::new(n, b_n * h1_n, b_n * h2_n), |mut a, b| {
                    a.merge(&b);
                    a
                });

            full_diag.merge(&batch_diag);
            true
        })?;
    }
    println!();

    Ok(full_diag)
}
