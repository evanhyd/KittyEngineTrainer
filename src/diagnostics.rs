use std::io;
use std::path::{Path, PathBuf};
use rayon::prelude::*;
use bulletformat::{ChessBoard, DataLoader};
use crate::inputs::KittyHalfKA;
use bullet_lib::game::inputs::SparseInputType;
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
        if self.count > 0 {
            (self.fires as f64 / self.count as f64) * 100.0
        } else {
            0.0
        }
    }

    pub fn mean_activation(&self) -> f64 {
        if self.count > 0 {
            self.sum_act / self.count as f64
        } else {
            0.0
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LayerDiagnostics {
    pub neurons: Vec<NeuronActivity>,
}

impl LayerDiagnostics {
    pub fn new(size: usize) -> Self {
        Self {
            neurons: vec![NeuronActivity::default(); size],
        }
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
    pub accum_stm: LayerDiagnostics,
    pub accum_ntm: LayerDiagnostics,
    pub accum_combined: LayerDiagnostics,
    pub l1: LayerDiagnostics,
    pub l2: LayerDiagnostics,
}

impl NetworkDiagnostics {
    pub fn new() -> Self {
        Self {
            total_positions: 0,
            accum_stm: LayerDiagnostics::new(256),
            accum_ntm: LayerDiagnostics::new(256),
            accum_combined: LayerDiagnostics::new(256),
            l1: LayerDiagnostics::new(32),
            l2: LayerDiagnostics::new(32),
        }
    }

    pub fn merge(&mut self, other: &Self) {
        self.total_positions += other.total_positions;
        self.accum_stm.merge(&other.accum_stm);
        self.accum_ntm.merge(&other.accum_ntm);
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

    let mut full_diag = NetworkDiagnostics::new();
    let batch_size = 65_536;

    for (pack_idx, pack_path) in pack_paths.iter().enumerate() {
        if !pack_path.exists() {
            continue;
        }

        let pack_name = pack_path.file_name().unwrap_or_default().to_string_lossy();
        println!("Sampling pack {}/{} ('{}')...", pack_idx + 1, pack_paths.len(), pack_name);

        let loader = DataLoader::<ChessBoard>::new(pack_path, 64)?;
        let mut processed_in_pack = 0;

        loader.map_batches(batch_size, |batch| {
            if processed_in_pack >= positions_per_pack {
                return;
            }

            let slice = if processed_in_pack + batch.len() > positions_per_pack {
                &batch[..positions_per_pack - processed_in_pack]
            } else {
                batch
            };

            let batch_diag = slice
                .par_chunks(4096)
                .map(|chunk| {
                    let mut chunk_diag = NetworkDiagnostics::new();
                    chunk_diag.total_positions = chunk.len() as u64;

                    for board in chunk {
                        let mut stm_features = Vec::with_capacity(32);
                        let mut ntm_features = Vec::with_capacity(32);

                        KittyHalfKA.map_features(board, |stm, ntm| {
                            stm_features.push(stm);
                            ntm_features.push(ntm);
                        });

                        // 1. Accumulator forward pass
                        let mut stm_acc = weights.l0b.clone();
                        for &feat in &stm_features {
                            let row = &weights.l0w[feat * 256..(feat + 1) * 256];
                            for (acc, &w) in stm_acc.iter_mut().zip(row) {
                                *acc += w;
                            }
                        }

                        let mut ntm_acc = weights.l0b.clone();
                        for &feat in &ntm_features {
                            let row = &weights.l0w[feat * 256..(feat + 1) * 256];
                            for (acc, &w) in ntm_acc.iter_mut().zip(row) {
                                *acc += w;
                            }
                        }

                        let mut combined = [0.0f32; 512];
                        for i in 0..256 {
                            let stm_pre = stm_acc[i];
                            let ntm_pre = ntm_acc[i];

                            let c_stm = stm_pre.clamp(0.0, 1.0);
                            let stm_act = c_stm * c_stm;

                            let c_ntm = ntm_pre.clamp(0.0, 1.0);
                            let ntm_act = c_ntm * c_ntm;

                            combined[i] = stm_act;
                            combined[256 + i] = ntm_act;

                            chunk_diag.accum_stm.neurons[i].record(stm_pre, stm_act);
                            chunk_diag.accum_ntm.neurons[i].record(ntm_pre, ntm_act);

                            let max_act = stm_act.max(ntm_act);
                            let max_pre = stm_pre.max(ntm_pre);
                            chunk_diag.accum_combined.neurons[i].record(max_pre, max_act);
                        }

                        // 2. Hidden Layer 1 forward pass
                        let mut h1 = [0.0f32; 32];
                        for i in 0..32 {
                            let row = &weights.l1w[i * 512..(i + 1) * 512];
                            let mut dot = weights.l1b[i];
                            for (c, &w) in combined.iter().zip(row) {
                                dot += c * w;
                            }

                            let c = dot.clamp(0.0, 1.0);
                            let act = c * c;

                            h1[i] = act;
                            chunk_diag.l1.neurons[i].record(dot, act);
                        }

                        // 3. Hidden Layer 2 forward pass
                        let mut h2 = [0.0f32; 32];
                        for j in 0..32 {
                            let row = &weights.l2w[j * 32..(j + 1) * 32];
                            let mut dot = weights.l2b[j];
                            for (h, &w) in h1.iter().zip(row) {
                                dot += h * w;
                            }

                            let c = dot.clamp(0.0, 1.0);
                            let act = c * c;

                            h2[j] = act;
                            chunk_diag.l2.neurons[j].record(dot, act);
                        }
                    }

                    chunk_diag
                })
                .reduce(NetworkDiagnostics::new, |mut a, b| {
                    a.merge(&b);
                    a
                });

            full_diag.merge(&batch_diag);
            processed_in_pack += slice.len();
        });
    }

    Ok(full_diag)
}
