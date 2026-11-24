use std::io;
use std::path::Path;
use rayon::prelude::*;
use bulletformat::{BulletFormat, ChessBoard};

const MAGIC_V8: &[u8; 8] = b"NNUESGD8";

#[derive(Clone, Copy, Debug, Default)]
pub struct ErrorStats {
    pub count: u64,
    pub absolute: f64,
    pub squared: f64,
    pub bce: f64,
    pub target_sum: f64,
    pub target_sq_sum: f64,
}

impl ErrorStats {
    pub fn add(&mut self, prediction: f32, target: f32) {
        let err = (prediction as f64) - (target as f64);
        let p = (prediction as f64).clamp(1e-7, 1.0 - 1e-7);
        let t = (target as f64).clamp(0.0, 1.0);
        let bce_loss = -(t * p.ln() + (1.0 - t) * (1.0 - p).ln());
        self.count += 1;
        self.absolute += err.abs();
        self.squared += err * err;
        self.bce += bce_loss;
        self.target_sum += target as f64;
        self.target_sq_sum += (target as f64) * (target as f64);
    }

    pub fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.absolute += other.absolute;
        self.squared += other.squared;
        self.bce += other.bce;
        self.target_sum += other.target_sum;
        self.target_sq_sum += other.target_sq_sum;
    }

    pub fn mae(&self) -> f64 {
        if self.count > 0 { self.absolute / self.count as f64 } else { 0.0 }
    }

    pub fn mse(&self) -> f64 {
        if self.count > 0 { self.squared / self.count as f64 } else { 0.0 }
    }

    pub fn rmse(&self) -> f64 {
        self.mse().sqrt()
    }

    pub fn bce(&self) -> f64 {
        if self.count > 0 { self.bce / self.count as f64 } else { 0.0 }
    }

    pub fn smse(&self) -> f64 {
        if self.count <= 1 { return 0.0; }
        let n = self.count as f64;
        let mean = self.target_sum / n;
        let variance = (self.target_sq_sum / n) - (mean * mean);
        if variance > 1e-12 {
            self.mse() / variance
        } else {
            0.0
        }
    }
}

pub struct NnueWeights {
    #[allow(dead_code)]
    pub num_inputs: usize,
    pub accum_size: usize,
    pub hidden1_size: usize,
    pub hidden2_size: usize,
    pub num_buckets: usize,
    pub l0b: Vec<f32>,
    pub l0w: Vec<f32>,
    pub l1b: Vec<f32>,
    pub l1w: Vec<f32>,
    pub l2b: Vec<f32>,
    pub l2w: Vec<f32>,
    pub l3b: Vec<f32>,
    pub l3w: Vec<f32>,
}

impl NnueWeights {
    pub fn load_from_file(path: &Path) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() != crate::config::CHECKPOINT_SIZE as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Expected {} bytes (HalfKA-16 768->[16->32->1]x8), got {}",
                    crate::config::CHECKPOINT_SIZE,
                    bytes.len()
                ),
            ));
        }

        if &bytes[0..8] != MAGIC_V8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid NNUESGD signature (expected NNUESGD8)",
            ));
        }

        let num_inputs = crate::config::INPUT_SIZE;
        let accum_size = crate::config::ACCUM_SIZE;
        let pairwise_size = crate::config::PAIRWISE_SIZE;
        let hidden1_size = crate::config::HIDDEN1_SIZE;
        let hidden2_size = crate::config::HIDDEN2_SIZE;
        let num_buckets = crate::config::NUM_OUTPUT_BUCKETS;

        let payload_floats: &[f32] = unsafe {
            std::slice::from_raw_parts(bytes[72..].as_ptr() as *const f32, (bytes.len() - 72) / 4)
        };

        let mut offset = 0;
        let l0b = payload_floats[offset..offset + accum_size].to_vec();
        offset += accum_size;
        let l0w = payload_floats[offset..offset + num_inputs * accum_size].to_vec();
        offset += num_inputs * accum_size;
        let l1b = payload_floats[offset..offset + num_buckets * hidden1_size].to_vec();
        offset += num_buckets * hidden1_size;
        let l1w = payload_floats[offset..offset + num_buckets * hidden1_size * pairwise_size].to_vec();
        offset += num_buckets * hidden1_size * pairwise_size;
        let l2b = payload_floats[offset..offset + num_buckets * hidden2_size].to_vec();
        offset += num_buckets * hidden2_size;
        let l2w = payload_floats[offset..offset + num_buckets * hidden2_size * hidden1_size].to_vec();
        offset += num_buckets * hidden2_size * hidden1_size;
        let l3b = payload_floats[offset..offset + num_buckets].to_vec();
        offset += num_buckets;
        let l3w = payload_floats[offset..offset + num_buckets * hidden2_size].to_vec();

        Ok(Self {
            num_inputs,
            accum_size,
            hidden1_size,
            hidden2_size,
            num_buckets,
            l0b,
            l0w,
            l1b,
            l1w,
            l2b,
            l2w,
            l3b,
            l3w,
        })
    }

    #[inline(always)]
    fn crelu(x: f32) -> f32 {
        x.clamp(0.0, 1.0)
    }

    #[inline(always)]
    fn screlu(x: f32) -> f32 {
        let c = x.clamp(0.0, 1.0);
        c * c
    }

    pub fn evaluate_cp(&self, board: &ChessBoard) -> f32 {
        let mut stm_features = Vec::with_capacity(32);
        let mut ntm_features = Vec::with_capacity(32);

        crate::inputs::map_halfka_hm(board, |stm, ntm| {
            stm_features.push(stm);
            ntm_features.push(ntm);
        });

        let n = self.accum_size;
        let half = n / 2;
        let h1_n = self.hidden1_size;
        let h2_n = self.hidden2_size;

        let mut stm_acc = self.l0b.clone();
        for &feat in &stm_features {
            let row = &self.l0w[feat * n..(feat + 1) * n];
            for (acc, &w) in stm_acc.iter_mut().zip(row) {
                *acc += w;
            }
        }

        let mut ntm_acc = self.l0b.clone();
        for &feat in &ntm_features {
            let row = &self.l0w[feat * n..(feat + 1) * n];
            for (acc, &w) in ntm_acc.iter_mut().zip(row) {
                *acc += w;
            }
        }

        let mut pairwise = vec![0.0f32; n];
        for i in 0..half {
            pairwise[i] = Self::crelu(stm_acc[i]) * Self::crelu(stm_acc[half + i]);
            pairwise[half + i] = Self::crelu(ntm_acc[i]) * Self::crelu(ntm_acc[half + i]);
        }

        let occ = board.occ().count_ones() as usize;
        let bucket = ((occ.saturating_sub(2)) / 4).min(self.num_buckets - 1);

        let l1b_b = &self.l1b[bucket * h1_n..(bucket + 1) * h1_n];
        let l1w_b = &self.l1w[bucket * h1_n * n..(bucket + 1) * h1_n * n];
        let mut h1 = vec![0.0f32; h1_n];
        for r in 0..h1_n {
            let row = &l1w_b[r * n..(r + 1) * n];
            let mut sum = l1b_b[r];
            for (&p, &w) in pairwise.iter().zip(row) {
                sum += p * w;
            }
            h1[r] = Self::screlu(sum);
        }

        let l2b_b = &self.l2b[bucket * h2_n..(bucket + 1) * h2_n];
        let l2w_b = &self.l2w[bucket * h2_n * h1_n..(bucket + 1) * h2_n * h1_n];
        let mut h2 = vec![0.0f32; h2_n];
        for r in 0..h2_n {
            let row = &l2w_b[r * h1_n..(r + 1) * h1_n];
            let mut sum = l2b_b[r];
            for (&v, &w) in h1.iter().zip(row) {
                sum += v * w;
            }
            h2[r] = Self::screlu(sum);
        }

        let l3w_b = &self.l3w[bucket * h2_n..(bucket + 1) * h2_n];
        let mut logit = self.l3b[bucket];
        for (&v, &w) in h2.iter().zip(l3w_b) {
            logit += v * w;
        }

        logit * 400.0
    }

    pub fn evaluate(&self, board: &ChessBoard) -> f32 {
        let raw_eval = self.evaluate_cp(board);
        1.0 / (1.0 + (-raw_eval / 400.0).exp())
    }
}

pub fn evaluate_checkpoint_packs(
    weights_path: &Path,
    pack_paths: &[std::path::PathBuf],
    positions_per_pack: usize,
    wdl: f32,
) -> io::Result<(ErrorStats, ErrorStats, ErrorStats)> {
    let weights = NnueWeights::load_from_file(weights_path)?;

    let mut overall = ErrorStats::default();
    let mut non_mate = ErrorStats::default();
    let mut mate = ErrorStats::default();

    let num_packs = pack_paths.len();
    let batch_size = 131_072;

    for (pack_idx, pack_path) in pack_paths.iter().enumerate() {
        if !pack_path.exists() {
            continue;
        }

        crate::convert::for_each_board_in_pack(pack_path, batch_size, positions_per_pack, |batch| {
            let (batch_overall, batch_non_mate, batch_mate) = batch
                .par_chunks(8192)
                .map(|chunk| {
                    let mut chunk_overall = ErrorStats::default();
                    let mut chunk_non_mate = ErrorStats::default();
                    let mut chunk_mate = ErrorStats::default();

                    for board in chunk {
                        let pred = weights.evaluate(board);
                        let score = board.score as f32;
                        let score_sig = 1.0 / (1.0 + (-score / 400.0).exp());
                        let target = if wdl > 0.0 {
                            let result = (board.result_idx() as f32) / 2.0;
                            wdl * result + (1.0 - wdl) * score_sig
                        } else {
                            score_sig
                        };
                        let is_mate = score.abs() >= 2360.0;

                        chunk_overall.add(pred, target);
                        if is_mate {
                            chunk_mate.add(pred, target);
                        } else {
                            chunk_non_mate.add(pred, target);
                        }
                    }

                    (chunk_overall, chunk_non_mate, chunk_mate)
                })
                .reduce(
                    || (ErrorStats::default(), ErrorStats::default(), ErrorStats::default()),
                    |mut a, b| {
                        a.0.merge(&b.0);
                        a.1.merge(&b.1);
                        a.2.merge(&b.2);
                        a
                    },
                );

            overall.merge(&batch_overall);
            non_mate.merge(&batch_non_mate);
            mate.merge(&batch_mate);
            true
        })?;

        print!(
            "\r  Evaluating validation set ({}/{} packs)...",
            pack_idx + 1,
            num_packs
        );
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }

    print!("\r                                                      \r");
    Ok((overall, non_mate, mate))
}

#[derive(Clone, Debug, Default)]
pub struct InversionStats {
    pub total: u64,
    // Decisive targets
    pub count_target_ge_200: u64,
    pub count_target_ge_300: u64,
    pub count_target_ge_500: u64,
    pub count_target_ge_800: u64,
    pub count_target_mate: u64,

    // Complete polar inversions (|target| >= T && |pred| >= T && sign(target) != sign(pred))
    pub inv_200: u64,
    pub inv_300: u64,
    pub inv_500: u64,
    pub inv_800: u64,
    pub inv_mate: u64, // target mate (|target| >= 2360) and pred has opposite sign with |pred| >= 200

    // One-way polarity disagreements
    pub win_pred_loss_300: u64, // target >= +300, pred < 0
    pub loss_pred_win_300: u64, // target <= -300, pred > 0
    pub win_pred_loss_500: u64, // target >= +500, pred < 0
    pub loss_pred_win_500: u64, // target <= -500, pred > 0

    // Absolute error thresholds
    pub err_ge_300: u64,
    pub err_ge_500: u64,
    pub err_ge_800: u64,
    pub err_ge_1200: u64,

    // Outliers: (target_cp, pred_cp, error, description/string)
    pub worst_inversions: Vec<(f32, f32, f32, String)>,
}

impl InversionStats {
    pub fn merge(&mut self, other: &Self) {
        self.total += other.total;
        self.count_target_ge_200 += other.count_target_ge_200;
        self.count_target_ge_300 += other.count_target_ge_300;
        self.count_target_ge_500 += other.count_target_ge_500;
        self.count_target_ge_800 += other.count_target_ge_800;
        self.count_target_mate += other.count_target_mate;

        self.inv_200 += other.inv_200;
        self.inv_300 += other.inv_300;
        self.inv_500 += other.inv_500;
        self.inv_800 += other.inv_800;
        self.inv_mate += other.inv_mate;

        self.win_pred_loss_300 += other.win_pred_loss_300;
        self.loss_pred_win_300 += other.loss_pred_win_300;
        self.win_pred_loss_500 += other.win_pred_loss_500;
        self.loss_pred_win_500 += other.loss_pred_win_500;

        self.err_ge_300 += other.err_ge_300;
        self.err_ge_500 += other.err_ge_500;
        self.err_ge_800 += other.err_ge_800;
        self.err_ge_1200 += other.err_ge_1200;

        self.worst_inversions.extend(other.worst_inversions.iter().cloned());
        self.worst_inversions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        if self.worst_inversions.len() > 10 {
            self.worst_inversions.truncate(10);
        }
    }
}

pub fn board_to_fen(board: &ChessBoard) -> String {
    let mut grid = [' '; 64];

    for (piece, square) in board.into_iter() {
        let piece_type = (piece & 7) as usize;
        let is_enemy = (piece & 8) != 0;
        let ch = match piece_type {
            0 => if is_enemy { 'p' } else { 'P' },
            1 => if is_enemy { 'n' } else { 'N' },
            2 => if is_enemy { 'b' } else { 'B' },
            3 => if is_enemy { 'r' } else { 'R' },
            4 => if is_enemy { 'q' } else { 'Q' },
            5 => if is_enemy { 'k' } else { 'K' },
            _ => continue,
        };
        grid[square as usize] = ch;
    }

    let mut fen = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            let sq = rank * 8 + file;
            let ch = grid[sq];
            if ch == ' ' {
                empty += 1;
            } else {
                if empty > 0 {
                    fen.push_str(&empty.to_string());
                    empty = 0;
                }
                fen.push(ch);
            }
        }
        if empty > 0 {
            fen.push_str(&empty.to_string());
        }
        if rank > 0 {
            fen.push('/');
        }
    }
    fen.push_str(" w - - 0 1");
    fen
}

pub fn evaluate_inversions(
    weights_path: &Path,
    pack_paths: &[std::path::PathBuf],
    positions_per_pack: usize,
) -> io::Result<InversionStats> {
    let weights = NnueWeights::load_from_file(weights_path)?;
    let mut stats = InversionStats::default();

    let num_packs = pack_paths.len();
    let batch_size = 131_072;

    for (pack_idx, pack_path) in pack_paths.iter().enumerate() {
        if !pack_path.exists() {
            continue;
        }

        let mut evaluated_in_pack = 0;

        crate::convert::for_each_board_in_pack(pack_path, batch_size, positions_per_pack, |batch| {
            let batch_stats = batch
                .par_chunks(8192)
                .map(|chunk| {
                    let mut chunk_stats = InversionStats::default();

                    for board in chunk {
                        let target_cp = board.score as f32;
                        let pred_cp = weights.evaluate_cp(board);
                        let err = (pred_cp - target_cp).abs();

                        chunk_stats.total += 1;
                        let t_abs = target_cp.abs();
                        let p_abs = pred_cp.abs();
                        let opposite_sign = (target_cp > 0.0 && pred_cp < 0.0) || (target_cp < 0.0 && pred_cp > 0.0);

                        if t_abs >= 200.0 { chunk_stats.count_target_ge_200 += 1; }
                        if t_abs >= 300.0 { chunk_stats.count_target_ge_300 += 1; }
                        if t_abs >= 500.0 { chunk_stats.count_target_ge_500 += 1; }
                        if t_abs >= 800.0 { chunk_stats.count_target_ge_800 += 1; }
                        if t_abs >= 2360.0 { chunk_stats.count_target_mate += 1; }

                        if opposite_sign {
                            if t_abs >= 200.0 && p_abs >= 200.0 { chunk_stats.inv_200 += 1; }
                            if t_abs >= 300.0 && p_abs >= 300.0 { chunk_stats.inv_300 += 1; }
                            if t_abs >= 500.0 && p_abs >= 500.0 { chunk_stats.inv_500 += 1; }
                            if t_abs >= 800.0 && p_abs >= 800.0 { chunk_stats.inv_800 += 1; }
                            if t_abs >= 2360.0 && p_abs >= 200.0 { chunk_stats.inv_mate += 1; }
                        }

                        if target_cp >= 300.0 && pred_cp < 0.0 { chunk_stats.win_pred_loss_300 += 1; }
                        if target_cp <= -300.0 && pred_cp > 0.0 { chunk_stats.loss_pred_win_300 += 1; }
                        if target_cp >= 500.0 && pred_cp < 0.0 { chunk_stats.win_pred_loss_500 += 1; }
                        if target_cp <= -500.0 && pred_cp > 0.0 { chunk_stats.loss_pred_win_500 += 1; }

                        if err >= 300.0 { chunk_stats.err_ge_300 += 1; }
                        if err >= 500.0 { chunk_stats.err_ge_500 += 1; }
                        if err >= 800.0 { chunk_stats.err_ge_800 += 1; }
                        if err >= 1200.0 { chunk_stats.err_ge_1200 += 1; }

                        if opposite_sign && (300.0..2000.0).contains(&t_abs) && p_abs >= 300.0 && err >= 600.0
                            && chunk_stats.worst_inversions.len() < 5 {
                                chunk_stats.worst_inversions.push((target_cp, pred_cp, err, board_to_fen(board)));
                            }
                    }

                    chunk_stats
                })
                .reduce(
                    InversionStats::default,
                    |mut a, b| {
                        a.merge(&b);
                        a
                    },
                );

            stats.merge(&batch_stats);
            evaluated_in_pack += batch.len();
            true
        })?;

        print!(
            "\rPack {}/{} ('{}'): evaluated {} pos | Total across all packs: {} pos...",
            pack_idx + 1,
            num_packs,
            pack_path.file_name().unwrap_or_default().to_string_lossy(),
            evaluated_in_pack,
            stats.total
        );
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }

    println!();
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_to_fen() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.5";
        let board = fen.parse::<ChessBoard>().unwrap();
        let gen_fen = board_to_fen(&board);
        assert_eq!(gen_fen, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");
    }
}
