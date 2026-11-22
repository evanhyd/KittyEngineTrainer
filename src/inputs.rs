use bulletformat::ChessBoard;
use bullet_lib::game::inputs::SparseInputType;
use bullet_lib::game::outputs::OutputBuckets;

/// HalfKA input feature extractor with horizontal mirror folding (HalfKAv2 HM):
/// - Input size: 24,576 (32 folded king squares * 768 piece planes)
/// - Files E-H fold to A-D (flipMask = 7 when file > 3).
#[derive(Clone, Copy, Debug, Default)]
pub struct KittyHalfKAHM;

impl SparseInputType for KittyHalfKAHM {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        24_576
    }

    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        map_halfka_hm(pos, f);
    }

    fn shorthand(&self) -> String {
        "HalfKAv2-hm".to_string()
    }

    fn description(&self) -> String {
        "HalfKAv2 with horizontal mirror folding (24576 features, 32 folded king squares x 768 planes) matching KittyEngine coordinates".to_string()
    }
}

pub fn map_halfka_hm<F: FnMut(usize, usize)>(pos: &ChessBoard, mut f: F) {
    let our_ksq = (pos.our_ksq() ^ 56) as usize;
    let opp_ksq = (pos.opp_ksq() ^ 56) as usize;

    let our_flip = if (our_ksq & 7) > 3 { 7 } else { 0 };
    let our_folded_k = our_ksq ^ our_flip;
    let our_k_idx = ((our_folded_k >> 3) * 4) + (our_folded_k & 7);

    let opp_flip = if (opp_ksq & 7) > 3 { 7 } else { 0 };
    let opp_folded_k = opp_ksq ^ opp_flip;
    let opp_k_idx = ((opp_folded_k >> 3) * 4) + (opp_folded_k & 7);

    for (piece, square) in pos.into_iter() {
        let piece_type = (piece & 7) as usize;
        let colour = usize::from((piece & 8) != 0);
        let sq = square as usize;

        let stm_piece_idx = colour * 6 + piece_type;
        let stm_sq = (sq ^ 56) ^ our_flip;
        let stm_feature = our_k_idx * 768 + stm_piece_idx * 64 + stm_sq;

        let ntm_piece_idx = (1 - colour) * 6 + piece_type;
        let ntm_sq = sq ^ opp_flip;
        let ntm_feature = opp_k_idx * 768 + ntm_piece_idx * 64 + ntm_sq;

        f(stm_feature, ntm_feature);
    }
}

/// Major and minor piece output bucketing (excludes pawns and kings):
/// Counts Knights (1), Bishops (2), Rooks (3), Queens (4) on board (0..=16).
/// Bucket formula: min(N - 1, count / 2).
#[derive(Clone, Copy, Debug, Default)]
pub struct MajorMinorCount<const N: usize>;

impl<const N: usize> OutputBuckets<ChessBoard> for MajorMinorCount<N> {
    const BUCKETS: usize = N;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let mut count = 0;
        for (piece, _) in pos.into_iter() {
            let piece_type = piece & 7;
            if (1..=4).contains(&piece_type) {
                count += 1;
            }
        }
        std::cmp::min((N - 1) as u8, (count / 2) as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halfka_hm_startpos_features() {
        let board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.5"
            .parse::<ChessBoard>()
            .expect("Valid start position");

        let mut stm_feats = Vec::new();
        let mut ntm_feats = Vec::new();
        KittyHalfKAHM.map_features(&board, |stm, ntm| {
            stm_feats.push(stm);
            ntm_feats.push(ntm);
        });

        // Start position has 32 pieces, all should be active in HalfKA HM
        assert_eq!(stm_feats.len(), 32);
        assert_eq!(ntm_feats.len(), 32);

        // Every feature index must be within [0, 24576)
        for &feat in &stm_feats {
            assert!(feat < 24_576, "stm feature {} exceeds 24575", feat);
        }
        for &feat in &ntm_feats {
            assert!(feat < 24_576, "ntm feature {} exceeds 24575", feat);
        }

        // White king on E1 in folded coords:
        // Rank 7 folded idx = 31. 31 * 768 + 5 * 64 + 59 = 24,187
        assert!(stm_feats.contains(&24_187), "STM features must contain White king at 24,187");

        // Verify against KittyEngine C++ GoogleTest (evaluation_test.cpp L335-336):
        // Position: "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
        // Expected: white[0] = 23,859 (White Pawn at E2 from White perspective)
        //           black[1] = 24,203 (White Pawn at E2 from Black perspective)
        let test_pos = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1 | 0 | 0.5".parse::<ChessBoard>().unwrap();
        let mut test_stm = Vec::new();
        let mut test_ntm = Vec::new();
        KittyHalfKAHM.map_features(&test_pos, |stm, ntm| {
            test_stm.push(stm);
            test_ntm.push(ntm);
        });
        assert!(test_stm.contains(&23_859), "STM features must contain White pawn at 23,859");
        assert!(test_ntm.contains(&24_203), "NTM features must contain White pawn at 24,203");
    }

    #[test]
    fn test_inspect_binpack_samples() {
        use std::path::Path;
        use bulletformat::BulletFormat;

        let binpack_path = Path::new("data/pack1.binpack");
        if !binpack_path.exists() {
            println!("Pack1 not found, skipping");
            return;
        }

        let mut count = 0;
        let mut scores = Vec::new();
        let mut results = Vec::new();
        let mut total_pieces_bucket_counts = [0usize; 8];
        let mut mm_bucket_counts = [0usize; 8];

        let mm = MajorMinorCount::<8>;

        let _ = crate::convert::for_each_board_in_pack(binpack_path, 16384, 1000, |batch| {
            for board in batch {
                count += 1;
                scores.push(board.score);
                results.push(board.result_idx());
                let occ = board.occ().count_ones() as usize;
                let b_tp = ((occ.saturating_sub(2)) / 4).min(7);
                let b_mm = mm.bucket(board) as usize;
                total_pieces_bucket_counts[b_tp] += 1;
                mm_bucket_counts[b_mm] += 1;
            }
            true
        });

        println!("\n=== BUCKET DISTRIBUTION COMPARISON (1000 positions) ===");
        println!("(Total Pieces - 2) / 4: {:?}", total_pieces_bucket_counts);
        println!("Major/Minor Pieces / 2: {:?}", mm_bucket_counts);
    }

    #[test]
    fn test_inspect_bucket_weights() {
        use std::path::Path;
        use crate::eval::NnueWeights;

        let p = Path::new("checkpoints/champion_weights.bin");
        if !p.exists() {
            println!("champion_weights.bin not found, skipping inspection");
            return;
        }
        let w = match NnueWeights::load_from_file(p) {
            Ok(w) => w,
            Err(e) => {
                println!("Skipping inspection (different architecture or invalid file): {}", e);
                return;
            }
        };

        println!("\n=== OUTPUT BUCKET BIASES & WEIGHT NORMS ===");
        for b in 0..8 {
            let bias = w.l3b[b];
            let weights = &w.l3w[b * 32..(b + 1) * 32];
            let l2_norm: f32 = weights.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mean: f32 = weights.iter().sum::<f32>() / 32.0;
            println!("Bucket {}: bias = {:8.2} cp, weight L2 norm = {:7.2}, mean = {:7.2}", b, bias, l2_norm, mean);
        }
    }

    #[test]
    fn test_major_minor_count_buckets() {
        use bullet_lib::game::outputs::OutputBuckets;
        let mm = MajorMinorCount::<8>;

        // Start position has 14 major/minor pieces (4N + 4B + 4R + 2Q) -> 14 / 2 = 7 (bucket 7)
        let start_pos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.5"
            .parse::<ChessBoard>().unwrap();
        assert_eq!(mm.bucket(&start_pos), 7u8);

        // Kings only -> 0 major/minors -> bucket 0
        let kings_only = "4k3/8/8/8/8/8/8/4K3 w - - 0 1 | 0 | 0.5"
            .parse::<ChessBoard>().unwrap();
        assert_eq!(mm.bucket(&kings_only), 0u8);

        // K + P endgame -> 0 major/minors -> bucket 0
        let kp_endgame = "4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1 | 0 | 0.5"
            .parse::<ChessBoard>().unwrap();
        assert_eq!(mm.bucket(&kp_endgame), 0u8);

        // K + R vs K + R -> 2 major/minors -> 2 / 2 = 1 (bucket 1)
        let rook_endgame = "4k3/4r3/8/8/8/8/4R3/4K3 w - - 0 1 | 0 | 0.5"
            .parse::<ChessBoard>().unwrap();
        assert_eq!(mm.bucket(&rook_endgame), 1u8);

        // 2 Knights + 2 Bishops = 4 -> bucket 2
        let minor_endgame = "1nb1k3/8/8/8/8/8/8/1NB1K3 w - - 0 1 | 0 | 0.5"
            .parse::<ChessBoard>().unwrap();
        assert_eq!(mm.bucket(&minor_endgame), 2u8);
    }
}
