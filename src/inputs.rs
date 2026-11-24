use bulletformat::ChessBoard;
use bullet_lib::game::inputs::SparseInputType;
use bullet_lib::game::outputs::OutputBuckets;

/// Maps 32 horizontally-folded king squares (rank 8..1 top-to-bottom, files A..D) into 16 king buckets.
/// Must match KittyEngine C++ `NnueLayout::kKingBuckets` identically.
pub const KING_BUCKETS: [usize; 32] = [
    14, 14, 15, 15, // Rank 8 (foldedRank 0)
    14, 14, 15, 15, // Rank 7 (foldedRank 1)
    12, 12, 13, 13, // Rank 6 (foldedRank 2)
    12, 12, 13, 13, // Rank 5 (foldedRank 3)
    10, 10, 11, 11, // Rank 4 (foldedRank 4)
     8,  8,  9,  9, // Rank 3 (foldedRank 5)
     4,  5,  6,  7, // Rank 2 (foldedRank 6)
     0,  1,  2,  3, // Rank 1 (foldedRank 7)
];

/// HalfKA-16 input feature extractor with horizontal mirror folding (HalfKAv2-16 HM):
/// - Input size: 12,288 (16 king buckets * 768 piece planes)
/// - Files E-H fold to A-D (flipMask = 7 when file > 3).
#[derive(Clone, Copy, Debug, Default)]
pub struct KittyHalfKAHM;

impl SparseInputType for KittyHalfKAHM {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        12_288
    }

    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        map_halfka_hm(pos, f);
    }

    fn shorthand(&self) -> String {
        "HalfKAv2-16-hm".to_string()
    }

    fn description(&self) -> String {
        "HalfKAv2-16 with horizontal mirror folding (12288 features, 16 king buckets x 768 planes) matching KittyEngine coordinates".to_string()
    }
}

pub fn map_halfka_hm<F: FnMut(usize, usize)>(pos: &ChessBoard, mut f: F) {
    let our_ksq = (pos.our_ksq() ^ 56) as usize;
    let opp_ksq = (pos.opp_ksq() ^ 56) as usize;

    let our_flip = if (our_ksq & 7) > 3 { 7 } else { 0 };
    let our_folded_k = our_ksq ^ our_flip;
    let our_k_idx = KING_BUCKETS[((our_folded_k >> 3) * 4) + (our_folded_k & 7)];

    let opp_flip = if (opp_ksq & 7) > 3 { 7 } else { 0 };
    let opp_folded_k = opp_ksq ^ opp_flip;
    let opp_k_idx = KING_BUCKETS[((opp_folded_k >> 3) * 4) + (opp_folded_k & 7)];

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

/// Total-piece output bucketing (includes pawns and kings):
/// Bucket formula: min(N - 1, (total_pieces - 2) / 4).
#[derive(Clone, Copy, Debug, Default)]
pub struct TotalPieceCount<const N: usize>;

impl<const N: usize> OutputBuckets<ChessBoard> for TotalPieceCount<N> {
    const BUCKETS: usize = N;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let occ = pos.occ().count_ones() as usize;
        ((occ.saturating_sub(2)) / 4).min(N - 1) as u8
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

        assert_eq!(stm_feats.len(), 32);
        assert_eq!(ntm_feats.len(), 32);

        for &feat in &stm_feats {
            assert!(feat < 12_288, "stm feature {} exceeds 12287", feat);
        }
        for &feat in &ntm_feats {
            assert!(feat < 12_288, "ntm feature {} exceeds 12287", feat);
        }

        // White king on E1 folds to D1 -> bucket 3 -> base 2304.
        // King plane = 5 * 64 + 59 = 379 -> 2304 + 379 = 2683.
        assert!(stm_feats.contains(&2_683), "STM features must contain White king at 2,683");

        // Verify against KittyEngine C++ GoogleTest (evaluation_test.cpp):
        // Position: "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
        // Expected: white[0] = 2,355 (White Pawn at E2 from White perspective)
        //           black[1] = 2,699 (White Pawn at E2 from Black perspective)
        let test_pos = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1 | 0 | 0.5".parse::<ChessBoard>().unwrap();
        let mut test_stm = Vec::new();
        let mut test_ntm = Vec::new();
        KittyHalfKAHM.map_features(&test_pos, |stm, ntm| {
            test_stm.push(stm);
            test_ntm.push(ntm);
        });
        assert!(test_stm.contains(&2_355), "STM features must contain White pawn at 2,355");
        assert!(test_ntm.contains(&2_699), "NTM features must contain White pawn at 2,699");
    }

    #[test]
    fn test_total_piece_count_buckets() {
        let tp = TotalPieceCount::<8>;

        // Start position: 32 pieces -> (32 - 2) / 4 = 7
        let start_pos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.5"
            .parse::<ChessBoard>().unwrap();
        assert_eq!(tp.bucket(&start_pos), 7u8);

        // Kings only: 2 pieces -> 0
        let kings_only = "4k3/8/8/8/8/8/8/4K3 w - - 0 1 | 0 | 0.5"
            .parse::<ChessBoard>().unwrap();
        assert_eq!(tp.bucket(&kings_only), 0u8);

        // 16 pawns + 2 kings = 18 pieces -> (18 - 2) / 4 = 4
        let pawn_endgame = "4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1 | 0 | 0.5"
            .parse::<ChessBoard>().unwrap();
        assert_eq!(tp.bucket(&pawn_endgame), 4u8);

        // K + R vs K + R + 2 pawns = 6 pieces -> (6 - 2) / 4 = 1
        let rook_endgame = "4k3/4rp2/8/8/8/8/4RP2/4K3 w - - 0 1 | 0 | 0.5"
            .parse::<ChessBoard>().unwrap();
        assert_eq!(tp.bucket(&rook_endgame), 1u8);
    }
}
