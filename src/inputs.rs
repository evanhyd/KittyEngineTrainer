use bulletformat::ChessBoard;
use bullet_lib::game::inputs::SparseInputType;

/// HalfKA input feature extractor matching KittyEngine's exact layout:
/// - Input size: 49,152 (64 king squares * 768 piece planes)
/// - 12 relative piece types: 6 friendly (P, N, B, R, Q, K) + 6 enemy (P, N, B, R, Q, K)
/// - Square coordinates: Rank 8 is square 0..7 (A8=0), Rank 1 is square 56..63 (A1=56)
#[derive(Clone, Copy, Debug, Default)]
pub struct KittyHalfKA;

impl SparseInputType for KittyHalfKA {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        49_152
    }

    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        // Convert bulletformat squares (where A1=0, A8=56) to KittyEngine coordinates (where A8=0, A1=56)
        let our_ksq = (pos.our_ksq() ^ 56) as usize;
        let opp_ksq = (pos.opp_ksq() ^ 56) as usize;

        for (piece, square) in pos.into_iter() {
            let piece_type = (piece & 7) as usize;
            let colour = usize::from((piece & 8) != 0); // 0 = STM (friendly), 1 = NTM (enemy)
            let sq = square as usize;

            // STM perspective: friendly = 0..5, enemy = 6..11; square flipped to KittyEngine coords (sq ^ 56)
            let stm_piece_idx = colour * 6 + piece_type;
            let stm_sq = sq ^ 56;
            let stm_feature = our_ksq * 768 + stm_piece_idx * 64 + stm_sq;

            // NTM perspective: friendly = 6..11 (from STM's view, enemy is friendly in NTM), enemy = 0..5
            // Square in NTM view is already (sq ^ 56) ^ 56 = sq
            let ntm_piece_idx = (1 - colour) * 6 + piece_type;
            let ntm_sq = sq;
            let ntm_feature = opp_ksq * 768 + ntm_piece_idx * 64 + ntm_sq;

            f(stm_feature, ntm_feature);
        }
    }

    fn shorthand(&self) -> String {
        "HalfKA-Kitty".to_string()
    }

    fn description(&self) -> String {
        "HalfKA (49152 features) matching KittyEngine exact board coordinates".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halfka_startpos_features() {
        let board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.5"
            .parse::<ChessBoard>()
            .expect("Valid start position");

        let mut stm_feats = Vec::new();
        let mut ntm_feats = Vec::new();
        KittyHalfKA.map_features(&board, |stm, ntm| {
            stm_feats.push(stm);
            ntm_feats.push(ntm);
        });

        // Start position has 32 pieces, all should be active in HalfKA
        assert_eq!(stm_feats.len(), 32);
        assert_eq!(ntm_feats.len(), 32);

        // Every feature index must be within [0, 49152)
        for &feat in &stm_feats {
            assert!(feat < 49_152, "stm feature {} exceeds 49151", feat);
        }
        for &feat in &ntm_feats {
            assert!(feat < 49_152, "ntm feature {} exceeds 49151", feat);
        }

        // White king on E1 in KittyEngine coords is square 60.
        // Stride is 768. Friendly king piece index is 5. Square is 60.
        // Expected white king feature = 60 * 768 + 5 * 64 + 60 = 46,080 + 320 + 60 = 46,460.
        assert!(stm_feats.contains(&46_460), "STM features must contain White king at 46,460");
    }

    #[test]
    fn test_halfka_custom_position() {
        let board = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1 | 0 | 0.5"
            .parse::<ChessBoard>()
            .expect("Valid position");

        let mut stm_feats = Vec::new();
        KittyHalfKA.map_features(&board, |stm, _ntm| {
            stm_feats.push(stm);
        });

        assert_eq!(stm_feats.len(), 3, "HalfKA must encode Pawn + White King + Black King");
        // White Pawn on E2: 60 * 768 + 0 * 64 + 52 = 46,132
        assert!(stm_feats.contains(&46_132));
        // White King on E1: 60 * 768 + 5 * 64 + 60 = 46,460
        assert!(stm_feats.contains(&46_460));
        // Black King on E8: 60 * 768 + 11 * 64 + 4 = 46,788
        assert!(stm_feats.contains(&46_788));
    }
}
