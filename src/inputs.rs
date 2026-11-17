use bulletformat::ChessBoard;
use bullet_lib::game::inputs::SparseInputType;

/// HalfKA input feature extractor with horizontal symmetry mirroring (HalfKA-hm):
/// - Input size: 24,576 (32 folded king squares * 768 piece planes)
/// - 12 relative piece types: 6 friendly (P, N, B, R, Q, K) + 6 enemy (P, N, B, R, Q, K)
/// - Square coordinates: Rank 8 is square 0..7 (A8=0), Rank 1 is square 56..63 (A1=56)
/// - Files A-D are canonical (file <= 3). If king is on files E-H (file > 3), horizontally mirror (sq ^ 7).
#[derive(Clone, Copy, Debug, Default)]
pub struct KittyHalfKA;

impl SparseInputType for KittyHalfKA {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        24_576
    }

    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        // Convert bulletformat squares (where A1=0, A8=56) to KittyEngine coordinates (where A8=0, A1=56)
        let our_ksq = (pos.our_ksq() ^ 56) as usize;
        let opp_ksq = (pos.opp_ksq() ^ 56) as usize;

        let stm_flip = if (our_ksq & 7) > 3 { 7 } else { 0 };
        let our_ksq_folded = our_ksq ^ stm_flip;
        let our_king_idx = ((our_ksq_folded >> 3) * 4) + (our_ksq_folded & 7);

        let ntm_flip = if (opp_ksq & 7) > 3 { 7 } else { 0 };
        let opp_ksq_folded = opp_ksq ^ ntm_flip;
        let opp_king_idx = ((opp_ksq_folded >> 3) * 4) + (opp_ksq_folded & 7);

        for (piece, square) in pos.into_iter() {
            let piece_type = (piece & 7) as usize;
            let colour = usize::from((piece & 8) != 0); // 0 = STM (friendly), 1 = NTM (enemy)
            let sq = square as usize;

            // STM perspective: friendly = 0..5, enemy = 6..11; square flipped to KittyEngine coords (sq ^ 56) and mirrored
            let stm_piece_idx = colour * 6 + piece_type;
            let stm_sq = (sq ^ 56) ^ stm_flip;
            let stm_feature = our_king_idx * 768 + stm_piece_idx * 64 + stm_sq;

            // NTM perspective: friendly = 6..11 (from STM's view, enemy is friendly in NTM), enemy = 0..5
            let ntm_piece_idx = (1 - colour) * 6 + piece_type;
            let ntm_sq = sq ^ ntm_flip;
            let ntm_feature = opp_king_idx * 768 + ntm_piece_idx * 64 + ntm_sq;

            f(stm_feature, ntm_feature);
        }
    }

    fn shorthand(&self) -> String {
        "HalfKA-hm-Kitty".to_string()
    }

    fn description(&self) -> String {
        "HalfKA-hm (24576 features) with horizontal king folding matching KittyEngine coordinates".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Exact replica of KittyEngine C++ NnueLayout for verification
    fn kitty_encode_cpp(board_fen: &str) -> (Vec<usize>, Vec<usize>) {
        // Parse FEN into White/Black pieces
        // In KittyEngine: A8=0, H8=7, A1=56, H1=63
        // White=0, Black=1
        // Pieces: Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4, King=5
        let parts: Vec<&str> = board_fen.split_whitespace().collect();
        let ranks: Vec<&str> = parts[0].split('/').collect();
        let stm_is_white = parts[1] == "w";

        let mut white_pieces = Vec::new(); // (piece, sq)
        let mut black_pieces = Vec::new();

        for (r, rank_str) in ranks.iter().enumerate() {
            let mut file = 0;
            for ch in rank_str.chars() {
                if let Some(digit) = ch.to_digit(10) {
                    file += digit as usize;
                } else {
                    let sq = r * 8 + file;
                    let (side, piece) = match ch {
                        'P' => (0, 0), 'N' => (0, 1), 'B' => (0, 2), 'R' => (0, 3), 'Q' => (0, 4), 'K' => (0, 5),
                        'p' => (1, 0), 'n' => (1, 1), 'b' => (1, 2), 'r' => (1, 3), 'q' => (1, 4), 'k' => (1, 5),
                        _ => panic!("Unknown char {}", ch),
                    };
                    if side == 0 {
                        white_pieces.push((piece, sq));
                    } else {
                        black_pieces.push((piece, sq));
                    }
                    file += 1;
                }
            }
        }

        let encode_for_side = |perspective: usize| -> Vec<usize> {
            let orient = |persp: usize, sq: usize| -> usize {
                if persp == 0 { sq } else { sq ^ 56 }
            };
            let get_flip = |oriented_ksq: usize| -> usize {
                if (oriented_ksq & 7) > 3 { 7 } else { 0 }
            };
            let king_sq = if perspective == 0 {
                white_pieces.iter().find(|(p, _)| *p == 5).unwrap().1
            } else {
                black_pieces.iter().find(|(p, _)| *p == 5).unwrap().1
            };
            let oriented_king = orient(perspective, king_sq);
            let flip = get_flip(oriented_king);
            let folded_king = oriented_king ^ flip;
            let king_idx = ((folded_king >> 3) * 4) + (folded_king & 7);
            let king_base = king_idx * 768;

            let mut feats = Vec::new();
            for &(piece, sq) in &white_pieces {
                let piece_side = 0;
                let rel_side = piece_side ^ perspective;
                let oriented_sq = orient(perspective, sq);
                let folded_sq = oriented_sq ^ flip;
                let piece_idx = rel_side * 6 + piece;
                feats.push(king_base + piece_idx * 64 + folded_sq);
            }
            for &(piece, sq) in &black_pieces {
                let piece_side = 1;
                let rel_side = piece_side ^ perspective;
                let oriented_sq = orient(perspective, sq);
                let folded_sq = oriented_sq ^ flip;
                let piece_idx = rel_side * 6 + piece;
                feats.push(king_base + piece_idx * 64 + folded_sq);
            }
            feats.sort();
            feats
        };

        let white_feats = encode_for_side(0);
        let black_feats = encode_for_side(1);

        if stm_is_white {
            (white_feats, black_feats) // (stm, ntm)
        } else {
            (black_feats, white_feats) // (stm, ntm)
        }
    }

    #[test]
    fn test_inspect_binpack_samples() {
        use std::path::Path;
        use bulletformat::{DataLoader, ChessBoard};

        let binpack_path = Path::new("data/eval_chunk_001.binpack");
        if !binpack_path.exists() {
            println!("Binpack not found, skipping");
            return;
        }

        let loader = DataLoader::<ChessBoard>::new(binpack_path, 1024).expect("Failed to open binpack");
        let mut count = 0;
        let mut scores = Vec::new();
        let mut results = Vec::new();
        let mut bucket_counts = [0usize; 8];

        loader.map_batches(16384, |batch| {
            if count >= 100_000 {
                return;
            }
            for board in batch {
                count += 1;
                scores.push(board.score);
                results.push(board.result);
                let occ = board.occ().count_ones() as usize;
                let b = ((occ.saturating_sub(2)) / 4).min(7);
                bucket_counts[b] += 1;

                if count <= 10 {
                    println!("Sample {:2}: score = {:5} cp, result = {:3} ({:.1}), occ = {:2}, bucket = {}",
                        count, board.score, board.result, board.result as f32 / 2.0, occ, b);
                }
                if count >= 100_000 {
                    break;
                }
            }
        });

        println!("\n=== 100,000 SAMPLES SUMMARY ===");
        println!("Bucket distribution: {:?}", bucket_counts);
        let win_cnt = results.iter().filter(|&&r| r == 2).count();
        let draw_cnt = results.iter().filter(|&&r| r == 1).count();
        let loss_cnt = results.iter().filter(|&&r| r == 0).count();
        println!("Results: Win={}, Draw={}, Loss={}", win_cnt, draw_cnt, loss_cnt);

        let mut gt_1000 = 0;
        let mut gt_5000 = 0;
        let mut gt_20000 = 0;
        let mut max_score = i16::MIN;
        let mut min_score = i16::MAX;
        for &s in &scores {
            max_score = max_score.max(s);
            min_score = min_score.min(s);
            if s.abs() >= 1000 { gt_1000 += 1; }
            if s.abs() >= 5000 { gt_5000 += 1; }
            if s.abs() >= 20000 { gt_20000 += 1; }
        }
        println!("Scores: min = {}, max = {}, abs >= 1000: {}, abs >= 5000: {}, abs >= 20000: {}",
            min_score, max_score, gt_1000, gt_5000, gt_20000);
    }

    #[test]
    fn test_inspect_bucket_weights() {
        use std::path::Path;
        use crate::eval::NnueWeights;

        let w = NnueWeights::load_from_file(Path::new("checkpoints/champion_weights.bin"))
            .expect("Failed to load champion_weights.bin");

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
    fn test_compare_kitty_cpp_vs_rust_inputs() {
        let test_fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
            "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
            "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4",
            "8/8/8/4k3/8/8/4K3/4R3 w - - 0 1",
            "8/8/8/4k3/8/8/4K3/4R3 b - - 0 1",
        ];

        for fen in test_fens {
            let (cpp_stm, cpp_ntm) = kitty_encode_cpp(fen);
            let board = format!("{fen} | 0 | 0.5").parse::<ChessBoard>().unwrap();
            let mut rust_stm = Vec::new();
            let mut rust_ntm = Vec::new();
            KittyHalfKA.map_features(&board, |stm, ntm| {
                rust_stm.push(stm);
                rust_ntm.push(ntm);
            });
            rust_stm.sort();
            rust_ntm.sort();

            println!("\nTesting FEN: {}", fen);
            assert_eq!(rust_stm.len(), cpp_stm.len(), "STM feature count mismatch on {}", fen);
            assert_eq!(rust_ntm.len(), cpp_ntm.len(), "NTM feature count mismatch on {}", fen);

            if rust_stm != cpp_stm {
                println!("STM MISMATCH on {}", fen);
                println!("CPP:  {:?}", cpp_stm);
                println!("RUST: {:?}", rust_stm);
                panic!("STM mismatch!");
            }
            if rust_ntm != cpp_ntm {
                println!("NTM MISMATCH on {}", fen);
                println!("CPP:  {:?}", cpp_ntm);
                println!("RUST: {:?}", rust_ntm);
                panic!("NTM mismatch!");
            }
            println!("MATCH OK!");
        }
    }

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

        // Start position has 32 pieces, all should be active in HalfKA-hm
        assert_eq!(stm_feats.len(), 32);
        assert_eq!(ntm_feats.len(), 32);

        // Every feature index must be within [0, 24576)
        for &feat in &stm_feats {
            assert!(feat < 24_576, "stm feature {} exceeds 24575", feat);
        }
        for &feat in &ntm_feats {
            assert!(feat < 24_576, "ntm feature {} exceeds 24575", feat);
        }

        // White king on E1 in KittyEngine coords is square 60.
        // File 4 > 3 -> flipped by 7 to D1 (sq 59). Folded king index is 7 * 4 + 3 = 31.
        // Stride is 768. Friendly king piece index is 5. Folded square is 59.
        // Expected white king feature = 31 * 768 + 5 * 64 + 59 = 23,808 + 320 + 59 = 24,187.
        assert!(stm_feats.contains(&24_187), "STM features must contain White king at 24,187");
    }

    #[test]
    fn test_bullet_material_count_buckets() {
        use bullet_lib::game::outputs::MaterialCount;
        use bullet_lib::game::outputs::OutputBuckets;
        let mb = MaterialCount::<8>;

        // Print buckets for boards with 2..=32 pieces
        // In bulletformat:
        // We can create a ChessBoard by parsing FEN
        for p in 2..=32 {
            // rank 8: king on e8 (sq 4)
            // rank 1: king on e1 (sq 4)
            // rest are pawns
            let mut pawns_needed = p - 2;
            let mut ranks = Vec::new();
            for r in 0..8 {
                let mut rank_str = String::new();
                for f in 0..8 {
                    if r == 0 && f == 4 {
                        rank_str.push('k');
                    } else if r == 7 && f == 4 {
                        rank_str.push('K');
                    } else if pawns_needed > 0 && r > 0 && r < 7 {
                        rank_str.push('P');
                        pawns_needed -= 1;
                    } else {
                        rank_str.push('1');
                    }
                }
                // compress consecutive 1s
                let mut compressed = String::new();
                let mut empty_count = 0;
                for ch in rank_str.chars() {
                    if ch == '1' {
                        empty_count += 1;
                    } else {
                        if empty_count > 0 {
                            compressed.push_str(&empty_count.to_string());
                            empty_count = 0;
                        }
                        compressed.push(ch);
                    }
                }
                if empty_count > 0 {
                    compressed.push_str(&empty_count.to_string());
                }
                ranks.push(compressed);
            }
            let full_fen = format!("{} w - - 0 1 | 0 | 0.5", ranks.join("/"));
            let board = full_fen.parse::<ChessBoard>().unwrap();
            let b_bucket = mb.bucket(&board);
            let k_bucket = if p <= 2 { 0 } else { std::cmp::min(7, (p - 2) / 4) };
            assert_eq!(b_bucket, k_bucket, "Bucket mismatch for {} pieces!", p);
        }
    }
}
