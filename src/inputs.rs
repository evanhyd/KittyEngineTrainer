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
    fn test_eval_startpos_moves() {
        use std::path::Path;
        use crate::eval::NnueWeights;

        let w_new = NnueWeights::load_from_file(Path::new("checkpoints/champion_weights.bin"))
            .expect("Failed to load champion_weights.bin");

        let old_bytes = std::fs::read("checkpoints_v5_backup/champion_weights.bin").unwrap();
        let payload_floats: &[f32] = unsafe {
            std::slice::from_raw_parts(
                old_bytes[64..].as_ptr() as *const f32,
                (old_bytes.len() - 64) / 4,
            )
        };
        let mut offset = 0;
        let l0b = &payload_floats[offset..offset + 256]; offset += 256;
        let l0w = &payload_floats[offset..offset + 49_152 * 256]; offset += 49_152 * 256;
        let l1b = &payload_floats[offset..offset + 32]; offset += 32;
        let l1w = &payload_floats[offset..offset + 32 * 512]; offset += 32 * 512;
        let l2b = &payload_floats[offset..offset + 32]; offset += 32;
        let l2w = &payload_floats[offset..offset + 32 * 32]; offset += 32 * 32;
        let l3b = payload_floats[offset]; offset += 1;
        let l3w = &payload_floats[offset..offset + 32];

        let act = |x: f32| -> f32 { let c = x.clamp(0.0, 1.0); c * c };

        let candidate_moves = [
            ("startpos (quiet)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("1. e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
            ("1. d4", "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1"),
            ("1. e3", "rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
            ("1. c4", "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1"),
            ("1. Nf3", "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"),
            ("1. h4 (blunder)", "rnbqkbnr/pppppppp/8/8/7P/8/PPPPPPP1/RNBQKBNR b KQkq h3 0 1"),
            ("1. a4 (blunder)", "rnbqkbnr/pppppppp/8/8/P7/8/1PPPPPPP/RNBQKBNR b KQkq a3 0 1"),
            ("1. g4 (blunder)", "rnbqkbnr/pppppppp/8/8/6P1/8/PPPPPP1P/RNBQKBNR b KQkq g3 0 1"),
        ];

        println!("\n=== STARTPOS CANDIDATE MOVE STATIC EVALUATIONS ===");
        println!("{:<20} | {:<16} | {:<16}", "Move", "Old (NNUESGD5)", "New (NNUESGD6)");
        println!("{:-<20}-+-{:-<16}-+-{:-<16}", "", "", "");

        for (desc, fen) in candidate_moves {
            let board = format!("{fen} | 0 | 0.5").parse::<ChessBoard>().unwrap();
            
            // New Net eval
            let cp_new_raw = w_new.evaluate_cp(&board);
            let cp_new = if fen.contains(" w ") { cp_new_raw } else { -cp_new_raw };

            // Old Net eval
            let our_ksq = (board.our_ksq() ^ 56) as usize;
            let opp_ksq = (board.opp_ksq() ^ 56) as usize;
            let mut stm_acc = l0b.to_vec();
            let mut ntm_acc = l0b.to_vec();
            for (piece, square) in board.into_iter() {
                let piece_type = (piece & 7) as usize;
                let colour = usize::from((piece & 8) != 0);
                let sq = square as usize;
                let stm_piece_idx = colour * 6 + piece_type;
                let stm_sq = sq ^ 56;
                let stm_feat = our_ksq * 768 + stm_piece_idx * 64 + stm_sq;
                let ntm_piece_idx = (1 - colour) * 6 + piece_type;
                let ntm_sq = sq;
                let ntm_feat = opp_ksq * 768 + ntm_piece_idx * 64 + ntm_sq;
                for (a, &w) in stm_acc.iter_mut().zip(&l0w[stm_feat * 256..(stm_feat + 1) * 256]) { *a += w; }
                for (a, &w) in ntm_acc.iter_mut().zip(&l0w[ntm_feat * 256..(ntm_feat + 1) * 256]) { *a += w; }
            }
            let mut combined = [0.0f32; 512];
            for i in 0..256 { combined[i] = act(stm_acc[i]); combined[256 + i] = act(ntm_acc[i]); }
            let mut h1 = [0.0f32; 32];
            for i in 0..32 {
                let mut dot = l1b[i];
                for j in 0..512 { dot += combined[j] * l1w[i * 512 + j]; }
                h1[i] = act(dot);
            }
            let mut h2 = [0.0f32; 32];
            for i in 0..32 {
                let mut dot = l2b[i];
                for j in 0..32 { dot += h1[j] * l2w[i * 32 + j]; }
                h2[i] = act(dot);
            }
            let mut raw_old = l3b;
            for i in 0..32 { raw_old += h2[i] * l3w[i]; }
            let cp_old = if fen.contains(" w ") { raw_old } else { -raw_old };

            println!("{:<20} | {:+8.2} cp       | {:+8.2} cp", desc, cp_old, cp_new);
        }
    }

    #[test]
    fn test_eval_krk_moves() {
        use std::path::Path;
        use crate::eval::NnueWeights;

        let w = NnueWeights::load_from_file(Path::new("checkpoints/champion_weights.bin"))
            .expect("Failed to load champion_weights.bin");
        let w_warm = NnueWeights::load_from_file(Path::new("checkpoints/champion_warmstart_8bucket.bin"))
            .expect("Failed to load champion_warmstart_8bucket.bin");

        // FEN: 8/8/8/4k3/8/8/4K3/4R3 w - - 0 1
        // White King on e2, Rook on e1, Black King on e5
        let candidate_moves = [
            ("e2f3 (King forward)", "8/8/8/4k3/8/5K2/8/4R3 b - - 1 1"),
            ("e2e3 (King forward)", "8/8/8/4k3/8/4K3/8/4R3 b - - 1 1"),
            ("e2d3 (King forward)", "8/8/8/4k3/8/3K4/8/4R3 b - - 1 1"),
            ("e2f2 (King sideways)", "8/8/8/4k3/8/8/5K2/4R3 b - - 1 1"),
            ("e2d2 (King sideways)", "8/8/8/4k3/8/8/3K4/4R3 b - - 1 1"),
            ("e2d1 (King retreat)", "8/8/8/4k3/8/8/8/3K1R2 b - - 1 1"),
            ("e2e1 (King captures own rook?? No, e2f1)", "8/8/8/4k3/8/8/8/4KR2 b - - 1 1"),
            ("e1e3 (Rook check)", "8/8/8/4k3/8/4R3/4K3/8 b - - 1 1"),
            ("e1f1 (Rook move)", "8/8/8/4k3/8/8/4K3/5R2 b - - 1 1"),
            ("e1a1 (Rook to corner)", "8/8/8/4k3/8/8/4K3/R7 b - - 1 1"),
        ];

        println!("\n=== KRK CANDIDATE MOVE STATIC EVALUATIONS ===");
        println!("{:<25} | {:<16} | {:<16}", "Move", "NNUESGD6 (Step 70)", "Warmstart 8-Bucket");
        println!("{:-<25}-+-{:-<16}-+-{:-<16}", "", "", "");
        for (desc, fen) in candidate_moves {
            let board = format!("{fen} | 0 | 0.5").parse::<ChessBoard>().unwrap();
            let cp_w6 = -w.evaluate_cp(&board);
            let cp_warm = -w_warm.evaluate_cp(&board);
            println!("{:<25} | {:+8.2} cp       | {:+8.2} cp", desc, cp_w6, cp_warm);
        }

        // Now evaluate with Old Engine NNUESGD5
        let old_bytes = std::fs::read("checkpoints_v5_backup/champion_weights.bin").unwrap();
        let payload_floats: &[f32] = unsafe {
            std::slice::from_raw_parts(
                old_bytes[64..].as_ptr() as *const f32,
                (old_bytes.len() - 64) / 4,
            )
        };
        let mut offset = 0;
        let l0b = &payload_floats[offset..offset + 256]; offset += 256;
        let l0w = &payload_floats[offset..offset + 49_152 * 256]; offset += 49_152 * 256;
        let l1b = &payload_floats[offset..offset + 32]; offset += 32;
        let l1w = &payload_floats[offset..offset + 32 * 512]; offset += 32 * 512;
        let l2b = &payload_floats[offset..offset + 32]; offset += 32;
        let l2w = &payload_floats[offset..offset + 32 * 32]; offset += 32 * 32;
        let l3b = payload_floats[offset]; offset += 1;
        let l3w = &payload_floats[offset..offset + 32];

        let act = |x: f32| -> f32 { let c = x.clamp(0.0, 1.0); c * c };

        println!("\n=== OLD NNUESGD5 KRK CANDIDATE MOVE EVALUATIONS ===");
        for (desc, fen) in candidate_moves {
            let board = format!("{fen} | 0 | 0.5").parse::<ChessBoard>().unwrap();
            let our_ksq = (board.our_ksq() ^ 56) as usize;
            let opp_ksq = (board.opp_ksq() ^ 56) as usize;

            let mut stm_acc = l0b.to_vec();
            let mut ntm_acc = l0b.to_vec();

            for (piece, square) in board.into_iter() {
                let piece_type = (piece & 7) as usize;
                let colour = usize::from((piece & 8) != 0);
                let sq = square as usize;

                let stm_piece_idx = colour * 6 + piece_type;
                let stm_sq = sq ^ 56;
                let stm_feat = our_ksq * 768 + stm_piece_idx * 64 + stm_sq;

                let ntm_piece_idx = (1 - colour) * 6 + piece_type;
                let ntm_sq = sq;
                let ntm_feat = opp_ksq * 768 + ntm_piece_idx * 64 + ntm_sq;

                let r_stm = &l0w[stm_feat * 256..(stm_feat + 1) * 256];
                for (a, &w) in stm_acc.iter_mut().zip(r_stm) { *a += w; }
                let r_ntm = &l0w[ntm_feat * 256..(ntm_feat + 1) * 256];
                for (a, &w) in ntm_acc.iter_mut().zip(r_ntm) { *a += w; }
            }

            let mut combined = [0.0f32; 512];
            for i in 0..256 {
                combined[i] = act(stm_acc[i]);
                combined[256 + i] = act(ntm_acc[i]);
            }
            let mut h1 = [0.0f32; 32];
            for i in 0..32 {
                let mut dot = l1b[i];
                for j in 0..512 { dot += combined[j] * l1w[i * 512 + j]; }
                h1[i] = act(dot);
            }
            let mut h2 = [0.0f32; 32];
            for i in 0..32 {
                let mut dot = l2b[i];
                for j in 0..32 { dot += h1[j] * l2w[i * 32 + j]; }
                h2[i] = act(dot);
            }
            let mut raw = l3b;
            for i in 0..32 { raw += h2[i] * l3w[i]; }

            let cp_for_white = -raw;
            println!("{:<25} | Old White score: {:+8.2} cp", desc, cp_for_white);
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
            let mut fen = String::new();
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

    #[test]
    fn test_fold_nnuesgd5() {
        let old_bytes = std::fs::read("checkpoints_v5_backup/champion_weights.bin").unwrap();
        let payload_floats: &[f32] = unsafe {
            std::slice::from_raw_parts(
                old_bytes[64..].as_ptr() as *const f32,
                (old_bytes.len() - 64) / 4,
            )
        };
        let mut offset = 0;
        let l0b = &payload_floats[offset..offset + 256]; offset += 256;
        let l0w = &payload_floats[offset..offset + 49_152 * 256]; offset += 49_152 * 256;
        let l1b = &payload_floats[offset..offset + 32]; offset += 32;
        let l1w = &payload_floats[offset..offset + 32 * 512]; offset += 32 * 512;
        let l2b = &payload_floats[offset..offset + 32]; offset += 32;
        let l2w = &payload_floats[offset..offset + 32 * 32]; offset += 32 * 32;
        let l3b = payload_floats[offset]; offset += 1;
        let l3w = &payload_floats[offset..offset + 32];

        // Fold accumulator weights from 49152 to 24576
        let mut folded_l0w = vec![0.0f32; 24_576 * 256];
        for k in 0..64 {
            let file = k & 7;
            if file <= 3 {
                let rank = k >> 3;
                let folded_k = rank * 4 + file;
                let mirror_k = k ^ 7;

                for p in 0..12 {
                    for s in 0..64 {
                        let mirror_s = s ^ 7;
                        let feat_left = k * 768 + p * 64 + s;
                        let feat_right = mirror_k * 768 + p * 64 + mirror_s;
                        let folded_feat = folded_k * 768 + p * 64 + s;

                        let left_row = &l0w[feat_left * 256..(feat_left + 1) * 256];
                        let right_row = &l0w[feat_right * 256..(feat_right + 1) * 256];
                        let target_row = &mut folded_l0w[folded_feat * 256..(folded_feat + 1) * 256];

                        for j in 0..256 {
                            target_row[j] = (left_row[j] + right_row[j]) * 0.5;
                        }
                    }
                }
            }
        }

        let act = |x: f32| -> f32 { let c = x.clamp(0.0, 1.0); c * c };

        // Test function using folded weights with KittyHalfKA feature mapping
        let eval_folded = |fen: &str| -> f32 {
            let board = format!("{fen} | 0 | 0.5").parse::<ChessBoard>().unwrap();
            let mut stm_acc = l0b.to_vec();
            let mut ntm_acc = l0b.to_vec();

            KittyHalfKA.map_features(&board, |stm, ntm| {
                for (a, &w) in stm_acc.iter_mut().zip(&folded_l0w[stm * 256..(stm + 1) * 256]) { *a += w; }
                for (a, &w) in ntm_acc.iter_mut().zip(&folded_l0w[ntm * 256..(ntm + 1) * 256]) { *a += w; }
            });

            let mut combined = [0.0f32; 512];
            for i in 0..256 {
                combined[i] = act(stm_acc[i]);
                combined[256 + i] = act(ntm_acc[i]);
            }
            let mut h1 = [0.0f32; 32];
            for i in 0..32 {
                let mut dot = l1b[i];
                for j in 0..512 { dot += combined[j] * l1w[i * 512 + j]; }
                h1[i] = act(dot);
            }
            let mut h2 = [0.0f32; 32];
            for i in 0..32 {
                let mut dot = l2b[i];
                for j in 0..32 { dot += h1[j] * l2w[i * 32 + j]; }
                h2[i] = act(dot);
            }
            let mut raw = l3b;
            for i in 0..32 { raw += h2[i] * l3w[i]; }

            if fen.contains(" w ") { raw } else { -raw }
        };

        // Test function using original unfolded weights with unfolded feature mapping
        let eval_unfolded = |fen: &str| -> f32 {
            let board = format!("{fen} | 0 | 0.5").parse::<ChessBoard>().unwrap();
            let our_ksq = (board.our_ksq() ^ 56) as usize;
            let opp_ksq = (board.opp_ksq() ^ 56) as usize;
            let mut stm_acc = l0b.to_vec();
            let mut ntm_acc = l0b.to_vec();

            for (piece, square) in board.into_iter() {
                let piece_type = (piece & 7) as usize;
                let colour = usize::from((piece & 8) != 0);
                let sq = square as usize;
                let stm_piece_idx = colour * 6 + piece_type;
                let stm_sq = sq ^ 56;
                let stm_feat = our_ksq * 768 + stm_piece_idx * 64 + stm_sq;
                let ntm_piece_idx = (1 - colour) * 6 + piece_type;
                let ntm_sq = sq;
                let ntm_feat = opp_ksq * 768 + ntm_piece_idx * 64 + ntm_sq;
                for (a, &w) in stm_acc.iter_mut().zip(&l0w[stm_feat * 256..(stm_feat + 1) * 256]) { *a += w; }
                for (a, &w) in ntm_acc.iter_mut().zip(&l0w[ntm_feat * 256..(ntm_feat + 1) * 256]) { *a += w; }
            }

            let mut combined = [0.0f32; 512];
            for i in 0..256 { combined[i] = act(stm_acc[i]); combined[256 + i] = act(ntm_acc[i]); }
            let mut h1 = [0.0f32; 32];
            for i in 0..32 {
                let mut dot = l1b[i];
                for j in 0..512 { dot += combined[j] * l1w[i * 512 + j]; }
                h1[i] = act(dot);
            }
            let mut h2 = [0.0f32; 32];
            for i in 0..32 {
                let mut dot = l2b[i];
                for j in 0..32 { dot += h1[j] * l2w[i * 32 + j]; }
                h2[i] = act(dot);
            }
            let mut raw = l3b;
            for i in 0..32 { raw += h2[i] * l3w[i]; }

            if fen.contains(" w ") { raw } else { -raw }
        };

        println!("\n=== FOLDED (24,576) VS UNFOLDED (49,152) COMPARISON ===");
        let test_positions = [
            ("Startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("1. e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
            ("1. d4", "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1"),
            ("1. e3", "rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
            ("1. a4 (blunder)", "rnbqkbnr/pppppppp/8/8/P7/8/1PPPPPPP/RNBQKBNR b KQkq a3 0 1"),
            ("KRK e2f3 (King advance)", "8/8/8/4k3/8/5K2/8/4R3 b - - 1 1"),
            ("KRK e2d1 (King retreat)", "8/8/8/4k3/8/8/8/3K1R2 b - - 1 1"),
            ("Kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        ];

        println!("{:<25} | {:>15} | {:>15} | {:>10}", "Position", "Unfolded (49k)", "Folded (24.5k)", "Diff");
        println!("{:-<25}-+-{:-<15}-+-{:-<15}-+-{:-<10}", "", "", "", "");
        for (desc, fen) in test_positions {
            let u_val = eval_unfolded(fen);
            let f_val = eval_folded(fen);
            let diff = (u_val - f_val).abs();
            println!("{:<25} | {:>12.2} cp | {:>12.2} cp | {:>8.2} cp", desc, u_val, f_val, diff);
        }
    }
}
