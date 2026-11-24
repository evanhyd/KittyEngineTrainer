use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::sync::{Arc, Mutex, mpsc};
use bullet_trainer::reader::DataReader;
use bulletformat::ChessBoard;
use sfbinpack::{
    TrainingDataEntry,
    chess::{color::Color, piecetype::PieceType},
};

/// Detects whether a file begins with Stockfish binpack magic header (`BINP`).
pub fn is_sfbinpack_file(path: &Path) -> bool {
    if let Ok(mut f) = File::open(path) {
        let mut magic = [0u8; 4];
        if f.read_exact(&mut magic).is_ok() {
            return &magic == b"BINP";
        }
    }
    false
}

/// Converts a single Stockfish `TrainingDataEntry` into Bullet's `ChessBoard` struct.
pub fn convert_sfentry_to_chessboard(entry: &TrainingDataEntry) -> Option<ChessBoard> {
    if entry.score.abs() > 10000 {
        return None;
    }

    let mut bbs = [0u64; 8];

    let stm = usize::from(entry.pos.side_to_move().ordinal());
    let pc_bb = |pt| {
        entry.pos.pieces_bb_color(Color::Black, pt).bits()
            | entry.pos.pieces_bb_color(Color::White, pt).bits()
    };

    bbs[0] = entry.pos.pieces_bb(Color::White).bits();
    bbs[1] = entry.pos.pieces_bb(Color::Black).bits();
    bbs[2] = pc_bb(PieceType::Pawn);
    bbs[3] = pc_bb(PieceType::Knight);
    bbs[4] = pc_bb(PieceType::Bishop);
    bbs[5] = pc_bb(PieceType::Rook);
    bbs[6] = pc_bb(PieceType::Queen);
    bbs[7] = pc_bb(PieceType::King);

    let mut score = entry.score;
    let mut result = f32::from(1 + entry.result) / 2.0;

    if stm > 0 {
        score = -score;
        result = 1.0 - result;
    }

    ChessBoard::from_raw(bbs, stm, score, result).ok()
}

/// Iterates over batches of `ChessBoard` positions from either compressed Stockfish binpacks (`BINP`)
/// or standard uncompressed bulletformat binpacks.
pub fn for_each_board_in_pack<F: FnMut(&[ChessBoard]) -> bool>(
    pack_path: &Path,
    batch_size: usize,
    max_positions: usize,
    mut f: F,
) -> io::Result<usize> {
    if is_sfbinpack_file(pack_path) {
        let file = File::open(pack_path)?;
        let mut reader = match sfbinpack::CompressedTrainingDataEntryReader::new(file) {
            Ok(r) => r,
            Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{:?}", e))),
        };

        let mut buffer = Vec::with_capacity(batch_size);
        let mut total_read = 0;

        while reader.has_next() {
            if max_positions > 0 && total_read >= max_positions {
                break;
            }

            let entry = reader.next();
            if entry.pos.is_checked(entry.pos.side_to_move()) || entry.score.abs() > 10000 {
                continue;
            }

            if let Some(board) = convert_sfentry_to_chessboard(&entry) {
                buffer.push(board);
                total_read += 1;

                if buffer.len() >= batch_size {
                    if !f(&buffer) {
                        return Ok(total_read);
                    }
                    buffer.clear();
                }
            }
        }

        if !buffer.is_empty() {
            f(&buffer);
        }

        Ok(total_read)
    } else {
        let loader = bulletformat::DataLoader::<ChessBoard>::new(pack_path, 64)?;
        let mut total_read = 0;

        loader.map_batches(batch_size, |batch| {
            if max_positions > 0 && total_read >= max_positions {
                return;
            }

            let slice = if max_positions > 0 && total_read + batch.len() > max_positions {
                &batch[..max_positions - total_read]
            } else {
                batch
            };

            total_read += slice.len();
            f(slice);
        });

        Ok(total_read)
    }
}

/// Wraps any `DataReader<T>` (such as `SfBinpackLoader`, which ignores `skip_count` and
/// reopens from byte 0 on every `read_chunks` call) in a single persistent background stream.
/// Each superbatch resumes at the exact next chunk across all binpack files instead of
/// restarting at byte 0 of `pack1.binpack`.
pub struct PersistentDataReader<T: Send + 'static> {
    rx: Arc<Mutex<mpsc::Receiver<Vec<T>>>>,
}

impl<T: Send + 'static> Clone for PersistentDataReader<T> {
    fn clone(&self) -> Self {
        Self { rx: Arc::clone(&self.rx) }
    }
}

impl<T: Clone + Send + 'static> PersistentDataReader<T> {
    pub fn new<R: DataReader<T>>(inner: R) -> Self {
        let (tx, rx) = mpsc::sync_channel::<Vec<T>>(1);
        std::thread::spawn(move || {
            inner.read_chunks(0, |chunk| tx.send(chunk.to_vec()).is_err());
        });
        Self { rx: Arc::new(Mutex::new(rx)) }
    }
}

impl<T: Clone + Send + Sync + 'static> DataReader<T> for PersistentDataReader<T> {
    fn read_chunks<F: FnMut(&[T]) -> bool>(&self, _skip_count: usize, mut f: F) {
        let rx = self.rx.lock().expect("PersistentDataReader mutex poisoned");
        while let Ok(chunk) = rx.recv() {
            if f(&chunk) {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct SeqReader(usize);

    impl DataReader<u32> for SeqReader {
        fn read_chunks<F: FnMut(&[u32]) -> bool>(&self, _skip: usize, mut f: F) {
            for i in 0..self.0 as u32 {
                if f(&[i]) {
                    break;
                }
            }
        }
    }

    #[test]
    fn test_persistent_data_reader_advances_across_superbatches() {
        let reader = PersistentDataReader::new(SeqReader(10));
        let mut seen = Vec::new();

        for _ in 0..3 {
            let mut count = 0;
            reader.read_chunks(0, |chunk| {
                seen.extend_from_slice(chunk);
                count += 1;
                count >= 2
            });
        }

        assert_eq!(seen, vec![0, 1, 2, 3, 4, 5]);
    }
}
