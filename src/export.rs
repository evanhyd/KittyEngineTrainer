use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use byteorder::{LittleEndian, WriteBytesExt};
use crate::config::{INPUT_SIZE, ACCUM_SIZE, PAIRWISE_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, NUM_OUTPUT_BUCKETS};

const MAGIC_V8: &[u8; 8] = b"NNUESGD8";
const DIMENSIONS: [u64; 5] = [
    INPUT_SIZE as u64,
    ACCUM_SIZE as u64,
    HIDDEN1_SIZE as u64,
    HIDDEN2_SIZE as u64,
    NUM_OUTPUT_BUCKETS as u64,
];
pub const EXACT_FILE_SIZE: u64 = crate::config::CHECKPOINT_SIZE;

#[derive(Debug, Clone, Default)]
pub struct QuantizationReport {
    pub l0_clipped: usize,
    pub l1_clipped: usize,
    pub l2_clipped: usize,
    pub l3_clipped: usize,
}

impl QuantizationReport {
    pub fn total_clipped(&self) -> usize {
        self.l0_clipped + self.l1_clipped + self.l2_clipped + self.l3_clipped
    }
}

fn clamp_slice(slice: &mut [f32], limit: f32) -> usize {
    let mut clipped = 0;
    for v in slice.iter_mut() {
        if *v > limit {
            *v = limit;
            clipped += 1;
        } else if *v < -limit {
            *v = -limit;
            clipped += 1;
        }
    }
    clipped
}

#[allow(clippy::too_many_arguments)]
pub fn guard_quantization_ranges(
    l0w: &mut [f32],
    l0b: &mut [f32],
    l1w: &mut [f32],
    l1b: &mut [f32],
    l2w: &mut [f32],
    l2b: &mut [f32],
    l3w: &mut [f32],
    l3b: &mut [f32],
) -> QuantizationReport {
    // L0: QA=255 -> int16_t [-128.0, +128.0]
    // L1: QB=64 -> int8_t [-1.984, +1.984]
    // L2/L3: QB=64 -> int16_t [-511.98, +511.98]
    // Biases: scale 8128 -> int32_t [-260000, +260000]
    QuantizationReport {
        l0_clipped: clamp_slice(l0w, 128.0) + clamp_slice(l0b, 128.0),
        l1_clipped: clamp_slice(l1w, 1.984) + clamp_slice(l1b, 260_000.0),
        l2_clipped: clamp_slice(l2w, 511.98) + clamp_slice(l2b, 260_000.0),
        l3_clipped: clamp_slice(l3w, 511.98) + clamp_slice(l3b, 260_000.0),
    }
}

/// Converts Bullet's `quantised.bin` (which applies `SavedFormat::transform` to fold `l0f` into `l0w`,
/// padded to a 64-byte boundary by Bullet) into KittyEngine's exact NNUESGD8 weight file (38,164,072 bytes).
pub fn export_from_raw_checkpoint(raw_bin_path: &Path, output_path: &Path) -> io::Result<()> {
    let raw_bytes = std::fs::read(raw_bin_path)?;
    if raw_bytes.len() % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checkpoint byte length is not a multiple of 4",
        ));
    }

    let raw_floats: &[f32] = unsafe {
        std::slice::from_raw_parts(raw_bytes.as_ptr() as *const f32, raw_bytes.len() / 4)
    };

    let l0w_len = INPUT_SIZE * ACCUM_SIZE;
    let l0b_len = ACCUM_SIZE;
    let l1w_len = NUM_OUTPUT_BUCKETS * HIDDEN1_SIZE * PAIRWISE_SIZE;
    let l1b_len = NUM_OUTPUT_BUCKETS * HIDDEN1_SIZE;
    let l2w_len = NUM_OUTPUT_BUCKETS * HIDDEN2_SIZE * HIDDEN1_SIZE;
    let l2b_len = NUM_OUTPUT_BUCKETS * HIDDEN2_SIZE;
    let l3w_len = NUM_OUTPUT_BUCKETS * HIDDEN2_SIZE;
    let l3b_len = NUM_OUTPUT_BUCKETS;

    let expected_floats = l0w_len + l0b_len + l1w_len + l1b_len + l2w_len + l2b_len + l3w_len + l3b_len;
    // Bullet pads `quantised.bin` to a 64-byte boundary (up to 15 trailing zero f32s).
    if raw_floats.len() < expected_floats || raw_floats.len() > expected_floats + 15 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("checkpoint contains {} floats, expected {}..={}", raw_floats.len(), expected_floats, expected_floats + 15),
        ));
    }

    let mut offset = 0;
    let mut kitty_l0w = raw_floats[offset..offset + l0w_len].to_vec();
    offset += l0w_len;
    let mut kitty_l0b = raw_floats[offset..offset + l0b_len].to_vec();
    offset += l0b_len;

    let l1w = &raw_floats[offset..offset + l1w_len];
    offset += l1w_len;
    let mut kitty_l1b = raw_floats[offset..offset + l1b_len].to_vec();
    offset += l1b_len;

    let l2w = &raw_floats[offset..offset + l2w_len];
    offset += l2w_len;
    let mut kitty_l2b = raw_floats[offset..offset + l2b_len].to_vec();
    offset += l2b_len;

    let l3w = &raw_floats[offset..offset + l3w_len];
    offset += l3w_len;
    let mut kitty_l3b = raw_floats[offset..offset + l3b_len].to_vec();

    // Transpose Bullet col-major [col * (B * H) + b * H + r] -> KittyEngine row-major [b][r][col]
    let mut kitty_l1w = vec![0.0f32; l1w_len];
    let l1_rows = NUM_OUTPUT_BUCKETS * HIDDEN1_SIZE;
    for b in 0..NUM_OUTPUT_BUCKETS {
        for r in 0..HIDDEN1_SIZE {
            let dst_base = (b * HIDDEN1_SIZE + r) * PAIRWISE_SIZE;
            let src_row = b * HIDDEN1_SIZE + r;
            for c in 0..PAIRWISE_SIZE {
                kitty_l1w[dst_base + c] = l1w[c * l1_rows + src_row];
            }
        }
    }

    let mut kitty_l2w = vec![0.0f32; l2w_len];
    let l2_rows = NUM_OUTPUT_BUCKETS * HIDDEN2_SIZE;
    for b in 0..NUM_OUTPUT_BUCKETS {
        for r in 0..HIDDEN2_SIZE {
            let dst_base = (b * HIDDEN2_SIZE + r) * HIDDEN1_SIZE;
            let src_row = b * HIDDEN2_SIZE + r;
            for c in 0..HIDDEN1_SIZE {
                kitty_l2w[dst_base + c] = l2w[c * l2_rows + src_row];
            }
        }
    }

    let mut kitty_l3w = vec![0.0f32; l3w_len];
    for b in 0..NUM_OUTPUT_BUCKETS {
        for c in 0..HIDDEN2_SIZE {
            kitty_l3w[b * HIDDEN2_SIZE + c] = l3w[c * NUM_OUTPUT_BUCKETS + b];
        }
    }

    let report = guard_quantization_ranges(
        &mut kitty_l0w, &mut kitty_l0b,
        &mut kitty_l1w, &mut kitty_l1b,
        &mut kitty_l2w, &mut kitty_l2b,
        &mut kitty_l3w, &mut kitty_l3b,
    );
    if report.total_clipped() > 0 {
        eprintln!(
            "⚠️ [quantization-guard] Clipped {} weights (L0:{}, L1:{}, L2:{}, L3:{})",
            report.total_clipped(), report.l0_clipped, report.l1_clipped, report.l2_clipped, report.l3_clipped
        );
    }

    let tmp_path = output_path.with_extension("tmp");
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    {
        let out_file = File::create(&tmp_path)?;
        let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, out_file);

        writer.write_all(MAGIC_V8)?;
        for &dim in &DIMENSIONS {
            writer.write_u64::<LittleEndian>(dim)?;
        }
        for _ in 0..3 {
            writer.write_u64::<LittleEndian>(0)?;
        }

        for slice in [
            kitty_l0b.as_slice(), kitty_l0w.as_slice(),
            kitty_l1b.as_slice(), kitty_l1w.as_slice(),
            kitty_l2b.as_slice(), kitty_l2w.as_slice(),
            kitty_l3b.as_slice(), kitty_l3w.as_slice(),
        ] {
            let bytes = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4) };
            writer.write_all(bytes)?;
        }

        writer.flush()?;
        writer.into_inner()?.sync_all()?;
    }

    let final_size = std::fs::metadata(&tmp_path)?.len();
    assert_eq!(final_size, EXACT_FILE_SIZE, "Exported checkpoint must match exact size ({} bytes)", EXACT_FILE_SIZE);

    if output_path.exists() {
        let _ = std::fs::remove_file(output_path);
    }
    std::fs::rename(&tmp_path, output_path)?;

    Ok(())
}

fn write_bullet_tensor(writer: &mut impl Write, id: &str, data: &[f32]) -> io::Result<()> {
    writer.write_all(format!("{id}\n").as_bytes())?;
    writer.write_all(&data.len().to_le_bytes())?;
    let slice_bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    writer.write_all(slice_bytes)?;
    Ok(())
}

/// Imports KittyEngine's NNUESGD8 weights into Bullet's native tagged weights format (with zeroed l0f).
pub fn import_from_kitty_weights(kitty_path: &Path, bullet_path: &Path) -> io::Result<()> {
    let kitty_bytes = std::fs::read(kitty_path)?;
    if kitty_bytes.len() != EXACT_FILE_SIZE as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Expected KittyEngine weights file of {} bytes, got {}", EXACT_FILE_SIZE, kitty_bytes.len()),
        ));
    }

    if &kitty_bytes[0..8] != MAGIC_V8 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid KittyEngine signature (expected NNUESGD8)"));
    }

    let payload_floats: &[f32] = unsafe {
        std::slice::from_raw_parts(kitty_bytes[72..].as_ptr() as *const f32, (kitty_bytes.len() - 72) / 4)
    };

    let l0w_len = INPUT_SIZE * ACCUM_SIZE;
    let l1w_len = NUM_OUTPUT_BUCKETS * HIDDEN1_SIZE * PAIRWISE_SIZE;
    let l1b_len = NUM_OUTPUT_BUCKETS * HIDDEN1_SIZE;
    let l2w_len = NUM_OUTPUT_BUCKETS * HIDDEN2_SIZE * HIDDEN1_SIZE;
    let l2b_len = NUM_OUTPUT_BUCKETS * HIDDEN2_SIZE;
    let l3w_len = NUM_OUTPUT_BUCKETS * HIDDEN2_SIZE;

    let mut offset = 0;
    let l0b = &payload_floats[offset..offset + ACCUM_SIZE];
    offset += ACCUM_SIZE;
    let l0w = &payload_floats[offset..offset + l0w_len];
    offset += l0w_len;
    let l1b = &payload_floats[offset..offset + l1b_len];
    offset += l1b_len;
    let l1w = &payload_floats[offset..offset + l1w_len];
    offset += l1w_len;
    let l2b = &payload_floats[offset..offset + l2b_len];
    offset += l2b_len;
    let l2w = &payload_floats[offset..offset + l2w_len];
    offset += l2w_len;
    let l3b = &payload_floats[offset..offset + NUM_OUTPUT_BUCKETS];
    offset += NUM_OUTPUT_BUCKETS;
    let l3w = &payload_floats[offset..offset + l3w_len];

    if let Some(parent) = bullet_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let out_file = File::create(bullet_path)?;
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, out_file);

    write_bullet_tensor(&mut writer, "l0w", l0w)?;
    write_bullet_tensor(&mut writer, "l0b", l0b)?;

    let mut bullet_l1w = vec![0.0f32; l1w_len];
    let l1_rows = NUM_OUTPUT_BUCKETS * HIDDEN1_SIZE;
    for b in 0..NUM_OUTPUT_BUCKETS {
        for r in 0..HIDDEN1_SIZE {
            let src_base = (b * HIDDEN1_SIZE + r) * PAIRWISE_SIZE;
            let dst_row = b * HIDDEN1_SIZE + r;
            for c in 0..PAIRWISE_SIZE {
                bullet_l1w[c * l1_rows + dst_row] = l1w[src_base + c];
            }
        }
    }
    write_bullet_tensor(&mut writer, "l1w", &bullet_l1w)?;
    write_bullet_tensor(&mut writer, "l1b", l1b)?;

    let mut bullet_l2w = vec![0.0f32; l2w_len];
    let l2_rows = NUM_OUTPUT_BUCKETS * HIDDEN2_SIZE;
    for b in 0..NUM_OUTPUT_BUCKETS {
        for r in 0..HIDDEN2_SIZE {
            let src_base = (b * HIDDEN2_SIZE + r) * HIDDEN1_SIZE;
            let dst_row = b * HIDDEN2_SIZE + r;
            for c in 0..HIDDEN1_SIZE {
                bullet_l2w[c * l2_rows + dst_row] = l2w[src_base + c];
            }
        }
    }
    write_bullet_tensor(&mut writer, "l2w", &bullet_l2w)?;
    write_bullet_tensor(&mut writer, "l2b", l2b)?;

    let mut bullet_l3w = vec![0.0f32; l3w_len];
    for b in 0..NUM_OUTPUT_BUCKETS {
        for c in 0..HIDDEN2_SIZE {
            bullet_l3w[c * NUM_OUTPUT_BUCKETS + b] = l3w[b * HIDDEN2_SIZE + c];
        }
    }
    write_bullet_tensor(&mut writer, "l3w", &bullet_l3w)?;
    write_bullet_tensor(&mut writer, "l3b", l3b)?;

    writer.flush()?;
    println!("Successfully converted KittyEngine weights -> Bullet format at: {}", bullet_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::NnueWeights;

    #[test]
    fn test_export_and_import_roundtrip() {
        let temp_dir = std::env::temp_dir();
        let raw_path = temp_dir.join("test_raw_v8.bin");
        let kitty_path = temp_dir.join("test_kitty_v8.bin");
        let bullet_path = temp_dir.join("test_bullet_v8.bin");

        let total_floats = INPUT_SIZE * ACCUM_SIZE
            + ACCUM_SIZE
            + NUM_OUTPUT_BUCKETS * HIDDEN1_SIZE * PAIRWISE_SIZE
            + NUM_OUTPUT_BUCKETS * HIDDEN1_SIZE
            + NUM_OUTPUT_BUCKETS * HIDDEN2_SIZE * HIDDEN1_SIZE
            + NUM_OUTPUT_BUCKETS * HIDDEN2_SIZE
            + NUM_OUTPUT_BUCKETS * HIDDEN2_SIZE
            + NUM_OUTPUT_BUCKETS;
        let raw = vec![0.01f32; total_floats];
        let raw_bytes = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const u8, raw.len() * 4) };
        std::fs::write(&raw_path, raw_bytes).unwrap();

        export_from_raw_checkpoint(&raw_path, &kitty_path).expect("export failed");
        assert_eq!(std::fs::metadata(&kitty_path).unwrap().len(), EXACT_FILE_SIZE);

        let w = NnueWeights::load_from_file(&kitty_path).expect("load failed");
        assert_eq!(w.accum_size, ACCUM_SIZE);
        assert_eq!(w.hidden1_size, HIDDEN1_SIZE);
        assert_eq!(w.hidden2_size, HIDDEN2_SIZE);
        assert_eq!(w.num_buckets, NUM_OUTPUT_BUCKETS);

        import_from_kitty_weights(&kitty_path, &bullet_path).expect("import failed");
        assert!(bullet_path.exists());

        let _ = std::fs::remove_file(raw_path);
        let _ = std::fs::remove_file(kitty_path);
        let _ = std::fs::remove_file(bullet_path);
    }

    #[test]
    fn test_quantization_guard_clamping() {
        let mut l0w = vec![200.0, -200.0, 1.0];
        let mut l0b = vec![0.0];
        let mut l1w = vec![2.5, -3.0, 0.5];
        let mut l1b = vec![0.0];
        let mut l2w = vec![600.0, -600.0];
        let mut l2b = vec![0.0];
        let mut l3w = vec![600.0, -600.0];
        let mut l3b = vec![0.0];

        let report = guard_quantization_ranges(
            &mut l0w, &mut l0b, &mut l1w, &mut l1b,
            &mut l2w, &mut l2b, &mut l3w, &mut l3b,
        );
        assert_eq!(report.l0_clipped, 2);
        assert_eq!(report.l1_clipped, 2);
        assert_eq!(report.l2_clipped, 2);
        assert_eq!(report.l3_clipped, 2);
        assert!((l0w[0] - 128.0).abs() < 1e-4);
        assert!((l1w[0] - 1.984).abs() < 1e-4);
        assert!((l2w[0] - 511.98).abs() < 1e-4);
        assert!((l3w[0] - 511.98).abs() < 1e-4);
    }
}
