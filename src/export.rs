use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use byteorder::{LittleEndian, WriteBytesExt};

const MAGIC_V6: &[u8; 8] = b"NNUESGD6";
const DIMENSIONS: [u64; 5] = [24_576, 256, 32, 32, 8];

/// Converts Bullet's `raw.bin` into KittyEngine's exact weight file (25,237,864 bytes).
pub fn export_from_raw_checkpoint(raw_bin_path: &Path, output_path: &Path) -> io::Result<()> {
    // Read the entire file into memory instantly (25 MB)
    let raw_bytes = std::fs::read(raw_bin_path)?;
    if raw_bytes.len() % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "raw.bin byte length is not a multiple of 4",
        ));
    }

    let raw_floats: &[f32] = unsafe {
        std::slice::from_raw_parts(
            raw_bytes.as_ptr() as *const f32,
            raw_bytes.len() / 4,
        )
    };

    let expected_floats = 24_576 * 256 + 256 + 512 * 32 + 32 + 32 * 32 + 32 + 32 * 8 + 8;
    if raw_floats.len() != expected_floats {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "raw.bin contains {} floats, expected {}",
                raw_floats.len(),
                expected_floats
            ),
        ));
    }

    let mut offset = 0;

    // Bullet stores:
    // 1. l0w: (256, 24576)
    let l0w_slice = &raw_floats[offset..offset + 256 * 24_576];
    offset += 256 * 24_576;

    // 2. l0b: (256, 1)
    let l0b = &raw_floats[offset..offset + 256];
    offset += 256;

    // 3. l1w: (32, 512)
    let l1w = &raw_floats[offset..offset + 32 * 512];
    offset += 32 * 512;

    // 4. l1b: (32, 1)
    let l1b = &raw_floats[offset..offset + 32];
    offset += 32;

    // 5. l2w: (32, 32)
    let l2w = &raw_floats[offset..offset + 32 * 32];
    offset += 32 * 32;

    // 6. l2b: (32, 1)
    let l2b = &raw_floats[offset..offset + 32];
    offset += 32;

    // 7. l3w: (8, 32)
    let l3w = &raw_floats[offset..offset + 8 * 32];
    offset += 8 * 32;

    // 8. l3b: (8, 1)
    let l3b = &raw_floats[offset..offset + 8];

    let out_file = File::create(output_path)?;
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, out_file);

    // Header (72 bytes)
    writer.write_all(MAGIC_V6)?;
    for &dim in &DIMENSIONS {
        writer.write_u64::<LittleEndian>(dim)?;
    }
    // Progress: [records=0, examples=0, batches=0]
    for _ in 0..3 {
        writer.write_u64::<LittleEndian>(0)?;
    }

    // Accumulator biases (256 floats)
    let bytes_l0b = unsafe { std::slice::from_raw_parts(l0b.as_ptr() as *const u8, l0b.len() * 4) };
    writer.write_all(bytes_l0b)?;

    // Accumulator weights (24576 * 256 floats) - Direct copy: both Bullet and KittyEngine use feature-major
    let bytes_accum = unsafe {
        std::slice::from_raw_parts(
            l0w_slice.as_ptr() as *const u8,
            l0w_slice.len() * 4,
        )
    };
    writer.write_all(bytes_accum)?;

    // Hidden layer 1 biases & weights (transpose from Bullet col-major 512x32 to KittyEngine row-major 32x512)
    let bytes_l1b = unsafe { std::slice::from_raw_parts(l1b.as_ptr() as *const u8, l1b.len() * 4) };
    writer.write_all(bytes_l1b)?;
    let mut kitty_l1w = vec![0.0f32; 32 * 512];
    for row in 0..32 {
        for col in 0..512 {
            kitty_l1w[row * 512 + col] = l1w[col * 32 + row];
        }
    }
    let bytes_l1w = unsafe { std::slice::from_raw_parts(kitty_l1w.as_ptr() as *const u8, kitty_l1w.len() * 4) };
    writer.write_all(bytes_l1w)?;

    // Hidden layer 2 biases & weights (transpose from Bullet col-major 32x32 to KittyEngine row-major 32x32)
    let bytes_l2b = unsafe { std::slice::from_raw_parts(l2b.as_ptr() as *const u8, l2b.len() * 4) };
    writer.write_all(bytes_l2b)?;
    let mut kitty_l2w = vec![0.0f32; 32 * 32];
    for row in 0..32 {
        for col in 0..32 {
            kitty_l2w[row * 32 + col] = l2w[col * 32 + row];
        }
    }
    let bytes_l2w = unsafe { std::slice::from_raw_parts(kitty_l2w.as_ptr() as *const u8, kitty_l2w.len() * 4) };
    writer.write_all(bytes_l2w)?;

    // Output biases & weights (scale by 400.0 from logit space to centipawns for KittyEngine)
    let scale = 400.0f32;
    for &b in l3b {
        writer.write_f32::<LittleEndian>(b * scale)?;
    }

    // Transpose output weights from Bullet col-major 32x8 to KittyEngine row-major 8x32
    let mut kitty_l3w = vec![0.0f32; 8 * 32];
    for bucket in 0..8 {
        for col in 0..32 {
            kitty_l3w[bucket * 32 + col] = l3w[col * 8 + bucket] * scale;
        }
    }
    let bytes_l3w = unsafe { std::slice::from_raw_parts(kitty_l3w.as_ptr() as *const u8, kitty_l3w.len() * 4) };
    writer.write_all(bytes_l3w)?;

    writer.flush()?;

    let final_size = std::fs::metadata(output_path)?.len();
    assert_eq!(
        final_size, 25_237_864,
        "Exported checkpoint must match KittyEngine exact size (25,237_864 bytes)"
    );

    println!("Successfully exported checkpoint to: {}", output_path.display());
    println!("File size: {} bytes (exact match). Ready for KittyEngine!", final_size);

    Ok(())
}

/// Folds NNUESGD5 (49,152 inputs, 1 head, 50,402,756 bytes) into NNUESGD6 (24,576 inputs, 8 buckets, 25,237,864 bytes).
/// - Accumulator weights are folded across the horizontal symmetry axis:
///   W_folded = 0.5 * (W_left + W_right)
/// - Accumulator biases, L1, L2 are directly preserved.
/// - L3 single-head weights and bias are duplicated across all 8 buckets so that
///   there is ZERO bucket discontinuity at initialization!
pub fn fold_v5_to_v6_warmstart(v5_path: &Path, v6_path: &Path) -> io::Result<()> {
    let old_bytes = std::fs::read(v5_path)?;
    if old_bytes.len() != 50_402_756 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Expected 50,402,756 bytes for NNUESGD5, got {}", old_bytes.len()),
        ));
    }

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

    // Fold accumulator weights from 49,152 to 24,576
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

    if let Some(parent) = v6_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let out_file = File::create(v6_path)?;
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, out_file);

    // Header (72 bytes)
    writer.write_all(MAGIC_V6)?;
    for &dim in &DIMENSIONS {
        writer.write_u64::<LittleEndian>(dim)?;
    }
    for _ in 0..3 {
        writer.write_u64::<LittleEndian>(0)?;
    }

    // Accumulator biases (256 floats)
    let bytes_l0b = unsafe { std::slice::from_raw_parts(l0b.as_ptr() as *const u8, l0b.len() * 4) };
    writer.write_all(bytes_l0b)?;

    // Folded accumulator weights (24576 * 256 floats)
    let bytes_accum = unsafe { std::slice::from_raw_parts(folded_l0w.as_ptr() as *const u8, folded_l0w.len() * 4) };
    writer.write_all(bytes_accum)?;

    // Hidden layer 1 biases & weights
    let bytes_l1b = unsafe { std::slice::from_raw_parts(l1b.as_ptr() as *const u8, l1b.len() * 4) };
    writer.write_all(bytes_l1b)?;
    let bytes_l1w = unsafe { std::slice::from_raw_parts(l1w.as_ptr() as *const u8, l1w.len() * 4) };
    writer.write_all(bytes_l1w)?;

    // Hidden layer 2 biases & weights
    let bytes_l2b = unsafe { std::slice::from_raw_parts(l2b.as_ptr() as *const u8, l2b.len() * 4) };
    writer.write_all(bytes_l2b)?;
    let bytes_l2w = unsafe { std::slice::from_raw_parts(l2w.as_ptr() as *const u8, l2w.len() * 4) };
    writer.write_all(bytes_l2w)?;

    // Output biases (8 floats): replicate old scalar bias across ALL 8 buckets!
    for _ in 0..8 {
        writer.write_f32::<LittleEndian>(l3b)?;
    }

    // Output weights (8 * 32 floats): replicate old 32 weights across ALL 8 buckets!
    for _ in 0..8 {
        for &w in l3w {
            writer.write_f32::<LittleEndian>(w)?;
        }
    }

    writer.flush()?;

    let final_size = std::fs::metadata(v6_path)?.len();
    assert_eq!(
        final_size, 25_237_864,
        "Folded warm-start weights must match KittyEngine NNUESGD6 exact size (25,237,864 bytes)"
    );

    println!(
        "Successfully created warm-started NNUESGD6 weights from NNUESGD5 ({} bytes) at: {}",
        final_size,
        v6_path.display()
    );

    Ok(())
}

fn write_bullet_tensor(writer: &mut impl Write, id: &str, data: &[f32]) -> io::Result<()> {
    writer.write_all(format!("{id}\n").as_bytes())?;
    writer.write_all(&(data.len() as usize).to_le_bytes())?;
    let slice_bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    writer.write_all(slice_bytes)?;
    Ok(())
}

/// Imports KittyEngine's NNUESGD weights (25,237,864 bytes) into Bullet's native tagged weights format.
pub fn import_from_kitty_weights(kitty_path: &Path, bullet_path: &Path) -> io::Result<()> {
    let kitty_bytes = std::fs::read(kitty_path)?;
    if kitty_bytes.len() != 25_237_864 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Expected KittyEngine weights file of 25,237,864 bytes, got {}",
                kitty_bytes.len()
            ),
        ));
    }

    if &kitty_bytes[0..8] != MAGIC_V6 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid KittyEngine signature (expected NNUESGD6)",
        ));
    }

    // Cast payload floats (after 72-byte header)
    let payload_floats: &[f32] = unsafe {
        std::slice::from_raw_parts(
            kitty_bytes[72..].as_ptr() as *const f32,
            (kitty_bytes.len() - 72) / 4,
        )
    };

    let mut offset = 0;

    // 1. l0b: (256)
    let l0b = &payload_floats[offset..offset + 256];
    offset += 256;

    // 2. l0w (KittyEngine input-major layout: input * 256 + neuron)
    let kitty_accum = &payload_floats[offset..offset + 24_576 * 256];
    offset += 24_576 * 256;

    // 3. l1b: (32)
    let l1b = &payload_floats[offset..offset + 32];
    offset += 32;

    // 4. l1w: (32 * 512)
    let l1w = &payload_floats[offset..offset + 32 * 512];
    offset += 32 * 512;

    // 5. l2b: (32)
    let l2b = &payload_floats[offset..offset + 32];
    offset += 32;

    // 6. l2w: (32 * 32)
    let l2w = &payload_floats[offset..offset + 32 * 32];
    offset += 32 * 32;

    // 7. l3b: (8)
    let l3b = &payload_floats[offset..offset + 8];
    offset += 8;

    // 8. l3w: (8 * 32)
    let l3w = &payload_floats[offset..offset + 8 * 32];

    if let Some(parent) = bullet_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let out_file = File::create(bullet_path)?;
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, out_file);

    // Accumulator weights: direct copy (both Bullet and KittyEngine use feature-major)
    write_bullet_tensor(&mut writer, "l0w", kitty_accum)?;
    write_bullet_tensor(&mut writer, "l0b", l0b)?;

    // Transpose l1w from KittyEngine row-major 32x512 to Bullet col-major 512x32
    let mut bullet_l1w = vec![0.0f32; 32 * 512];
    for row in 0..32 {
        for col in 0..512 {
            bullet_l1w[col * 32 + row] = l1w[row * 512 + col];
        }
    }
    write_bullet_tensor(&mut writer, "l1w", &bullet_l1w)?;
    write_bullet_tensor(&mut writer, "l1b", l1b)?;

    // Transpose l2w from KittyEngine row-major 32x32 to Bullet col-major 32x32
    let mut bullet_l2w = vec![0.0f32; 32 * 32];
    for row in 0..32 {
        for col in 0..32 {
            bullet_l2w[col * 32 + row] = l2w[row * 32 + col];
        }
    }
    write_bullet_tensor(&mut writer, "l2w", &bullet_l2w)?;
    write_bullet_tensor(&mut writer, "l2b", l2b)?;

    // Scale output from centipawns down to logit space for Bullet
    let scale = 400.0f32;
    let mut bullet_l3w = vec![0.0f32; 32 * 8];
    for bucket in 0..8 {
        for col in 0..32 {
            bullet_l3w[col * 8 + bucket] = l3w[bucket * 32 + col] / scale;
        }
    }
    let bullet_l3b: Vec<f32> = l3b.iter().map(|&b| b / scale).collect();
    write_bullet_tensor(&mut writer, "l3w", &bullet_l3w)?;
    write_bullet_tensor(&mut writer, "l3b", &bullet_l3b)?;

    writer.flush()?;

    println!(
        "Successfully converted KittyEngine weights -> Bullet format at: {}",
        bullet_path.display()
    );

    Ok(())
}

/// Simple deterministic linear congruential / xorshift PRNG for reproducible weight initialization.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x853c49e6748fea9b } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1 << 24) as f32
    }

    fn sample_normal(&mut self, mean: f32, std_dev: f32) -> f32 {
        let u1 = (self.next_f32() + 1e-7).min(1.0);
        let u2 = self.next_f32();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        mean + z0 * std_dev
    }
}

/// Generates a brand-new scratch NNUESGD6 weight file (25,237,864 bytes) with
/// positive bias initialization on hidden layers to prevent dying SCReLU neurons.
pub fn generate_scratch_kitty_weights(
    output_path: &Path,
    l0_bias_val: f32,
    l1_bias_val: f32,
    l2_bias_val: f32,
) -> io::Result<()> {
    let mut rng = SimpleRng::new(0x20261106_c0ffee);

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let out_file = File::create(output_path)?;
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, out_file);

    // Header (72 bytes)
    writer.write_all(MAGIC_V6)?;
    for &dim in &DIMENSIONS {
        writer.write_u64::<LittleEndian>(dim)?;
    }
    for _ in 0..3 {
        writer.write_u64::<LittleEndian>(0)?;
    }

    // Accumulator biases (256 floats): initialize to l0_bias_val (e.g. 0.05)
    for _ in 0..256 {
        writer.write_f32::<LittleEndian>(l0_bias_val)?;
    }

    // Accumulator weights (24576 * 256 floats): random normal with std 0.015
    for _ in 0..(24_576 * 256) {
        let w = rng.sample_normal(0.0, 0.015);
        writer.write_f32::<LittleEndian>(w)?;
    }

    // Hidden layer 1 biases (32 floats): initialize to l1_bias_val (e.g. 0.35)
    for _ in 0..32 {
        writer.write_f32::<LittleEndian>(l1_bias_val)?;
    }

    // Hidden layer 1 weights (32 * 512 floats): random normal with std 0.02
    for _ in 0..(32 * 512) {
        let w = rng.sample_normal(0.0, 0.02);
        writer.write_f32::<LittleEndian>(w)?;
    }

    // Hidden layer 2 biases (32 floats): initialize to l2_bias_val (e.g. 0.35)
    for _ in 0..32 {
        writer.write_f32::<LittleEndian>(l2_bias_val)?;
    }

    // Hidden layer 2 weights (32 * 32 floats): random normal with std 0.04
    for _ in 0..(32 * 32) {
        let w = rng.sample_normal(0.0, 0.04);
        writer.write_f32::<LittleEndian>(w)?;
    }

    // Output biases (8 floats): 0.0
    for _ in 0..8 {
        writer.write_f32::<LittleEndian>(0.0)?;
    }

    // Output weights (8 * 32 floats): scaled in centipawns (random with std 10.0 cp)
    for _ in 0..(8 * 32) {
        let w = rng.sample_normal(0.0, 10.0);
        writer.write_f32::<LittleEndian>(w)?;
    }

    writer.flush()?;

    let final_size = std::fs::metadata(output_path)?.len();
    assert_eq!(
        final_size, 25_237_864,
        "Generated scratch checkpoint must match KittyEngine exact size (25,237,864 bytes)"
    );

    println!(
        "Successfully generated scratch KittyEngine weights with positive biases (L0={:.2}, L1={:.2}, L2={:.2}) at: {}",
        l0_bias_val, l1_bias_val, l2_bias_val, output_path.display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::NnueWeights;

    #[test]
    fn test_scratch_weights_generation_and_loading() {
        let temp_dir = std::env::temp_dir();
        let kitty_path = temp_dir.join("test_scratch_v6.bin");
        let bullet_path = temp_dir.join("test_bullet_v6.bin");

        // 1. Generate scratch weights
        generate_scratch_kitty_weights(&kitty_path, 0.05, 0.35, 0.35).expect("Scratch generation failed");
        let meta = std::fs::metadata(&kitty_path).expect("Metadata failed");
        assert_eq!(meta.len(), 25_237_864, "Exact file size must match 25,237,864 bytes");

        // 2. Load into NnueWeights
        let weights = NnueWeights::load_from_file(&kitty_path).expect("Loading scratch weights failed");
        assert_eq!(weights.l0b.len(), 256);
        assert_eq!(weights.l0w.len(), 24_576 * 256);
        assert_eq!(weights.l1b.len(), 32);
        assert_eq!(weights.l1w.len(), 32 * 512);
        assert_eq!(weights.l2b.len(), 32);
        assert_eq!(weights.l2w.len(), 32 * 32);
        assert_eq!(weights.l3b.len(), 8);
        assert_eq!(weights.l3w.len(), 8 * 32);

        // Verify positive biases
        for &b in &weights.l0b {
            assert!((b - 0.05).abs() < 1e-6);
        }
        for &b in &weights.l1b {
            assert!((b - 0.35).abs() < 1e-6);
        }
        for &b in &weights.l2b {
            assert!((b - 0.35).abs() < 1e-6);
        }
        for &b in &weights.l3b {
            assert_eq!(b, 0.0);
        }

        // 3. Convert to Bullet format
        import_from_kitty_weights(&kitty_path, &bullet_path).expect("Import to bullet failed");
        assert!(bullet_path.exists());

        // Cleanup
        let _ = std::fs::remove_file(kitty_path);
        let _ = std::fs::remove_file(bullet_path);
    }
}



