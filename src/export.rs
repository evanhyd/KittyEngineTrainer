use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use byteorder::{LittleEndian, WriteBytesExt};

const MAGIC_V6: &[u8; 8] = b"NNUESGD6";
const DIMENSIONS: [u64; 5] = [24_576, 256, 64, 32, 8];
pub const EXACT_FILE_SIZE: u64 = crate::config::CHECKPOINT_SIZE;

/// Diagnostic telemetry and statistics from quantization range checking.
#[derive(Debug, Clone, Default)]
pub struct QuantizationReport {
    pub l0_min: f32,
    pub l0_max: f32,
    pub l0_clipped: usize,
    pub l1_min: f32,
    pub l1_max: f32,
    pub l1_clipped: usize,
    pub l2_min: f32,
    pub l2_max: f32,
    pub l2_clipped: usize,
    pub l3_min: f32,
    pub l3_max: f32,
    pub l3_clipped: usize,
}

impl QuantizationReport {
    pub fn total_clipped(&self) -> usize {
        self.l0_clipped + self.l1_clipped + self.l2_clipped + self.l3_clipped
    }
}

/// Guards neural network weights and biases to ensure safe integer quantization without clipping distortion.
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
    // Quantization limits:
    // L0: scale 64.0 -> int16_t range [-32768, 32767] -> [-511.98, +511.98]
    const L0_LIMIT: f32 = 511.98;
    // L1 & L2: scale 64.0 -> int8_t range [-128, 127] -> [-1.984, +1.984]
    const L1_LIMIT: f32 = 1.984;
    const L2_LIMIT: f32 = 1.984;
    // L3: scale 16.0 (in cp) -> int16_t range [-32768, 32767] -> [-2047.0, +2047.0]
    const L3_LIMIT: f32 = 2047.0;

    let mut report = QuantizationReport {
        l0_min: f32::MAX, l0_max: f32::MIN, l0_clipped: 0,
        l1_min: f32::MAX, l1_max: f32::MIN, l1_clipped: 0,
        l2_min: f32::MAX, l2_max: f32::MIN, l2_clipped: 0,
        l3_min: f32::MAX, l3_max: f32::MIN, l3_clipped: 0,
    };

    // 1. Guard L0 weights & biases
    for w in l0w.iter_mut().chain(l0b.iter_mut()) {
        report.l0_min = report.l0_min.min(*w);
        report.l0_max = report.l0_max.max(*w);
        if *w > L0_LIMIT {
            *w = L0_LIMIT;
            report.l0_clipped += 1;
        } else if *w < -L0_LIMIT {
            *w = -L0_LIMIT;
            report.l0_clipped += 1;
        }
    }

    // 2. Guard L1 weights & biases
    for w in l1w.iter_mut() {
        report.l1_min = report.l1_min.min(*w);
        report.l1_max = report.l1_max.max(*w);
        if *w > L1_LIMIT {
            *w = L1_LIMIT;
            report.l1_clipped += 1;
        } else if *w < -L1_LIMIT {
            *w = -L1_LIMIT;
            report.l1_clipped += 1;
        }
    }
    for b in l1b.iter_mut() {
        // L1 biases quantized to int32 (scale 8192) - check safe float range +-260,000
        report.l1_min = report.l1_min.min(*b);
        report.l1_max = report.l1_max.max(*b);
        if *b > 260_000.0 {
            *b = 260_000.0;
            report.l1_clipped += 1;
        } else if *b < -260_000.0 {
            *b = -260_000.0;
            report.l1_clipped += 1;
        }
    }

    // 3. Guard L2 weights & biases
    for w in l2w.iter_mut() {
        report.l2_min = report.l2_min.min(*w);
        report.l2_max = report.l2_max.max(*w);
        if *w > L2_LIMIT {
            *w = L2_LIMIT;
            report.l2_clipped += 1;
        } else if *w < -L2_LIMIT {
            *w = -L2_LIMIT;
            report.l2_clipped += 1;
        }
    }
    for b in l2b.iter_mut() {
        // L2 biases quantized to int32 (scale 8192)
        report.l2_min = report.l2_min.min(*b);
        report.l2_max = report.l2_max.max(*b);
        if *b > 260_000.0 {
            *b = 260_000.0;
            report.l2_clipped += 1;
        } else if *b < -260_000.0 {
            *b = -260_000.0;
            report.l2_clipped += 1;
        }
    }

    // 4. Guard L3 weights & biases (in centipawns)
    for w in l3w.iter_mut() {
        report.l3_min = report.l3_min.min(*w);
        report.l3_max = report.l3_max.max(*w);
        if *w > L3_LIMIT {
            *w = L3_LIMIT;
            report.l3_clipped += 1;
        } else if *w < -L3_LIMIT {
            *w = -L3_LIMIT;
            report.l3_clipped += 1;
        }
    }
    for b in l3b.iter_mut() {
        // L3 biases quantized to int32 (scale 2048)
        report.l3_min = report.l3_min.min(*b);
        report.l3_max = report.l3_max.max(*b);
        if *b > 1_000_000.0 {
            *b = 1_000_000.0;
            report.l3_clipped += 1;
        } else if *b < -1_000_000.0 {
            *b = -1_000_000.0;
            report.l3_clipped += 1;
        }
    }

    report
}

/// Converts Bullet's `raw.bin` into KittyEngine's exact weight file (25,242,088 bytes).
pub fn export_from_raw_checkpoint(raw_bin_path: &Path, output_path: &Path) -> io::Result<()> {
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

    let expected_floats = 24_576 * 256 + 256 + 256 * 64 + 64 + 64 * 32 + 32 + 32 * 8 + 8;
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

    // 3. l1w: (64, 256)
    let l1w = &raw_floats[offset..offset + 64 * 256];
    offset += 64 * 256;

    // 4. l1b: (64, 1)
    let l1b = &raw_floats[offset..offset + 64];
    offset += 64;

    // 5. l2w: (32, 64)
    let l2w = &raw_floats[offset..offset + 32 * 64];
    offset += 32 * 64;

    // 6. l2b: (32, 1)
    let l2b = &raw_floats[offset..offset + 32];
    offset += 32;

    // 7. l3w: (8, 32)
    let l3w = &raw_floats[offset..offset + 8 * 32];
    offset += 8 * 32;

    // 8. l3b: (8, 1)
    let l3b = &raw_floats[offset..offset + 8];

    // Transpose from Bullet col-major to KittyEngine row-major:
    let mut kitty_l0w = l0w_slice.to_vec();
    let mut kitty_l0b = l0b.to_vec();

    let mut kitty_l1w = vec![0.0f32; 64 * 256];
    for row in 0..64 {
        for col in 0..256 {
            kitty_l1w[row * 256 + col] = l1w[col * 64 + row];
        }
    }
    let mut kitty_l1b = l1b.to_vec();

    let mut kitty_l2w = vec![0.0f32; 32 * 64];
    for row in 0..32 {
        for col in 0..64 {
            kitty_l2w[row * 64 + col] = l2w[col * 32 + row];
        }
    }
    let mut kitty_l2b = l2b.to_vec();

    // Scale output from Bullet logit space to centipawns
    let scale = 400.0f32;
    let mut kitty_l3b: Vec<f32> = l3b.iter().map(|&b| b * scale).collect();
    let mut kitty_l3w = vec![0.0f32; 8 * 32];
    for bucket in 0..8 {
        for col in 0..32 {
            kitty_l3w[bucket * 32 + col] = l3w[col * 8 + bucket] * scale;
        }
    }

    // Apply Quantization Range Guards
    let report = guard_quantization_ranges(
        &mut kitty_l0w,
        &mut kitty_l0b,
        &mut kitty_l1w,
        &mut kitty_l1b,
        &mut kitty_l2w,
        &mut kitty_l2b,
        &mut kitty_l3w,
        &mut kitty_l3b,
    );

    if report.total_clipped() > 0 {
        eprintln!(
            "⚠️ [quantization-guard] Clipped {} weights to guarantee safe quantization (L0:{}, L1:{}, L2:{}, L3:{})",
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
        let bytes_l0b = unsafe { std::slice::from_raw_parts(kitty_l0b.as_ptr() as *const u8, kitty_l0b.len() * 4) };
        writer.write_all(bytes_l0b)?;

        // Accumulator weights (24576 * 256 floats)
        let bytes_accum = unsafe { std::slice::from_raw_parts(kitty_l0w.as_ptr() as *const u8, kitty_l0w.len() * 4) };
        writer.write_all(bytes_accum)?;

        // Hidden layer 1 biases & weights (64 biases, 64x256 weights)
        let bytes_l1b = unsafe { std::slice::from_raw_parts(kitty_l1b.as_ptr() as *const u8, kitty_l1b.len() * 4) };
        writer.write_all(bytes_l1b)?;
        let bytes_l1w = unsafe { std::slice::from_raw_parts(kitty_l1w.as_ptr() as *const u8, kitty_l1w.len() * 4) };
        writer.write_all(bytes_l1w)?;

        // Hidden layer 2 biases & weights (32 biases, 32x64 weights)
        let bytes_l2b = unsafe { std::slice::from_raw_parts(kitty_l2b.as_ptr() as *const u8, kitty_l2b.len() * 4) };
        writer.write_all(bytes_l2b)?;
        let bytes_l2w = unsafe { std::slice::from_raw_parts(kitty_l2w.as_ptr() as *const u8, kitty_l2w.len() * 4) };
        writer.write_all(bytes_l2w)?;

        // Output biases & weights (8 biases, 8x32 weights)
        let bytes_l3b = unsafe { std::slice::from_raw_parts(kitty_l3b.as_ptr() as *const u8, kitty_l3b.len() * 4) };
        writer.write_all(bytes_l3b)?;
        let bytes_l3w = unsafe { std::slice::from_raw_parts(kitty_l3w.as_ptr() as *const u8, kitty_l3w.len() * 4) };
        writer.write_all(bytes_l3w)?;

        writer.flush()?;
        writer.into_inner()?.sync_all()?;
    }

    let final_size = std::fs::metadata(&tmp_path)?.len();
    assert_eq!(
        final_size, EXACT_FILE_SIZE,
        "Exported checkpoint must match KittyEngine exact size (25,242,088 bytes)"
    );

    if output_path.exists() {
        let _ = std::fs::remove_file(output_path);
    }
    std::fs::rename(&tmp_path, output_path)?;

    Ok(())
}

fn write_bullet_tensor(writer: &mut impl Write, id: &str, data: &[f32]) -> io::Result<()> {
    writer.write_all(format!("{id}\n").as_bytes())?;
    writer.write_all(&data.len().to_le_bytes())?;
    let slice_bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    writer.write_all(slice_bytes)?;
    Ok(())
}

/// Imports KittyEngine's NNUESGD weights (25,242,088 bytes) into Bullet's native tagged weights format.
pub fn import_from_kitty_weights(kitty_path: &Path, bullet_path: &Path) -> io::Result<()> {
    let kitty_bytes = std::fs::read(kitty_path)?;
    if kitty_bytes.len() != EXACT_FILE_SIZE as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Expected KittyEngine weights file of {} bytes, got {}",
                EXACT_FILE_SIZE,
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

    // 2. l0w: (24576 * 256)
    let kitty_accum = &payload_floats[offset..offset + 24_576 * 256];
    offset += 24_576 * 256;

    // 3. l1b: (64)
    let l1b = &payload_floats[offset..offset + 64];
    offset += 64;

    // 4. l1w: (64 * 256)
    let l1w = &payload_floats[offset..offset + 64 * 256];
    offset += 64 * 256;

    // 5. l2b: (32)
    let l2b = &payload_floats[offset..offset + 32];
    offset += 32;

    // 6. l2w: (32 * 64)
    let l2w = &payload_floats[offset..offset + 32 * 64];
    offset += 32 * 64;

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

    // Accumulator weights: direct copy
    write_bullet_tensor(&mut writer, "l0w", kitty_accum)?;
    write_bullet_tensor(&mut writer, "l0b", l0b)?;

    // Transpose l1w from KittyEngine row-major 64x256 to Bullet col-major 256x64
    let mut bullet_l1w = vec![0.0f32; 64 * 256];
    for row in 0..64 {
        for col in 0..256 {
            bullet_l1w[col * 64 + row] = l1w[row * 256 + col];
        }
    }
    write_bullet_tensor(&mut writer, "l1w", &bullet_l1w)?;
    write_bullet_tensor(&mut writer, "l1b", l1b)?;

    // Transpose l2w from KittyEngine row-major 32x64 to Bullet col-major 64x32
    let mut bullet_l2w = vec![0.0f32; 32 * 64];
    for row in 0..32 {
        for col in 0..64 {
            bullet_l2w[col * 32 + row] = l2w[row * 64 + col];
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

/// Generates a brand-new scratch NNUESGD6 weight file (25,242,088 bytes) with
/// positive bias initialization on hidden layers to prevent dying SCReLU neurons.
pub fn generate_scratch_kitty_weights(
    output_path: &Path,
    l0_bias_val: f32,
    l1_bias_val: f32,
    l2_bias_val: f32,
) -> io::Result<()> {
    let mut rng = SimpleRng::new(0x0020_2611_06c0_ffee);

    let tmp_path = output_path.with_extension("tmp");
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    {
        let out_file = File::create(&tmp_path)?;
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

        // Hidden layer 1 biases (64 floats): initialize to l1_bias_val (e.g. 0.35)
        for _ in 0..64 {
            writer.write_f32::<LittleEndian>(l1_bias_val)?;
        }

        // Hidden layer 1 weights (64 * 256 floats): random normal with std 0.02
        for _ in 0..(64 * 256) {
            let w = rng.sample_normal(0.0, 0.02);
            writer.write_f32::<LittleEndian>(w)?;
        }

        // Hidden layer 2 biases (32 floats): initialize to l2_bias_val (e.g. 0.35)
        for _ in 0..32 {
            writer.write_f32::<LittleEndian>(l2_bias_val)?;
        }

        // Hidden layer 2 weights (32 * 64 floats): random normal with std 0.04
        for _ in 0..(32 * 64) {
            let w = rng.sample_normal(0.0, 0.04);
            writer.write_f32::<LittleEndian>(w)?;
        }

        // Output biases (8 floats): 0.0
        for _ in 0..8 {
            writer.write_f32::<LittleEndian>(0.0)?;
        }

        // Output weights (8 * 32 floats): scaled in centipawns (std 35.0 cp provides strong backprop gradient to Layer 2)
        for _ in 0..(8 * 32) {
            let w = rng.sample_normal(0.0, 35.0);
            writer.write_f32::<LittleEndian>(w)?;
        }

        writer.flush()?;
        writer.into_inner()?.sync_all()?;
    }

    let final_size = std::fs::metadata(&tmp_path)?.len();
    assert_eq!(
        final_size, EXACT_FILE_SIZE,
        "Generated scratch checkpoint must match KittyEngine exact size (25,242,088 bytes)"
    );

    if output_path.exists() {
        let _ = std::fs::remove_file(output_path);
    }
    std::fs::rename(&tmp_path, output_path)?;

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
        let kitty_path = temp_dir.join("test_scratch_pairwise_v6.bin");
        let bullet_path = temp_dir.join("test_bullet_pairwise_v6.bin");

        // 1. Generate scratch weights
        generate_scratch_kitty_weights(&kitty_path, 0.05, 0.35, 0.35).expect("Scratch generation failed");
        let meta = std::fs::metadata(&kitty_path).expect("Metadata failed");
        assert_eq!(meta.len(), EXACT_FILE_SIZE, "Exact file size must match 25,242,088 bytes");

        // 2. Load into NnueWeights
        let weights = NnueWeights::load_from_file(&kitty_path).expect("Loading scratch weights failed");
        assert_eq!(weights.num_inputs, 24_576);
        assert_eq!(weights.hidden1_size, 64);
        assert_eq!(weights.l1_input_size, 256);
        assert_eq!(weights.l0b.len(), 256);
        assert_eq!(weights.l0w.len(), 24_576 * 256);
        assert_eq!(weights.l1b.len(), 64);
        assert_eq!(weights.l1w.len(), 64 * 256);
        assert_eq!(weights.l2b.len(), 32);
        assert_eq!(weights.l2w.len(), 32 * 64);
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

    #[test]
    fn test_quantization_guard_clamping() {
        let mut l0w = vec![600.0, -600.0, 10.0];
        let mut l0b = vec![0.0];
        let mut l1w = vec![2.5, -3.0, 0.5];
        let mut l1b = vec![0.0];
        let mut l2w = vec![0.1, -2.5];
        let mut l2b = vec![0.0];
        let mut l3w = vec![2500.0, -2500.0, 100.0];
        let mut l3b = vec![0.0];

        let report = guard_quantization_ranges(
            &mut l0w, &mut l0b,
            &mut l1w, &mut l1b,
            &mut l2w, &mut l2b,
            &mut l3w, &mut l3b,
        );

        assert_eq!(report.l0_clipped, 2);
        assert_eq!(report.l1_clipped, 2);
        assert_eq!(report.l2_clipped, 1);
        assert_eq!(report.l3_clipped, 2);

        assert!((l0w[0] - 511.98).abs() < 1e-4);
        assert!((l0w[1] - (-511.98)).abs() < 1e-4);
        assert!((l1w[0] - 1.984).abs() < 1e-4);
        assert!((l1w[1] - (-1.984)).abs() < 1e-4);
        assert!((l2w[1] - (-1.984)).abs() < 1e-4);
        assert!((l3w[0] - 2047.0).abs() < 1e-4);
        assert!((l3w[1] - (-2047.0)).abs() < 1e-4);
    }
}
