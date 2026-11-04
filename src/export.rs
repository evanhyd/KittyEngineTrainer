use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use byteorder::{LittleEndian, WriteBytesExt};

const MAGIC_V5: &[u8; 8] = b"NNUESGD5";
const DIMENSIONS: [u64; 4] = [49_152, 256, 32, 32];

/// Converts Bullet's `raw.bin` into KittyEngine's exact weight file (50,402,756 bytes).
pub fn export_from_raw_checkpoint(raw_bin_path: &Path, output_path: &Path) -> io::Result<()> {
    // Read the entire file into memory instantly (50 MB)
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

    let expected_floats = 49_152 * 256 + 256 + 512 * 32 + 32 + 32 * 32 + 32 + 32 + 1;
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
    // 1. l0w: (256, 49152)
    let l0w_slice = &raw_floats[offset..offset + 256 * 49_152];
    offset += 256 * 49_152;

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

    // 7. l3w: (1, 32)
    let l3w = &raw_floats[offset..offset + 32];
    offset += 32;

    // 8. l3b: (1, 1)
    let l3b = raw_floats[offset];

    let out_file = File::create(output_path)?;
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, out_file);

    // Header (64 bytes)
    writer.write_all(MAGIC_V5)?;
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

    // Accumulator weights (49152 * 256 floats) - Direct copy: both Bullet and KittyEngine use feature-major
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

    // Output bias & weights (scale by 400.0 from logit space to centipawns for KittyEngine)
    let scale = 400.0f32;
    writer.write_f32::<LittleEndian>(l3b * scale)?;
    let scaled_l3w: Vec<f32> = l3w.iter().map(|&w| w * scale).collect();
    let bytes_l3w = unsafe { std::slice::from_raw_parts(scaled_l3w.as_ptr() as *const u8, scaled_l3w.len() * 4) };
    writer.write_all(bytes_l3w)?;

    writer.flush()?;

    let final_size = std::fs::metadata(output_path)?.len();
    assert_eq!(
        final_size, 50_402_756,
        "Exported checkpoint must match KittyEngine exact size (50,402,756 bytes)"
    );

    println!("Successfully exported checkpoint to: {}", output_path.display());
    println!("File size: {} bytes (exact match). Ready for KittyEngine!", final_size);

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

/// Imports KittyEngine's NNUESGD weights (50,402,756 bytes) into Bullet's native tagged weights format.
pub fn import_from_kitty_weights(kitty_path: &Path, bullet_path: &Path) -> io::Result<()> {
    let kitty_bytes = std::fs::read(kitty_path)?;
    if kitty_bytes.len() != 50_402_756 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Expected KittyEngine weights file of 50,402,756 bytes, got {}",
                kitty_bytes.len()
            ),
        ));
    }

    if &kitty_bytes[0..8] != MAGIC_V5 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid KittyEngine signature (expected NNUESGD5)",
        ));
    }

    // Cast payload floats (after 64-byte header)
    let payload_floats: &[f32] = unsafe {
        std::slice::from_raw_parts(
            kitty_bytes[64..].as_ptr() as *const f32,
            (kitty_bytes.len() - 64) / 4,
        )
    };

    let mut offset = 0;

    // 1. l0b: (256)
    let l0b = &payload_floats[offset..offset + 256];
    offset += 256;

    // 2. l0w (KittyEngine input-major layout: input * 256 + neuron)
    let kitty_accum = &payload_floats[offset..offset + 49_152 * 256];
    offset += 49_152 * 256;

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

    // 7. l3b: (1)
    let l3b = payload_floats[offset];
    offset += 1;

    // 8. l3w: (32)
    let l3w = &payload_floats[offset..offset + 32];

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
    let bullet_l3w: Vec<f32> = l3w.iter().map(|&w| w / scale).collect();
    write_bullet_tensor(&mut writer, "l3w", &bullet_l3w)?;
    write_bullet_tensor(&mut writer, "l3b", &[l3b / scale])?;

    writer.flush()?;

    println!(
        "Successfully converted KittyEngine weights -> Bullet format at: {}",
        bullet_path.display()
    );

    Ok(())
}

