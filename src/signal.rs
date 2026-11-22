use std::path::Path;

#[cfg(windows)]
mod imp {
    use std::sync::atomic::{AtomicBool, Ordering};

    static STOP_FLAG: AtomicBool = AtomicBool::new(false);
    static FORCE_EXIT: AtomicBool = AtomicBool::new(false);

    unsafe extern "system" fn ctrl_handler(ctrl_type: u32) -> i32 {
        // 0 = CTRL_C_EVENT, 1 = CTRL_BREAK_EVENT, 2 = CTRL_CLOSE_EVENT
        if ctrl_type <= 2 {
            if FORCE_EXIT.swap(true, Ordering::SeqCst) {
                eprintln!("\n⚠️ Force termination requested by user. Exiting immediately.");
                std::process::exit(1);
            }
            eprintln!("\n🛑 [STOP SIGNAL DETECTED: Ctrl+C]");
            eprintln!("   Graceful shutdown initiated! Completing the active superbatch and cleanly flushing");
            eprintln!("   verified weights to prevent file corruption.");
            eprintln!("   (Press Ctrl+C again if you wish to force immediate termination)\n");
            STOP_FLAG.store(true, Ordering::SeqCst);
            1 // 1 tells Windows the signal was handled; do NOT terminate process abruptly
        } else {
            0
        }
    }

    unsafe extern "system" {
        fn SetConsoleCtrlHandler(
            handler: Option<unsafe extern "system" fn(u32) -> i32>,
            add: i32,
        ) -> i32;
    }

    pub fn init_ctrl_handler() {
        unsafe {
            SetConsoleCtrlHandler(Some(ctrl_handler), 1);
        }
    }

    pub fn is_stop_requested() -> bool {
        STOP_FLAG.load(Ordering::SeqCst)
    }

    pub fn request_stop() {
        STOP_FLAG.store(true, Ordering::SeqCst);
    }
}

#[cfg(not(windows))]
mod imp {
    use std::sync::atomic::{AtomicBool, Ordering};
    static STOP_FLAG: AtomicBool = AtomicBool::new(false);

    pub fn init_ctrl_handler() {
        // Fallback for non-Windows platforms
    }

    pub fn is_stop_requested() -> bool {
        STOP_FLAG.load(Ordering::SeqCst)
    }

    pub fn request_stop() {
        STOP_FLAG.store(true, Ordering::SeqCst);
    }
}

pub const STOP_TRIGGER_FILE: &str = "checkpoints/stop";
pub const STOP_FLAG_FILE: &str = "stop.flag";

/// Initializes the manual stop signal system:
/// 1. Installs OS console control handlers (Ctrl+C, Ctrl+Break, Close window).
/// 2. Spawns an interactive stdin listener thread detecting `stop`, `quit`, `exit`, or `q`.
pub fn init() {
    imp::init_ctrl_handler();

    // Stdin listener thread for interactive CLI usage
    std::thread::spawn(move || {
        use std::io::BufRead;
        let stdin = std::io::stdin();
        let mut reader = stdin.lock();
        let mut line = String::new();
        while reader.read_line(&mut line).is_ok() {
            let trimmed = line.trim().to_lowercase();
            if trimmed == "stop" || trimmed == "quit" || trimmed == "exit" || trimmed == "q" {
                println!("\n🛑 [STOP COMMAND RECEIVED: '{}']", trimmed);
                println!("   Graceful shutdown initiated! Completing the active superbatch and cleanly flushing weights...");
                imp::request_stop();
                break;
            }
            line.clear();
        }
    });
}

/// Checks whether a stop signal has been triggered via:
/// 1. Ctrl+C console event.
/// 2. Console input command ('stop' / 'q' / 'exit').
/// 3. Stop trigger file on disk ('checkpoints/stop' or 'stop.flag').
pub fn is_stop_requested() -> bool {
    if imp::is_stop_requested() {
        return true;
    }
    if Path::new(STOP_TRIGGER_FILE).exists() || Path::new(STOP_FLAG_FILE).exists() {
        imp::request_stop();
        return true;
    }
    false
}

/// Creates a stop trigger file on disk (`checkpoints/stop`) to signal any running trainer.
pub fn create_stop_file() -> std::io::Result<()> {
    if let Some(parent) = Path::new(STOP_TRIGGER_FILE).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(STOP_TRIGGER_FILE, b"STOP\n")?;
    Ok(())
}

/// Cleans up any stop trigger files from disk.
pub fn clear_stop_files() {
    let _ = std::fs::remove_file(STOP_TRIGGER_FILE);
    let _ = std::fs::remove_file(STOP_FLAG_FILE);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stop_signal_file() {
        clear_stop_files();
        assert!(!is_stop_requested());

        create_stop_file().expect("Failed to create stop file");
        assert!(is_stop_requested());

        clear_stop_files();
        assert!(!Path::new(STOP_TRIGGER_FILE).exists());
    }
}
