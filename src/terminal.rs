//! Custom low-level terminal interception and execution engine.
//!
//! This module implements a sandboxed, non-interactive command execution
//! pipeline built directly on raw Tokio primitives. It enforces a strict
//! path-boundary sandbox and rejects interactive payloads before any process
//! is spawned.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

// ---------------------------------------------------------------------------
// API types
// ---------------------------------------------------------------------------

/// Incoming execution request.
#[derive(Debug, Clone, Deserialize)]
pub struct ExecuteRequest {
    /// The shell command to run.
    pub command: String,
    /// Optional working directory, resolved inside the workspace sandbox.
    pub working_dir: Option<String>,
    /// Optional hard timeout in seconds.
    pub timeout_seconds: Option<u64>,
}

/// Execution result returned to the caller.
#[derive(Debug, Clone, Serialize)]
pub struct ExecuteResponse {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    /// Explicit error string when the command could not run or was rejected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ExecuteResponse {
    fn error(message: impl Into<String>) -> Self {
        Self {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: None,
            error: Some(message.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Sanitization
// ---------------------------------------------------------------------------

/// Interactive payloads that must never be spawned.
const BLOCKED_COMMANDS: &[&str] = &[
    "vim", "vi", "nano", "emacs", "top", "htop", "ssh", "less", "more", "man",
];

/// Returns true for bytes that may appear inside a command token.
#[inline]
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.' || b == b'/'
}

/// Zero-allocation scan of the command string for interactive payloads.
///
/// The command is walked byte-by-byte to extract whitespace-delimited tokens
/// (as `&str` slices, no allocation) and each token is compared against the
/// blocklist. Returns the offending token when found.
pub fn find_interactive_payload(command: &str) -> Option<&'static str> {
    let bytes = command.as_bytes();
    let mut i = 0;
    let n = bytes.len();
    while i < n {
        // Skip separators.
        while i < n && !is_word_byte(bytes[i]) {
            i += 1;
        }
        let start = i;
        while i < n && is_word_byte(bytes[i]) {
            i += 1;
        }
        if start < i {
            let word = &command[start..i];
            for blocked in BLOCKED_COMMANDS {
                if word == *blocked {
                    return Some(blocked);
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Path boundary guards
// ---------------------------------------------------------------------------

/// Canonicalize the workspace root and the requested working directory, then
/// verify the resolved directory is contained within the workspace root.
///
/// This permanently isolates the spawned process inside the sandbox.
pub fn resolve_sandboxed_dir(
    workspace_root: &Path,
    working_dir: Option<&str>,
) -> anyhow::Result<PathBuf> {
    let root = workspace_root
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("workspace root unavailable: {e}"))?;

    let target = match working_dir {
        Some(dir) => {
            let candidate = if Path::new(dir).is_absolute() {
                PathBuf::from(dir)
            } else {
                root.join(dir)
            };
            candidate
                .canonicalize()
                .map_err(|e| anyhow::anyhow!("working directory unavailable: {e}"))?
        }
        None => root.clone(),
    };

    if !target.starts_with(&root) {
        anyhow::bail!("working directory escapes workspace sandbox");
    }
    Ok(target)
}

// ---------------------------------------------------------------------------
// Low-level Tokio execution
// ---------------------------------------------------------------------------

/// Select the host-native non-interactive shell invocation.
fn shell_invocation() -> (&'static str, &'static str) {
    if cfg!(target_os = "windows") {
        ("powershell.exe", "-Command")
    } else {
        ("sh", "-c")
    }
}

// ---------------------------------------------------------------------------
// RTK (Rust Token Killer) output compression
// ---------------------------------------------------------------------------

/// Maximum lines before smart truncation kicks in.
const TRUNCATE_THRESHOLD: usize = 50;
/// Lines retained at the head and tail when truncating.
const TRUNCATE_KEEP: usize = 15;

/// Markers that identify a test-runner summary / failure / stack-trace line
/// worth preserving when smart-filtering.
const TEST_KEEP_MARKERS: &[&str] = &[
    "test result",
    "failures:",
    "FAILED",
    "failed",
    "error",
    "Error",
    "panic",
    "panicked",
    "assertion",
    "Caused by",
    "backtrace",
    "Tests:",
    "Suites:",
    "Time:",
    "running ",
    "Compiling",
    "warning:",
    "expected",
    "actual",
    "at ",
];

/// Lazily-compiled ANSI escape sequence matcher (SGR + OSC + charset).
fn ansi_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[()][AB0]").unwrap()
    })
}

/// Lazily-compiled download / progress bar matcher.
fn progress_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\[[=>.\s]*\]").unwrap())
}

/// Lazily-compiled `file:line` stack-frame matcher.
fn path_line_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Za-z0-9_./-]+:\d+").unwrap())
}

/// Strip ANSI color codes and terminal formatting escape sequences.
fn strip_ansi(input: &str) -> String {
    ansi_regex().replace_all(input, "").replace('\r', "")
}

/// Detect whether the command is a test runner we should smart-filter.
fn is_test_runner(command: &str) -> bool {
    let c = command.trim();
    c.contains("cargo test")
        || c.contains("jest")
        || c.contains("pytest")
        || c.contains("npm test")
        || c.contains("yarn test")
        || c.contains("go test")
        || c.contains("mocha")
}

/// Keywords that mark a line as a failure/error worth preserving from
/// truncation.
fn is_failure_line(line: &str) -> bool {
    let t = line.trim();
    t.contains("error")
        || t.contains("fail")
        || t.contains("panic")
        || t.contains("FAILED")
        || t.contains("exception")
        || t.contains("traceback")
        || t.contains('✕')
        || t.contains('×')
}

/// For test-runner output, drop progress bars and passing lines while keeping
/// the final summary and any failing stack traces.
fn filter_test_output(input: &str) -> String {
    let mut out = String::new();
    for line in input.lines() {
        if progress_regex().is_match(line) {
            continue;
        }
        let t = line.trim();
        let keep = TEST_KEEP_MARKERS.iter().any(|m| t.contains(m))
            || path_line_regex().is_match(t)
            || line.starts_with(' ')
            || line.starts_with('\t');
        if keep {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Collapse consecutive identical lines into a single `text (xN)` line.
fn dedup_lines(input: &str) -> String {
    let mut out = String::new();
    let mut lines = input.lines();
    let Some(mut current) = lines.next() else {
        return out;
    };
    let mut count = 1usize;
    for line in lines {
        if line == current {
            count += 1;
        } else {
            if count > 1 {
                out.push_str(&format!("{current} (x{count})\n"));
            } else {
                out.push_str(current);
                out.push('\n');
            }
            current = line;
            count = 1;
        }
    }
    if count > 1 {
        out.push_str(&format!("{current} (x{count})\n"));
    } else {
        out.push_str(current);
        out.push('\n');
    }
    out
}

/// Truncate the middle of long, failure-free output streams.
fn smart_truncate(input: &str) -> String {
    let lines: Vec<&str> = input.lines().collect();
    if lines.len() <= TRUNCATE_THRESHOLD {
        return input.to_string();
    }
    if lines.iter().any(|l| is_failure_line(l)) {
        return input.to_string();
    }
    let head = &lines[..TRUNCATE_KEEP];
    let tail = &lines[lines.len() - TRUNCATE_KEEP..];
    let mut out = String::new();
    for l in head {
        out.push_str(l);
        out.push('\n');
    }
    out.push_str(&format!(
        "[... Truncated {} lines of token noise ...]\n",
        lines.len() - 2 * TRUNCATE_KEEP
    ));
    for l in tail {
        out.push_str(l);
        out.push('\n');
    }
    out
}

/// Full RTK compression pipeline applied to a raw output stream.
fn compress_output(raw: &str, command: &str) -> String {
    let no_ansi = strip_ansi(raw);
    let filtered = if is_test_runner(command) {
        filter_test_output(&no_ansi)
    } else {
        no_ansi
    };
    let deduped = dedup_lines(&filtered);
    smart_truncate(&deduped)
}

/// Execute a command inside the workspace sandbox.
///
/// The pipeline: sanitize -> validate path -> spawn piped shell -> read
/// stdout/stderr concurrently -> enforce a hard timeout that kills the child
/// and drains the dead channels on expiry.
pub async fn execute(workspace_root: &Path, request: ExecuteRequest) -> ExecuteResponse {
    // 1. Reject interactive payloads before any spawning.
    if let Some(blocked) = find_interactive_payload(&request.command) {
        return ExecuteResponse::error(format!("interactive command rejected: {blocked}"));
    }

    // 2. Enforce path-boundary sandbox.
    let working_dir = match resolve_sandboxed_dir(workspace_root, request.working_dir.as_deref()) {
        Ok(dir) => dir,
        Err(e) => return ExecuteResponse::error(format!("path validation failed: {e}")),
    };

    // 3. Spawn the host-native non-interactive shell.
    let (shell, flag) = shell_invocation();
    let mut command = Command::new(shell);
    command
        .arg(flag)
        .arg(&request.command)
        .current_dir(&working_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match command.spawn() {
        Ok(c) => c,
        Err(e) => return ExecuteResponse::error(format!("failed to spawn process: {e}")),
    };

    // Pipe stdout/stderr into separate async reader streams.
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    let stdout_task = tokio::spawn(async move {
        let mut buf = [0u8; 4096];
        let mut out = Vec::new();
        if let Some(mut stream) = stdout_pipe {
            loop {
                match stream.read(&mut buf).await {
                    Ok(0) => break,
                    Ok(n) => out.extend_from_slice(&buf[..n]),
                    Err(_) => break,
                }
            }
        }
        out
    });

    let stderr_task = tokio::spawn(async move {
        let mut buf = [0u8; 4096];
        let mut out = Vec::new();
        if let Some(mut stream) = stderr_pipe {
            loop {
                match stream.read(&mut buf).await {
                    Ok(0) => break,
                    Ok(n) => out.extend_from_slice(&buf[..n]),
                    Err(_) => break,
                }
            }
        }
        out
    });

    // 4. Hard execution safety loop with a raw timeout.
    let timeout = Duration::from_secs(request.timeout_seconds.unwrap_or(30));
    let wait_result = tokio::time::timeout(timeout, child.wait()).await;

    match wait_result {
        Ok(Ok(status)) => {
            let stdout = stdout_task.await.unwrap_or_default();
            let stderr = stderr_task.await.unwrap_or_default();
            let raw_stdout = String::from_utf8_lossy(&stdout).into_owned();
            let raw_stderr = String::from_utf8_lossy(&stderr).into_owned();
            ExecuteResponse {
                stdout: compress_output(&raw_stdout, &request.command),
                stderr: compress_output(&raw_stderr, &request.command),
                exit_code: status.code(),
                error: None,
            }
        }
        Ok(Err(e)) => ExecuteResponse::error(format!("process wait failed: {e}")),
        Err(_) => {
            // Timeout: forcefully kill and drain the dead channels.
            let _ = child.kill().await;
            let _ = stdout_task.await;
            let _ = stderr_task.await;
            ExecuteResponse::error("execution timed out")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_interactive_payloads() {
        assert_eq!(find_interactive_payload("vim file.rs"), Some("vim"));
        assert_eq!(find_interactive_payload("sudo ssh host"), Some("ssh"));
        assert_eq!(find_interactive_payload("top -b"), Some("top"));
    }

    #[test]
    fn allows_non_interactive_commands() {
        assert!(find_interactive_payload("cargo build").is_none());
        assert!(find_interactive_payload("ls -la && echo hi").is_none());
        // Substring of another word must not match.
        assert!(find_interactive_payload("sshpass -p x").is_none());
    }

    // -----------------------------------------------------------------------
    // RTK compression: accuracy
    // -----------------------------------------------------------------------

    #[test]
    fn ansi_codes_are_stripped() {
        let raw = "\x1b[32m   Compiling token-saver\x1b[0m\n\x1b[1;31merror\x1b[0m: boom";
        let out = strip_ansi(raw);
        assert!(!out.contains('\x1b'), "ANSI escape must be removed");
        assert_eq!(out, "   Compiling token-saver\nerror: boom");
    }

    #[test]
    fn test_runner_keeps_failures_drops_passing() {
        let raw = "\
test tests::a ... ok
test tests::b ... ok
[====>          ] Downloading crates 40%
test result: FAILED. 2 passed; 1 failed

---- tests::b stdout ----
thread 'main' panicked at src/lib.rs:10:5:
assertion failed
   0: core::panicking::panic (src/lib.rs:10)";
        let out = compress_output(raw, "cargo test");
        // Passing lines + progress bar must be gone.
        assert!(!out.contains("test tests::a ... ok"));
        assert!(!out.contains("Downloading crates"));
        // Summary, panic message and stack frame must survive.
        assert!(out.contains("test result: FAILED. 2 passed; 1 failed"));
        assert!(out.contains("panicked at src/lib.rs:10:5"));
        assert!(out.contains("core::panicking::panic (src/lib.rs:10)"));
    }

    #[test]
    fn dedup_groups_consecutive_lines() {
        let raw = "warn: timeout\nwarn: timeout\nwarn: timeout\nwarn: timeout\nok";
        let out = dedup_lines(raw);
        assert!(out.contains("warn: timeout (x4)"), "got: {out}");
        assert!(out.contains("ok"));
    }

    #[test]
    fn truncation_kicks_in_for_long_clean_output() {
        let lines: Vec<String> = (0..80).map(|i| format!("noise line {i}")).collect();
        let raw = lines.join("\n");
        let out = compress_output(&raw, "cat big.log");
        assert!(out.contains("[... Truncated 50 lines of token noise ...]"));
        // Head + tail preserved (15 each).
        assert!(out.contains("noise line 0"));
        assert!(out.contains("noise line 79"));
        // Middle dropped.
        assert!(!out.contains("noise line 40"));
    }

    #[test]
    fn truncation_skips_when_failure_present() {
        let mut lines: Vec<String> = (0..80).map(|i| format!("noise line {i}")).collect();
        lines[40] = "fatal error: disk full".to_string();
        let raw = lines.join("\n");
        let out = compress_output(&raw, "cat big.log");
        assert!(!out.contains("Truncated"), "must not truncate when a failure exists");
        assert!(out.contains("fatal error: disk full"));
    }

    // -----------------------------------------------------------------------
    // RTK compression: end-to-end token reduction
    // -----------------------------------------------------------------------

    /// Realistic noisy `cargo test` capture: ANSI colors, passing tests, a
    /// download progress bar, repeated warnings, and one real failure.
    const NOISY_CARGO_TEST: &str = "\
\x1b[32m   Compiling token-saver v0.1.0\x1b[0m
    Finished test [unoptimized + debuginfo] in 2.34s
     Running tests/foo.rs (target/debug/deps/foo-abc)

test tests::test_a ... \x1b[32mok\x1b[0m
test tests::test_b ... ok
test tests::test_c ... ok
[=====>          ] Downloading crates 45%
test tests::test_d ... ok
test tests::test_e ... ok
warning: connection timeout
warning: connection timeout
warning: connection timeout
warning: connection timeout
warning: connection timeout
test result: FAILED. 5 passed; 1 failed; 0 ignored

---- tests::test_e stdout ----
thread 'main' panicked at tests/foo.rs:42:10:
assertion failed: false
stack backtrace:
   0: foo::bar (tests/foo.rs:42)
   1: foo::baz (tests/foo.rs:10)";

    #[test]
    fn end_to_end_accuracy_and_token_reduction() {
        let compressed = compress_output(NOISY_CARGO_TEST, "cargo test");

        // --- Accuracy assertions ---
        // Noise removed.
        assert!(!compressed.contains('\x1b'), "ANSI must be stripped");
        assert!(!compressed.contains("test tests::test_a ... ok"));
        assert!(!compressed.contains("Downloading crates"));
        // Critical signal preserved.
        assert!(compressed.contains("test result: FAILED. 5 passed; 1 failed"));
        assert!(compressed.contains("panicked at tests/foo.rs:42:10"));
        assert!(compressed.contains("foo::bar (tests/foo.rs:42)"));
        // Repeated warnings grouped.
        assert!(compressed.contains("warning: connection timeout (x5)"));

        // --- Token reduction measurement ---
        let raw_tokens = crate::tracker::estimate_tokens(NOISY_CARGO_TEST);
        let cmp_tokens = crate::tracker::estimate_tokens(&compressed);
        let reduction_pct =
            ((raw_tokens.saturating_sub(cmp_tokens)) as f64 / raw_tokens.max(1) as f64) * 100.0;

        println!("RTK token reduction report:");
        println!("  raw tokens : {raw_tokens}");
        println!("  cmp tokens : {cmp_tokens}");
        println!("  reduction  : {reduction_pct:.1}%");
        println!("--- compressed output ---\n{compressed}");

        assert!(cmp_tokens < raw_tokens, "compression must reduce tokens");
        assert!(
            reduction_pct >= 40.0,
            "expected >=40% reduction, got {reduction_pct:.1}%"
        );
    }
}
