//! Lightweight terminal progress reporting for non-Burn model adapters.

use std::io::{self, IsTerminal, Write};

use crate::error::Ms2AtomsError;

/// Receives progress updates from model adapters that do not provide their own UI.
pub(crate) trait Reporter {
    /// Reports a plain status message.
    ///
    /// # Errors
    /// - Returns [`Ms2AtomsError`] if writing the progress update fails.
    fn report(&mut self, message: &str) -> Result<(), Ms2AtomsError>;

    /// Reports progress through a known number of steps.
    ///
    /// # Errors
    /// - Returns [`Ms2AtomsError`] if writing the progress update fails.
    fn report_step(
        &mut self,
        completed: usize,
        total: usize,
        message: &str,
    ) -> Result<(), Ms2AtomsError>;

    /// Finishes the progress view and moves to the next terminal line.
    ///
    /// # Errors
    /// - Returns [`Ms2AtomsError`] if writing the final progress update fails.
    fn finish(&mut self, message: &str) -> Result<(), Ms2AtomsError>;
}

/// Progress reporter that renders one reusable terminal line when stderr is a TTY.
pub(crate) struct TerminalReporter {
    label: String,
    dynamic: bool,
    active_line_width: usize,
}

impl TerminalReporter {
    /// Creates a progress reporter.
    #[must_use]
    pub(crate) fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            dynamic: io::stderr().is_terminal(),
            active_line_width: 0,
        }
    }

    fn render(&mut self, message: &str, end_line: bool) -> Result<(), Ms2AtomsError> {
        let line = self.line(message);
        let line_width = line.chars().count();
        let padding_width = self.active_line_width.saturating_sub(line_width);
        let padding = " ".repeat(padding_width);
        let mut stderr = io::stderr().lock();

        if self.dynamic {
            if end_line {
                writeln!(stderr, "\r{line}{padding}")?;
                self.active_line_width = 0;
            } else {
                write!(stderr, "\r{line}{padding}")?;
                stderr.flush()?;
                self.active_line_width = line_width;
            }
        } else {
            writeln!(stderr, "{line}")?;
        }

        Ok(())
    }

    fn line(&self, message: &str) -> String {
        format!("{} | {message}", self.label)
    }
}

impl Reporter for TerminalReporter {
    fn report(&mut self, message: &str) -> Result<(), Ms2AtomsError> {
        self.render(message, false)
    }

    fn report_step(
        &mut self,
        completed: usize,
        total: usize,
        message: &str,
    ) -> Result<(), Ms2AtomsError> {
        let capped_completed = completed.min(total);
        let percent = if total == 0 {
            100
        } else {
            let Some(quotient) = capped_completed.saturating_mul(100).checked_div(total) else {
                return Err(Ms2AtomsError::Arithmetic(format!(
                    "{capped_completed} / {total}"
                )));
            };
            quotient
        };

        self.render(
            &format!("[{capped_completed}/{total} {percent}%] {message}"),
            false,
        )
    }

    fn finish(&mut self, message: &str) -> Result<(), Ms2AtomsError> {
        self.render(message, true)
    }
}
