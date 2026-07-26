use std::path::PathBuf;

use anyhow::Result;
use clap::{Subcommand, ValueEnum};
use tongues_data::speech_corpus::{
    prepare_speech_corpus_with_progress, PrepareSpeechCorpusConfig, PrepareSpeechProgress,
    SpeechBatchConfig, SpeechCorpusFormat, SpeechSplitConfig, SplitUnit,
};

#[derive(Debug, Subcommand)]
pub enum SpeechCorpusCommands {
    /// Normalize a raw corpus into deterministic manifests and batch plans
    Prepare {
        /// Raw LJSpeech, VCTK, or generic corpus root
        #[arg(long)]
        input: PathBuf,

        /// Prepared dataset output directory
        #[arg(long)]
        out: PathBuf,

        /// Source corpus layout
        #[arg(long, value_enum)]
        format: SpeechCorpusFormatArg,

        /// Metadata path, relative to input unless absolute
        #[arg(long)]
        metadata: Option<PathBuf>,

        /// BCP-47 language/variety recorded in normalized rows
        #[arg(long, default_value = "en-US")]
        language: String,

        /// Fraction assigned to training
        #[arg(long, default_value_t = 0.8)]
        train_fraction: f64,

        /// Fraction assigned to validation
        #[arg(long, default_value_t = 0.1)]
        valid_fraction: f64,

        /// Stable split and batch seed
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Keep every speaker wholly within one split
        #[arg(long)]
        split_by_speaker: bool,

        /// Maximum utterances per batch
        #[arg(long, default_value_t = 16)]
        batch_size: usize,

        /// Maximum aggregate source-audio samples per batch; zero disables
        #[arg(long, default_value_t = 0)]
        max_batch_samples: u64,

        /// Audio-sample width used for length buckets
        #[arg(long, default_value_t = 22_050)]
        bucket_samples: u64,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum SpeechCorpusFormatArg {
    Ljspeech,
    Vctk,
    Generic,
}

impl From<SpeechCorpusFormatArg> for SpeechCorpusFormat {
    fn from(value: SpeechCorpusFormatArg) -> Self {
        match value {
            SpeechCorpusFormatArg::Ljspeech => Self::Ljspeech,
            SpeechCorpusFormatArg::Vctk => Self::Vctk,
            SpeechCorpusFormatArg::Generic => Self::Generic,
        }
    }
}

pub fn run_speech_corpus_command(command: SpeechCorpusCommands, quiet: bool) -> Result<()> {
    match command {
        SpeechCorpusCommands::Prepare {
            input,
            out,
            format,
            metadata,
            language,
            train_fraction,
            valid_fraction,
            seed,
            split_by_speaker,
            batch_size,
            max_batch_samples,
            bucket_samples,
        } => {
            let config = PrepareSpeechCorpusConfig {
                format: format.into(),
                metadata_path: metadata,
                language,
                split: SpeechSplitConfig {
                    train_fraction,
                    valid_fraction,
                    seed,
                    unit: if split_by_speaker {
                        SplitUnit::Speaker
                    } else {
                        SplitUnit::Utterance
                    },
                },
                batch: SpeechBatchConfig {
                    max_items: batch_size,
                    max_audio_samples: max_batch_samples,
                    bucket_width_samples: bucket_samples,
                    seed,
                },
            };
            let report = prepare_speech_corpus_with_progress(&input, &out, &config, |event| {
                if !quiet {
                    eprintln!("{}", format_progress(event));
                }
            })?;
            println!(
                "Prepared {} records at {}: {} train / {} valid / {} test; batches {}/{}/{}",
                report.records,
                report.output.display(),
                report.train,
                report.valid,
                report.test,
                report.train_batches,
                report.valid_batches,
                report.test_batches
            );
            Ok(())
        }
    }
}

fn format_progress(progress: PrepareSpeechProgress) -> String {
    match progress {
        PrepareSpeechProgress::Scan { format, root } => {
            format!("Scanning {format:?} corpus at {}", root.display())
        }
        PrepareSpeechProgress::Validate { checked, total } => {
            format!("Validated {checked}/{total} records")
        }
        PrepareSpeechProgress::Split { train, valid, test } => {
            format!("Split records: {train} train / {valid} valid / {test} test")
        }
        PrepareSpeechProgress::Batch { split, batches } => {
            format!("Planned {batches} length-aware {split} batches")
        }
        PrepareSpeechProgress::Write { rows, path } => {
            format!("Wrote {rows} rows to {}", path.display())
        }
        PrepareSpeechProgress::Complete { output, records } => {
            format!("Prepared {records} records at {}", output.display())
        }
    }
}
