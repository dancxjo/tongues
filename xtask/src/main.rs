use rand::seq::SliceRandom;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Duration, Instant};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("new-family") => {
            let family = args
                .next()
                .ok_or_else(|| format!("missing family slug\n\n{}", new_family_usage()))?;
            if args.next().is_some() {
                return Err(format!(
                    "new-family accepts exactly one family slug\n\n{}",
                    new_family_usage()
                ));
            }
            new_family(&family)
        }
        Some("audit-family-maturity") => audit_family_maturity(),
        Some("race") => {
            let args = args.collect::<Vec<_>>();
            if args.iter().any(|arg| arg == "-h" || arg == "--help") {
                print!("{}", race_usage());
                Ok(())
            } else {
                race(args)
            }
        }
        Some("continue") => {
            let args = args.collect::<Vec<_>>();
            if args.iter().any(|arg| arg == "-h" || arg == "--help") {
                print!("{}", continue_usage());
                Ok(())
            } else {
                continue_stream(args)
            }
        }
        Some("speech-demo") => {
            let args = args.collect::<Vec<_>>();
            if args.iter().any(|arg| arg == "-h" || arg == "--help") {
                print!("{}", speech_demo_usage());
                Ok(())
            } else {
                speech_demo(args)
            }
        }
        Some("-h") | Some("--help") | None => {
            print!("{}", usage());
            Ok(())
        }
        Some(command) => Err(format!("unknown xtask command `{command}`\n\n{}", usage())),
    }
}

fn usage() -> &'static str {
    "Usage: cargo xtask <command>\n\nCommands:\n  new-family <family-slug>  Create a non-runnable model-family scaffold\n  audit-family-maturity    Verify runnable-family labels and readiness docs\n  race [options] [words...] Run round-trip inference benchmarks\n  continue [options]       Generate text, phones, and speech chunks continuously\n  speech-demo [options]    Run every speech backend in one resident process\n"
}

#[derive(Debug, Clone, Copy)]
struct EstablishedFamily {
    id: &'static str,
    readiness: &'static str,
    crate_lib: &'static str,
}

const ESTABLISHED_MODEL_FAMILIES: &[EstablishedFamily] = &[
    EstablishedFamily {
        id: "g2p2g",
        readiness: "active",
        crate_lib: "crates/tongues-g2p2g/src/lib.rs",
    },
    EstablishedFamily {
        id: "wiktionary",
        readiness: "active",
        crate_lib: "crates/tongues-wiktionary/src/lib.rs",
    },
    EstablishedFamily {
        id: "sentence-parser",
        readiness: "experimental",
        crate_lib: "crates/tongues-sentence-parser/src/lib.rs",
    },
    EstablishedFamily {
        id: "head2phones",
        readiness: "experimental",
        crate_lib: "crates/tongues-head2phones/src/lib.rs",
    },
    EstablishedFamily {
        id: "common-phone",
        readiness: "experimental",
        crate_lib: "crates/tongues-common-phone/src/lib.rs",
    },
    EstablishedFamily {
        id: "interpretation",
        readiness: "experimental",
        crate_lib: "crates/tongues-interpretation/src/lib.rs",
    },
    EstablishedFamily {
        id: "emotions",
        readiness: "experimental",
        crate_lib: "crates/tongues-emotions/src/lib.rs",
    },
];

fn audit_family_maturity() -> Result<(), String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| "xtask has no workspace parent".to_string())?;
    let mut errors = Vec::new();
    let readme = read_audit_file(root, "README.md", &mut errors);

    for family in ESTABLISHED_MODEL_FAMILIES {
        let crate_source = read_audit_file(root, family.crate_lib, &mut errors);
        if crate_source.to_ascii_lowercase().contains("scaffold") {
            errors.push(format!(
                "{} is established but its public crate surface still contains `scaffold`",
                family.id
            ));
        }

        let expected_status = format!("| `{}` |", family.id);
        let status_matches = readme.lines().any(|line| {
            line.starts_with(&expected_status)
                && line
                    .rsplit('|')
                    .nth(1)
                    .is_some_and(|status| status.trim() == family.readiness)
        });
        if !status_matches {
            errors.push(format!(
                "README.md must list `{}` with code-owned readiness `{}`",
                family.id, family.readiness
            ));
        }
    }

    for path in [
        "crates/tongues-cli/src/main.rs",
        "crates/tongues-server/public/app.js",
        "docs/architecture.md",
        "docs/common-phone.md",
        "docs/interpretation.md",
        "docs/sentence-parser.md",
        "docs/wiktionary.md",
        "README.md",
    ] {
        let text = read_audit_file(root, path, &mut errors);
        if text.to_ascii_lowercase().contains("scaffold") {
            errors.push(format!(
                "{path} labels an established runnable family as a scaffold"
            ));
        }
    }

    if errors.is_empty() {
        println!(
            "Family maturity audit passed for {} established model families.",
            ESTABLISHED_MODEL_FAMILIES.len()
        );
        Ok(())
    } else {
        Err(format!(
            "family maturity audit failed:\n- {}",
            errors.join("\n- ")
        ))
    }
}

fn read_audit_file(root: &Path, relative: &str, errors: &mut Vec<String>) -> String {
    let path = root.join(relative);
    match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            errors.push(format!("could not read {}: {error}", path.display()));
            String::new()
        }
    }
}

fn race_usage() -> &'static str {
    "Usage: cargo xtask race [options] [words...]\n\nOptions:\n  --cpu                         Force CPU inference\n  --skip-build                  Do not build the tongues binary first\n  --g2p2g-model <path>          G2P2G model dir (default: models/g2p2g/openepd-v0)\n  --wiktionary-model <path>     Wiktionary model dir (default: models/wiktionary/enwiktionary-2026-06-01-v0-phones)\n  --wiktionary-config <path>    Wiktionary config (default: configs/wiktionary/default.toml)\n"
}

fn continue_usage() -> &'static str {
    "Usage: cargo xtask continue [options]\n\nOptions:\n  --cpu                  Force CPU for model-backed commands\n  --skip-build           Do not build the tongues binary first\n  --forever              Run until interrupted\n  --chunks <n>           Number of chunks to generate (default: 8)\n  --sleep-ms <n>         Delay between chunks (default: 250)\n  --speak-backend <name> Speech backend passed to `tongues speak` (default: mock)\n  --out-dir <path>       Directory for generated WAV files (default: runs/head2phones/continue)\n"
}

fn speech_demo_usage() -> &'static str {
    "Usage: cargo xtask speech-demo [options]\n\nOptions:\n  --cpu                  Force CPU inference\n  --skip-build           Reuse the existing target/release/tongues binary\n  --output-dir <path>    Write WAVs instead of playing audio\n  --timings              Emit startup and inference timing JSON\n  --quality <preset>     StyleTTS2 quality: fast (default) or balanced\n  --quiet                Silence normal CLI progress output\n  --verbose              Show device and diagnostic progress output\n"
}

#[derive(Debug)]
struct RaceConfig {
    cpu: bool,
    skip_build: bool,
    g2p2g_model: PathBuf,
    wiktionary_model: PathBuf,
    wiktionary_config: PathBuf,
    words: Vec<String>,
}

#[derive(Debug)]
struct ContinueConfig {
    cpu: bool,
    skip_build: bool,
    forever: bool,
    chunks: usize,
    sleep_ms: u64,
    speak_backend: String,
    out_dir: PathBuf,
}

#[derive(Debug)]
struct RaceResult {
    output: String,
    elapsed: Duration,
}

#[derive(Debug)]
struct RaceStats {
    runs: usize,
    failures: usize,
    total: Duration,
}

#[derive(Debug)]
struct Scorecard {
    rows: Vec<ScorecardRow>,
}

#[derive(Debug)]
struct ScorecardRow {
    behavior: &'static str,
    passed: usize,
    total: usize,
    notes: Vec<String>,
}

#[derive(Debug)]
struct ScorecardProbe {
    behavior: &'static str,
    passed: bool,
    note: String,
}

struct WiktionaryInferDemo<'a> {
    label: &'a str,
    task: &'a str,
    lang: &'a str,
    notation: &'a str,
    variety: Option<&'a str>,
    raw: bool,
    input: String,
}

struct WiktionaryRoundTripCase {
    word: String,
    lang: String,
    notation: &'static str,
    behavior: &'static str,
}

struct WiktionaryTaskDemoInputs<'a> {
    pronunciation_word: &'a str,
    normalize_word: &'a str,
    orthography_guess_word: &'a str,
    adversarial_orthography_word: &'a str,
    phonology_guess_input: &'static str,
    combined_guess_word: &'a str,
    combined_guess_phonology: &'static str,
}

impl RaceStats {
    fn new() -> Self {
        Self {
            runs: 0,
            failures: 0,
            total: Duration::ZERO,
        }
    }

    fn record(&mut self, elapsed: Duration) {
        self.runs += 1;
        self.total += elapsed;
    }

    fn fail(&mut self) {
        self.failures += 1;
    }
}

impl Scorecard {
    fn new() -> Self {
        let rows = [
            "English sight words",
            "English nonce words",
            "Long English morphology",
            "German compounds",
            "Spanish",
            "Latin",
            "Greek",
            "Sanskrit",
            "Script discipline",
            "Language ID from phonology",
            "Language ID from orthography",
        ]
        .into_iter()
        .map(|behavior| ScorecardRow {
            behavior,
            passed: 0,
            total: 0,
            notes: Vec::new(),
        })
        .collect();
        Self { rows }
    }

    fn record(&mut self, probe: ScorecardProbe) {
        let Some(row) = self
            .rows
            .iter_mut()
            .find(|row| row.behavior == probe.behavior)
        else {
            return;
        };
        row.total += 1;
        if probe.passed {
            row.passed += 1;
        }
        if row.notes.len() < 3 {
            row.notes.push(probe.note);
        }
    }

    fn print(&self) {
        println!();
        println!("Wiktionary behavior scorecard");
        println!("  {:<32} {:<14} {:<8} Notes", "Behavior", "Status", "Score");
        for row in &self.rows {
            let status = row.status();
            let score = if row.total == 0 {
                "-".to_string()
            } else {
                format!("{}/{}", row.passed, row.total)
            };
            let notes = if row.notes.is_empty() {
                "-".to_string()
            } else {
                row.notes.join("; ")
            };
            println!(
                "  {:<32} {:<14} {:<8} {}",
                row.behavior,
                status,
                score,
                clip(&notes, 84)
            );
        }
    }
}

impl ScorecardRow {
    fn status(&self) -> &'static str {
        if self.total == 0 {
            return "Untested";
        }
        let ratio = self.passed as f64 / self.total as f64;
        if ratio >= 0.90 {
            "Excellent"
        } else if ratio >= 0.75 {
            "Very good"
        } else if ratio >= 0.60 {
            "Good"
        } else if ratio >= 0.40 {
            "Mixed"
        } else if self.passed > 0 {
            "Weak"
        } else {
            "Still haunted"
        }
    }
}

fn speech_demo(raw_args: Vec<String>) -> Result<(), String> {
    let mut skip_build = false;
    let mut forwarded_args = Vec::new();
    for arg in raw_args {
        if arg == "--skip-build" {
            skip_build = true;
        } else {
            forwarded_args.push(arg);
        }
    }

    if !skip_build {
        println!("speech-demo: building the optimized tongues binary once");
        run_release_build()?;
    }
    let tongues = release_tongues_bin_path();
    if !tongues.exists() {
        return Err(format!(
            "{} does not exist; run without --skip-build first",
            tongues.display()
        ));
    }

    let mut sentences = [
        "Morning light rested on the cedar trees while the kettle began to sing.",
        "A clear river curved through the valley beneath a patient summer sky.",
        "Fresh bread cooled by the window as music drifted in from the garden.",
        "The old observatory opened its dome to a field of quiet stars.",
        "Rain polished the streets, and every lamp made a small golden harbor.",
        "Beyond the orchard, a train carried warm letters toward the coast.",
        "She found a blue feather on the path and tucked it into her notebook.",
        "At dusk, the library windows glowed softly above the sleeping square.",
    ];
    sentences.shuffle(&mut rand::thread_rng());

    println!(
        "speech-demo: starting one resident process for all backends ({})",
        tongues.display()
    );
    let mut process = Command::new(&tongues);
    process.arg("speech-demo").args(forwarded_args);
    for sentence in sentences.into_iter().take(6) {
        process.args(["--sentence", sentence]);
    }
    let status = process
        .status()
        .map_err(|error| format!("starting {}: {error}", tongues.display()))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("resident speech demo failed with {status}"))
    }
}

fn continue_stream(raw_args: Vec<String>) -> Result<(), String> {
    let config = parse_continue_args(raw_args)?;
    if !config.skip_build {
        println!("continue: building tongues binary");
        run_build()?;
    }
    let tongues = tongues_bin_path();
    if !tongues.exists() {
        return Err(format!(
            "{} does not exist; run without --skip-build first",
            tongues.display()
        ));
    }
    fs::create_dir_all(&config.out_dir)
        .map_err(|error| format!("creating {}: {error}", config.out_dir.display()))?;

    println!(
        "continue: chunks={} forever={} backend={} out={}",
        config.chunks,
        config.forever,
        config.speak_backend,
        config.out_dir.display()
    );

    let mut index = 0usize;
    loop {
        if !config.forever && index >= config.chunks {
            break;
        }
        let buffer = generated_continue_sentence(index);
        let text = continue_head_chunk(&buffer).unwrap_or(buffer.as_str());
        let phones = run_phones(&tongues, &config, text)?;
        let wav = config.out_dir.join(format!("chunk-{:04}.wav", index + 1));
        run_speak(&tongues, &config, text, &wav)?;
        println!(
            "  {:04} head={} phones={} wav={}",
            index + 1,
            clip(text, 72),
            clip(&phones, 72),
            wav.display()
        );
        index += 1;
        if config.sleep_ms > 0 {
            std::thread::sleep(Duration::from_millis(config.sleep_ms));
        }
    }
    Ok(())
}

fn race(raw_args: Vec<String>) -> Result<(), String> {
    let config = parse_race_args(raw_args)?;
    let languages = read_wiktionary_languages(&config.wiktionary_config)?;
    let words = if config.words.is_empty() {
        default_race_words()
    } else {
        config.words.clone()
    };

    if !config.skip_build {
        println!("race: building tongues binary");
        run_build()?;
    }

    let tongues = tongues_bin_path();
    if !tongues.exists() {
        return Err(format!(
            "{} does not exist; run without --skip-build first",
            tongues.display()
        ));
    }

    println!(
        "race: {} forms, {} configured Wiktionary languages, compact task coverage",
        words.len(),
        languages.len()
    );
    println!(
        "race: g2p2g={}, wiktionary={}",
        config.g2p2g_model.display(),
        config.wiktionary_model.display()
    );

    let total_start = Instant::now();
    let mut stats = RaceStats::new();
    let mut scorecard = Scorecard::new();
    let wiktionary_cases = wiktionary_round_trip_cases(&words, &languages);
    println!(
        "race plan: g2p2g={} rt, wiktionary={} rt, wiktionary task demos=10 + raw",
        words.len(),
        wiktionary_cases.len()
    );

    println!();
    println!("G2P2G round trips (compact stress sample)");
    for word in &words {
        match round_trip_g2p2g(&tongues, &config, word) {
            Ok((forward, reverse)) => {
                stats.record(forward.elapsed + reverse.elapsed);
                println!(
                    "  ok {:>6} + {:>6}  {:<14} -> {:<18} -> {}",
                    fmt_ms(forward.elapsed),
                    fmt_ms(reverse.elapsed),
                    clip(word, 14),
                    clip(&forward.output, 18),
                    clip(&reverse.output, 18)
                );
            }
            Err(error) => {
                stats.fail();
                println!("  fail {:<14} {}", clip(word, 14), error);
            }
        }
    }

    println!();
    println!(
        "Wiktionary orthography/phonology round trips ({} curated cases)",
        wiktionary_cases.len()
    );
    for case in &wiktionary_cases {
        match round_trip_wiktionary(&tongues, &config, &case.word, &case.lang, case.notation) {
            Ok((forward, reverse)) => {
                stats.record(forward.elapsed + reverse.elapsed);
                record_round_trip_scorecard(&mut scorecard, case, &forward.output, &reverse.output);
                println!(
                    "  ok {:>6} + {:>6}  {:<3}/{:<8} {:<18} -> {:<20} -> {}",
                    fmt_ms(forward.elapsed),
                    fmt_ms(reverse.elapsed),
                    case.lang,
                    case.notation,
                    clip(&case.word, 18),
                    clip(&forward.output, 20),
                    clip(&reverse.output, 20)
                );
            }
            Err(error) => {
                stats.fail();
                println!(
                    "  fail {:<3}/{:<8} {:<18} {}",
                    case.lang,
                    case.notation,
                    clip(&case.word, 18),
                    error
                );
            }
        }
    }

    println!();
    println!("Wiktionary task demos");
    let demo_lang = languages
        .iter()
        .find(|lang| lang.as_str() == "eng")
        .unwrap_or(&languages[0]);
    let demo_inputs = wiktionary_task_demo_inputs(&words);
    match run_wiktionary_infer(
        &tongues,
        &config,
        "orthography-to-phones",
        demo_lang,
        "phones",
        Some("en-GB.RP"),
        false,
        demo_inputs.pronunciation_word,
    ) {
        Ok(pronunciation) => {
            stats.record(pronunciation.elapsed);
            println!(
                "  ok {:>6}  {:<38} {} -> {}",
                fmt_ms(pronunciation.elapsed),
                "orthography-to-phones --variety en-GB.RP",
                clip(demo_inputs.pronunciation_word, 14),
                clip(&pronunciation.output, 28)
            );

            let phonemes = match run_wiktionary_infer(
                &tongues,
                &config,
                "orthography-to-phonemes",
                demo_lang,
                "phonemes",
                None,
                false,
                demo_inputs.pronunciation_word,
            ) {
                Ok(result) => {
                    stats.record(result.elapsed);
                    println!(
                        "  ok {:>6}  {:<38} {} -> {}",
                        fmt_ms(result.elapsed),
                        "orthography-to-phonemes",
                        clip(demo_inputs.pronunciation_word, 28),
                        clip(&result.output, 28)
                    );
                    Some(result.output)
                }
                Err(error) => {
                    stats.fail();
                    println!("  fail {:<38} {}", "orthography-to-phonemes", error);
                    None
                }
            };
            let phonemic_input = phonemes.as_deref().unwrap_or(&pronunciation.output);

            for demo in [
                WiktionaryInferDemo {
                    label: "phonemes-to-orthography",
                    task: "phonemes-to-orthography",
                    lang: demo_lang,
                    notation: "phonemes",
                    variety: None,
                    raw: false,
                    input: phonemic_input.to_string(),
                },
                WiktionaryInferDemo {
                    label: "phones-to-orthography",
                    task: "phones-to-orthography",
                    lang: demo_lang,
                    notation: "phones",
                    variety: Some("en-GB.RP"),
                    raw: false,
                    input: pronunciation.output.clone(),
                },
                WiktionaryInferDemo {
                    label: "phonetic-realization",
                    task: "phonetic-realization",
                    lang: demo_lang,
                    notation: "phonemes",
                    variety: Some("en-GB.RP"),
                    raw: false,
                    input: phonemic_input.to_string(),
                },
                WiktionaryInferDemo {
                    label: "normalize",
                    task: "normalize",
                    lang: demo_lang,
                    notation: "phones",
                    variety: None,
                    raw: false,
                    input: format!("{}!", demo_inputs.normalize_word),
                },
                WiktionaryInferDemo {
                    label: "guess-lang-from-orthography",
                    task: "guess-lang-from-orthography",
                    lang: demo_lang,
                    notation: "phones",
                    variety: None,
                    raw: false,
                    input: demo_inputs.orthography_guess_word.to_string(),
                },
                WiktionaryInferDemo {
                    label: "guess-lang-from-orthography adversarial",
                    task: "guess-lang-from-orthography",
                    lang: demo_lang,
                    notation: "phones",
                    variety: None,
                    raw: false,
                    input: demo_inputs.adversarial_orthography_word.to_string(),
                },
                WiktionaryInferDemo {
                    label: "guess-lang-from-phonology",
                    task: "guess-lang-from-phonology",
                    lang: demo_lang,
                    notation: "phones",
                    variety: None,
                    raw: false,
                    input: demo_inputs.phonology_guess_input.to_string(),
                },
                WiktionaryInferDemo {
                    label: "guess-lang-from-orthography-and-phonology",
                    task: "guess-lang-from-orthography-and-phonology",
                    lang: demo_lang,
                    notation: "phones",
                    variety: None,
                    raw: false,
                    input: format!(
                        "{} => {}",
                        demo_inputs.combined_guess_word, demo_inputs.combined_guess_phonology
                    ),
                },
                WiktionaryInferDemo {
                    label: "--raw tagged source",
                    task: "orthography-to-phones",
                    lang: demo_lang,
                    notation: "phones",
                    variety: None,
                    raw: true,
                    input: format!(
                        "<task:orthography_to_phonology> <lang:{demo_lang}> <repr:phones> {}",
                        demo_inputs.pronunciation_word
                    ),
                },
            ] {
                match run_wiktionary_infer(
                    &tongues,
                    &config,
                    demo.task,
                    demo.lang,
                    demo.notation,
                    demo.variety,
                    demo.raw,
                    &demo.input,
                ) {
                    Ok(result) => {
                        stats.record(result.elapsed);
                        if let Some(probe) = scorecard_probe_for_demo(&demo, &result.output) {
                            scorecard.record(probe);
                        }
                        println!(
                            "  ok {:>6}  {:<38} {} -> {}",
                            fmt_ms(result.elapsed),
                            demo.label,
                            clip(&demo.input, 28),
                            clip(&result.output, 28)
                        );
                    }
                    Err(error) => {
                        stats.fail();
                        println!("  fail {:<38} {}", demo.label, error);
                    }
                }
            }
        }
        Err(error) => {
            stats.fail();
            println!(
                "  fail {:<38} {}",
                "orthography-to-phones --variety en-GB.RP", error
            );
        }
    }

    scorecard.print();

    println!();
    println!(
        "race: done in {} wall; {} successful inference demos, {} failures, {} summed inference time",
        fmt_ms(total_start.elapsed()),
        stats.runs,
        stats.failures,
        fmt_ms(stats.total)
    );

    Ok(())
}

fn wiktionary_round_trip_cases(
    words: &[String],
    languages: &[String],
) -> Vec<WiktionaryRoundTripCase> {
    let mut cases = Vec::new();
    let english = preferred_lang(languages, "eng");
    let spanish = preferred_lang(languages, "spa");
    let french = preferred_lang(languages, "fra");
    let german = preferred_lang(languages, "deu");
    let latin = preferred_lang(languages, "lat");
    let greek = preferred_lang(languages, "grc").or_else(|| preferred_lang(languages, "ell"));
    let sanskrit = preferred_lang(languages, "san");

    for (word, lang, notation, behavior) in [
        (
            "said",
            english.as_deref(),
            "phonemes",
            "English sight words",
        ),
        ("where", english.as_deref(), "phones", "English sight words"),
        (
            "unhelpfulness",
            english.as_deref(),
            "phonemes",
            "Long English morphology",
        ),
        (
            "internationalization",
            english.as_deref(),
            "phones",
            "Long English morphology",
        ),
        (
            "glimmerthorn",
            english.as_deref(),
            "phonemes",
            "English nonce words",
        ),
        (
            "brindlewise",
            english.as_deref(),
            "phones",
            "English nonce words",
        ),
        (
            "Tyrannosaurus",
            english.as_deref(),
            "phonemes",
            "Long English morphology",
        ),
        (
            "Pachycephalosaurus",
            english.as_deref(),
            "phones",
            "Long English morphology",
        ),
        (
            "Velociraptor",
            latin.as_deref().or(english.as_deref()),
            "phonemes",
            "Latin",
        ),
        (
            "Quetzalcoatlus",
            english.as_deref(),
            "phones",
            "Long English morphology",
        ),
        (
            "Parasaurolophus",
            latin.as_deref().or(english.as_deref()),
            "phonemes",
            "Latin",
        ),
        ("mañana", spanish.as_deref(), "phonemes", "Spanish"),
        ("jalapeño", spanish.as_deref(), "phones", "Spanish"),
        (
            "desafortunadamente",
            spanish.as_deref(),
            "phonemes",
            "Spanish",
        ),
        ("clarolumbre", spanish.as_deref(), "phones", "Spanish"),
        ("rendezvous", french.as_deref(), "phones", "French"),
        ("déshumanisation", french.as_deref(), "phonemes", "French"),
        ("lumivrage", french.as_deref(), "phones", "French"),
        (
            "brötchen",
            german.as_deref(),
            "phonemes",
            "German compounds",
        ),
        (
            "Wiedervereinigung",
            german.as_deref(),
            "phones",
            "German compounds",
        ),
        (
            "Sonnenklangerei",
            german.as_deref(),
            "phonemes",
            "German compounds",
        ),
        ("ventoribus", latin.as_deref(), "phonemes", "Latin"),
        ("praefulgeo", latin.as_deref(), "phones", "Latin"),
        ("ἄνθρωπος", greek.as_deref(), "phonemes", "Greek"),
        ("φιλοσοφία", greek.as_deref(), "phones", "Greek"),
        ("νεφελόφως", greek.as_deref(), "phonemes", "Greek"),
        ("कर्म", sanskrit.as_deref(), "phonemes", "Sanskrit"),
        ("धर्मक्षेत्र", sanskrit.as_deref(), "phones", "Sanskrit"),
        ("सुगमनिका", sanskrit.as_deref(), "phonemes", "Sanskrit"),
    ] {
        if let Some(lang) = lang {
            cases.push(WiktionaryRoundTripCase {
                word: pick_word(words, word).to_string(),
                lang: lang.to_string(),
                notation,
                behavior,
            });
        }
    }
    cases
}

fn record_round_trip_scorecard(
    scorecard: &mut Scorecard,
    case: &WiktionaryRoundTripCase,
    forward: &str,
    reverse: &str,
) {
    let pass = round_trip_score_pass(&case.word, reverse);
    scorecard.record(ScorecardProbe {
        behavior: case.behavior,
        passed: pass,
        note: format!(
            "{} -> {} -> {}",
            clip(&case.word, 18),
            clip(forward, 18),
            clip(reverse, 18)
        ),
    });

    let script_ok = same_primary_script(&case.word, reverse);
    scorecard.record(ScorecardProbe {
        behavior: "Script discipline",
        passed: script_ok,
        note: format!(
            "{}:{}->{}",
            case.lang,
            script_name(primary_script(&case.word)),
            script_name(primary_script(reverse))
        ),
    });
}

fn round_trip_score_pass(expected: &str, actual: &str) -> bool {
    let expected = canonical_orthography(expected);
    let actual = canonical_orthography(actual);
    if expected == actual {
        return true;
    }

    let expected_script = primary_script(&expected);
    if expected_script == Script::Unknown || expected_script != primary_script(&actual) {
        return false;
    }

    let threshold = match expected_script {
        Script::Latin => 0.80,
        Script::Greek => 0.70,
        Script::Devanagari => 0.85,
        Script::Mixed | Script::Unknown => return false,
    };
    normalized_similarity(&expected, &actual) >= threshold
}

fn normalized_similarity(left: &str, right: &str) -> f64 {
    let left = left.chars().collect::<Vec<_>>();
    let right = right.chars().collect::<Vec<_>>();
    let max_len = left.len().max(right.len());
    if max_len == 0 {
        return 1.0;
    }
    1.0 - (levenshtein_distance(&left, &right) as f64 / max_len as f64)
}

fn levenshtein_distance(left: &[char], right: &[char]) -> usize {
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    let mut current = vec![0; right.len() + 1];
    for (left_index, left_ch) in left.iter().enumerate() {
        current[0] = left_index + 1;
        for (right_index, right_ch) in right.iter().enumerate() {
            let deletion = previous[right_index + 1] + 1;
            let insertion = current[right_index] + 1;
            let substitution = previous[right_index] + usize::from(left_ch != right_ch);
            current[right_index + 1] = deletion.min(insertion).min(substitution);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

fn scorecard_probe_for_demo(
    demo: &WiktionaryInferDemo<'_>,
    output: &str,
) -> Option<ScorecardProbe> {
    match demo.label {
        "guess-lang-from-orthography" => Some(ScorecardProbe {
            behavior: "Language ID from orthography",
            passed: output.trim() == "deu",
            note: format!("{} -> {}", clip(&demo.input, 18), clip(output, 18)),
        }),
        "guess-lang-from-orthography adversarial" => Some(ScorecardProbe {
            behavior: "Language ID from orthography",
            passed: output.trim() == "eng" || output.trim() == "lat",
            note: format!("{} -> {}", clip(&demo.input, 18), clip(output, 18)),
        }),
        "guess-lang-from-phonology" => Some(ScorecardProbe {
            behavior: "Language ID from phonology",
            passed: output.trim() == "spa",
            note: format!("{} -> {}", clip(&demo.input, 18), clip(output, 18)),
        }),
        "guess-lang-from-orthography-and-phonology" => Some(ScorecardProbe {
            behavior: "Language ID from phonology",
            passed: output.trim() == "san",
            note: format!("{} -> {}", clip(&demo.input, 18), clip(output, 18)),
        }),
        _ => None,
    }
}

fn canonical_orthography(value: &str) -> String {
    value
        .chars()
        .filter(|ch| !ch.is_whitespace() && !is_ascii_punctuation(*ch))
        .flat_map(char::to_lowercase)
        .collect()
}

fn is_ascii_punctuation(ch: char) -> bool {
    ch.is_ascii_punctuation()
}

fn same_primary_script(left: &str, right: &str) -> bool {
    let left = primary_script(left);
    let right = primary_script(right);
    left != Script::Unknown && left == right
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Script {
    Latin,
    Greek,
    Devanagari,
    Mixed,
    Unknown,
}

fn primary_script(value: &str) -> Script {
    let mut latin = 0usize;
    let mut greek = 0usize;
    let mut devanagari = 0usize;
    for ch in value.chars() {
        if ch.is_alphabetic() {
            match ch as u32 {
                0x0041..=0x007A | 0x00C0..=0x024F | 0x1E00..=0x1EFF => latin += 1,
                0x0370..=0x03FF | 0x1F00..=0x1FFF => greek += 1,
                0x0900..=0x097F => devanagari += 1,
                _ => {}
            }
        }
    }
    let scripts = [
        (Script::Latin, latin),
        (Script::Greek, greek),
        (Script::Devanagari, devanagari),
    ];
    let non_zero = scripts.iter().filter(|(_, count)| *count > 0).count();
    if non_zero == 0 {
        return Script::Unknown;
    }
    if non_zero > 1 {
        return Script::Mixed;
    }
    scripts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(script, _)| script)
        .unwrap_or(Script::Unknown)
}

fn script_name(script: Script) -> &'static str {
    match script {
        Script::Latin => "Latin",
        Script::Greek => "Greek",
        Script::Devanagari => "Devanagari",
        Script::Mixed => "Mixed",
        Script::Unknown => "Unknown",
    }
}

fn wiktionary_task_demo_inputs(words: &[String]) -> WiktionaryTaskDemoInputs<'_> {
    WiktionaryTaskDemoInputs {
        pronunciation_word: pick_word(words, "through"),
        normalize_word: pick_word(words, "déshumanisation"),
        orthography_guess_word: pick_word(words, "brötchen"),
        adversarial_orthography_word: pick_word(words, "Archaeopteryx"),
        phonology_guess_input: "maˈɲana",
        combined_guess_word: pick_word(words, "धर्मक्षेत्र"),
        combined_guess_phonology: "dʱɐɾmɐkʂeːt̪ɾɐ",
    }
}

fn preferred_lang(languages: &[String], target: &str) -> Option<String> {
    languages
        .iter()
        .find(|lang| lang.as_str() == target)
        .cloned()
}

fn pick_word<'a>(words: &'a [String], fallback: &'a str) -> &'a str {
    words
        .iter()
        .find(|word| word.as_str() == fallback)
        .map(String::as_str)
        .unwrap_or(fallback)
}

fn parse_race_args(args: Vec<String>) -> Result<RaceConfig, String> {
    let mut config = RaceConfig {
        cpu: false,
        skip_build: false,
        g2p2g_model: PathBuf::from("models/g2p2g/openepd-v0"),
        wiktionary_model: PathBuf::from("models/wiktionary/enwiktionary-2026-06-01-v0-phones"),
        wiktionary_config: PathBuf::from("configs/wiktionary/default.toml"),
        words: Vec::new(),
    };

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--cpu" => config.cpu = true,
            "--skip-build" => config.skip_build = true,
            "--g2p2g-model" => {
                config.g2p2g_model = PathBuf::from(next_race_value(&mut iter, "--g2p2g-model")?);
            }
            "--wiktionary-model" => {
                config.wiktionary_model =
                    PathBuf::from(next_race_value(&mut iter, "--wiktionary-model")?);
            }
            "--wiktionary-config" => {
                config.wiktionary_config =
                    PathBuf::from(next_race_value(&mut iter, "--wiktionary-config")?);
            }
            _ if arg.starts_with("--") => {
                return Err(format!("unknown race option `{arg}`\n\n{}", race_usage()));
            }
            _ => config.words.push(arg),
        }
    }

    Ok(config)
}

fn parse_continue_args(args: Vec<String>) -> Result<ContinueConfig, String> {
    let mut config = ContinueConfig {
        cpu: false,
        skip_build: false,
        forever: false,
        chunks: 8,
        sleep_ms: 250,
        speak_backend: "mock".to_string(),
        out_dir: PathBuf::from("runs/head2phones/continue"),
    };
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--cpu" => config.cpu = true,
            "--skip-build" => config.skip_build = true,
            "--forever" => config.forever = true,
            "--chunks" => {
                let value = next_continue_value(&mut iter, "--chunks")?;
                config.chunks = value.parse().map_err(|_| {
                    format!(
                        "--chunks expects a positive integer\n\n{}",
                        continue_usage()
                    )
                })?;
            }
            "--sleep-ms" => {
                let value = next_continue_value(&mut iter, "--sleep-ms")?;
                config.sleep_ms = value.parse().map_err(|_| {
                    format!("--sleep-ms expects an integer\n\n{}", continue_usage())
                })?;
            }
            "--speak-backend" => {
                config.speak_backend = next_continue_value(&mut iter, "--speak-backend")?;
            }
            "--out-dir" => {
                config.out_dir = PathBuf::from(next_continue_value(&mut iter, "--out-dir")?);
            }
            _ if arg.starts_with('-') => {
                return Err(format!(
                    "unknown continue option `{arg}`\n\n{}",
                    continue_usage()
                ));
            }
            _ => {
                return Err(format!(
                    "unexpected continue argument `{arg}`\n\n{}",
                    continue_usage()
                ))
            }
        }
    }
    if config.chunks == 0 && !config.forever {
        return Err(format!(
            "--chunks must be greater than zero unless --forever is set\n\n{}",
            continue_usage()
        ));
    }
    Ok(config)
}

fn next_race_value(
    iter: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("{option} requires a value\n\n{}", race_usage()))
}

fn next_continue_value(
    iter: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("{option} requires a value\n\n{}", continue_usage()))
}

fn run_build() -> Result<(), String> {
    let status = Command::new("cargo")
        .args(["build", "--quiet", "--bin", "tongues"])
        .status()
        .map_err(|error| format!("starting cargo build: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("cargo build failed with {status}"))
    }
}

fn run_release_build() -> Result<(), String> {
    let status = Command::new("cargo")
        .args([
            "build",
            "--quiet",
            "--release",
            "--package",
            "tongues-cli",
            "--bin",
            "tongues",
        ])
        .status()
        .map_err(|error| format!("starting release cargo build: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("release cargo build failed with {status}"))
    }
}

fn round_trip_g2p2g(
    tongues: &Path,
    config: &RaceConfig,
    word: &str,
) -> Result<(RaceResult, RaceResult), String> {
    let mut args = base_tongues_args(config);
    args.extend([
        "g2p2g".to_string(),
        "infer".to_string(),
        "--task".to_string(),
        "g2p".to_string(),
        "--model".to_string(),
        config.g2p2g_model.display().to_string(),
        "--".to_string(),
        word.to_string(),
    ]);
    let forward = run_infer(tongues, &args)?;

    let mut reverse_args = base_tongues_args(config);
    reverse_args.extend([
        "g2p2g".to_string(),
        "infer".to_string(),
        "--task".to_string(),
        "p2g".to_string(),
        "--model".to_string(),
        config.g2p2g_model.display().to_string(),
        "--".to_string(),
        forward.output.clone(),
    ]);
    let reverse = run_infer(tongues, &reverse_args)?;
    Ok((forward, reverse))
}

fn round_trip_wiktionary(
    tongues: &Path,
    config: &RaceConfig,
    word: &str,
    lang: &str,
    notation: &str,
) -> Result<(RaceResult, RaceResult), String> {
    let mut args = base_tongues_args(config);
    args.extend([
        "wiktionary".to_string(),
        "infer".to_string(),
        "--model".to_string(),
        config.wiktionary_model.display().to_string(),
        "--task".to_string(),
        wiktionary_orthography_to_phonology_task(notation).to_string(),
        "--lang".to_string(),
        lang.to_string(),
        "--notation".to_string(),
        notation.to_string(),
        "--".to_string(),
        word.to_string(),
    ]);
    let forward = run_infer(tongues, &args)?;

    let mut reverse_args = base_tongues_args(config);
    reverse_args.extend([
        "wiktionary".to_string(),
        "infer".to_string(),
        "--model".to_string(),
        config.wiktionary_model.display().to_string(),
        "--task".to_string(),
        wiktionary_phonology_to_orthography_task(notation).to_string(),
        "--lang".to_string(),
        lang.to_string(),
        "--notation".to_string(),
        notation.to_string(),
        "--".to_string(),
        forward.output.clone(),
    ]);
    let reverse = run_infer(tongues, &reverse_args)?;
    Ok((forward, reverse))
}

fn wiktionary_orthography_to_phonology_task(notation: &str) -> &'static str {
    match notation {
        "phonemes" => "orthography-to-phonemes",
        "phones" => "orthography-to-phones",
        _ => "orthography-to-phonology",
    }
}

fn wiktionary_phonology_to_orthography_task(notation: &str) -> &'static str {
    match notation {
        "phonemes" => "phonemes-to-orthography",
        "phones" => "phones-to-orthography",
        _ => "phonology-to-orthography",
    }
}

#[allow(clippy::too_many_arguments)]
fn run_wiktionary_infer(
    tongues: &Path,
    config: &RaceConfig,
    task: &str,
    lang: &str,
    notation: &str,
    variety: Option<&str>,
    raw: bool,
    input: &str,
) -> Result<RaceResult, String> {
    let mut args = base_tongues_args(config);
    args.extend([
        "wiktionary".to_string(),
        "infer".to_string(),
        "--model".to_string(),
        config.wiktionary_model.display().to_string(),
        "--task".to_string(),
        task.to_string(),
        "--lang".to_string(),
        lang.to_string(),
        "--notation".to_string(),
        notation.to_string(),
    ]);
    if let Some(variety) = variety {
        args.extend(["--variety".to_string(), variety.to_string()]);
    }
    if raw {
        args.push("--raw".to_string());
    }
    args.extend(["--".to_string(), input.to_string()]);
    run_infer(tongues, &args)
}

fn base_tongues_args(config: &RaceConfig) -> Vec<String> {
    if config.cpu {
        vec!["--cpu".to_string()]
    } else {
        Vec::new()
    }
}

fn run_infer(tongues: &Path, args: &[String]) -> Result<RaceResult, String> {
    let start = Instant::now();
    let output = Command::new(tongues)
        .args(args)
        .output()
        .map_err(|error| format!("starting {}: {error}", tongues.display()))?;
    let elapsed = start.elapsed();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "exited {}: {}",
            output.status,
            clip(stderr.trim(), 80)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let prediction = extract_prediction(&stdout)
        .ok_or_else(|| format!("prediction output not found: {}", clip(stdout.trim(), 80)))?;
    Ok(RaceResult {
        output: prediction,
        elapsed,
    })
}

fn run_phones(tongues: &Path, config: &ContinueConfig, text: &str) -> Result<String, String> {
    let mut args = Vec::new();
    if config.cpu {
        args.push("--cpu".to_string());
    }
    args.extend(["phones".to_string(), text.to_string()]);
    let output = Command::new(tongues)
        .args(&args)
        .output()
        .map_err(|error| format!("starting {} phones: {error}", tongues.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "phones exited {}: {}",
            output.status,
            clip(stderr.trim(), 100)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn run_speak(
    tongues: &Path,
    config: &ContinueConfig,
    text: &str,
    wav: &Path,
) -> Result<(), String> {
    let mut args = Vec::new();
    if config.cpu {
        args.push("--cpu".to_string());
    }
    args.extend([
        "speak".to_string(),
        "--backend".to_string(),
        config.speak_backend.clone(),
        "--output".to_string(),
        wav.display().to_string(),
        text.to_string(),
    ]);
    let output = Command::new(tongues)
        .args(&args)
        .output()
        .map_err(|error| format!("starting {} speak: {error}", tongues.display()))?;
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!(
            "speak exited {}: {}",
            output.status,
            clip(stderr.trim(), 100)
        ))
    }
}

fn extract_prediction(stdout: &str) -> Option<String> {
    let mut lines = stdout.lines();
    while let Some(line) = lines.next() {
        if line.trim() == "Prediction output:" {
            return lines.next().map(|value| value.trim().to_string());
        }
    }
    let mut non_empty = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let prediction = non_empty.next()?;
    if non_empty.next().is_none() {
        Some(prediction.to_string())
    } else {
        None
    }
}

fn generated_continue_sentence(index: usize) -> String {
    const SENTENCES: &[&str] = &[
        "Dr. Smith went home. Then he checked the porch light.",
        "This is the next sentence; and then the next thought arrives.",
        "Wait... really? I thought the file was already open.",
        "\"No.\" she said. The room became quiet again.",
        "First, open the small panel. Then press the green switch.",
        "The package is ready, but the driver is late.",
        "- Bring the blue folder. Leave the red folder on the desk.",
        "In short: the answer changed. The stream keeps moving.",
    ];
    SENTENCES[index % SENTENCES.len()].to_string()
}

fn continue_head_chunk(buffer: &str) -> Option<&str> {
    let mut search_start = 0usize;
    while let Some((relative, ch)) = buffer[search_start..]
        .char_indices()
        .find(|(_, ch)| matches!(ch, '.' | '!' | '?' | ';' | ':'))
    {
        let index = search_start + relative;
        let after = index + ch.len_utf8();
        if ch == '.' && continue_dot_is_abbreviation(buffer, index) {
            search_start = after;
            continue;
        }
        return Some(buffer[..continue_closing_punctuation_end(buffer, after)].trim());
    }
    None
}

fn continue_closing_punctuation_end(buffer: &str, mut index: usize) -> usize {
    while let Some(ch) = buffer[index..].chars().next() {
        if matches!(ch, '"' | '\'' | ')' | ']' | '}') {
            index += ch.len_utf8();
        } else {
            break;
        }
    }
    index
}

fn continue_dot_is_abbreviation(buffer: &str, dot_index: usize) -> bool {
    let after_dot = dot_index + 1;
    if buffer[after_dot..].chars().next() == Some('.') {
        return false;
    }
    let prefix = buffer[..dot_index].trim_end();
    let token = prefix
        .split_whitespace()
        .last()
        .unwrap_or("")
        .trim_matches(|ch: char| matches!(ch, '"' | '\'' | '(' | '[' | '{' | '*' | '_' | '-'));
    matches!(
        token.to_ascii_lowercase().as_str(),
        "mr" | "mrs"
            | "ms"
            | "dr"
            | "prof"
            | "sr"
            | "jr"
            | "st"
            | "mt"
            | "vs"
            | "etc"
            | "e.g"
            | "i.e"
    )
}

#[cfg(test)]
mod tests {
    use super::{
        audit_family_maturity, crate_lib_rs, extract_prediction, primary_script,
        round_trip_score_pass, same_primary_script, wiktionary_round_trip_cases,
        wiktionary_task_demo_inputs, Script,
    };

    #[test]
    fn extracts_prediction_from_verbose_output() {
        let stdout = "Source:\n  have\n\nPrediction output:\n  hæv\nTotal time elapsed: 1ms\n";
        assert_eq!(extract_prediction(stdout), Some("hæv".to_string()));
    }

    #[test]
    fn extracts_prediction_from_quiet_output() {
        assert_eq!(extract_prediction("hæv\n"), Some("hæv".to_string()));
    }

    #[test]
    fn rejects_unlabeled_multi_line_output() {
        assert_eq!(extract_prediction("one\ntwo\n"), None);
    }

    #[test]
    fn wiktionary_race_cases_use_diverse_language_probes() {
        let words = super::default_race_words();
        let languages = ["eng", "spa", "fra", "deu", "lat", "grc", "san"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let cases = wiktionary_round_trip_cases(&words, &languages);

        assert!(cases
            .iter()
            .any(|case| case.lang == "san" && case.word == "धर्मक्षेत्र"));
        assert!(cases
            .iter()
            .any(|case| case.lang == "deu" && case.word == "brötchen"));
        assert!(cases.iter().all(|case| case.word != "Archaeopteryx"));
    }

    #[test]
    fn wiktionary_task_demos_do_not_reuse_international_words_for_language_guessing() {
        let words = super::default_race_words();
        let demos = wiktionary_task_demo_inputs(&words);

        assert_eq!(demos.pronunciation_word, "through");
        assert_eq!(demos.orthography_guess_word, "brötchen");
        assert_eq!(demos.adversarial_orthography_word, "Archaeopteryx");
        assert_eq!(demos.phonology_guess_input, "maˈɲana");
        assert_eq!(demos.combined_guess_word, "धर्मक्षेत्र");
    }

    #[test]
    fn scorecard_script_detection_flags_mixed_script_drift() {
        assert_eq!(primary_script("ἄνθρωπος"), Script::Greek);
        assert_eq!(primary_script("कर्म"), Script::Devanagari);
        assert_eq!(primary_script("URक्GACATSA"), Script::Mixed);
        assert!(same_primary_script("कर्म", "क्रम"));
        assert!(!same_primary_script("धर्मक्षेत्र", "URक्GACATSA"));
    }

    #[test]
    fn scorecard_round_trip_scoring_allows_close_same_script_misses() {
        assert!(round_trip_score_pass("ἄνθρωπος", "άνθροπος"));
        assert!(round_trip_score_pass("brötchen", "Bröttchen"));
        assert!(!round_trip_score_pass("कर्म", "क्रम"));
        assert!(!round_trip_score_pass("धर्मक्षेत्र", "URक्GACATSA"));
    }

    #[test]
    fn established_family_inventory_has_no_stale_scaffold_labels() {
        audit_family_maturity().unwrap();
    }

    #[test]
    fn new_family_template_is_explicitly_non_runnable() {
        let source = crate_lib_rs("allophone-realizer");

        assert!(source.contains("unimplemented-family-template"));
        assert!(source.contains(".as_family_template()"));
        assert!(source.contains("write_family_template"));
        assert!(!source.contains("model.bin"));
        assert!(!source.contains("train_state.json"));
    }
}

fn read_wiktionary_languages(path: &Path) -> Result<Vec<String>, String> {
    let raw =
        fs::read_to_string(path).map_err(|error| format!("reading {}: {error}", path.display()))?;
    for line in raw.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("languages") {
            let Some((_, value)) = rest.split_once('=') else {
                continue;
            };
            return parse_toml_string_array(value)
                .ok_or_else(|| format!("could not parse languages in {}", path.display()));
        }
    }
    Ok(vec!["eng".to_string()])
}

fn parse_toml_string_array(value: &str) -> Option<Vec<String>> {
    let start = value.find('[')?;
    let end = value.rfind(']')?;
    let inner = &value[start + 1..end];
    let mut out = Vec::new();
    for item in inner.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        out.push(item.trim_matches('"').to_string());
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn default_race_words() -> Vec<String> {
    [
        // Sight words and compact irregulars.
        "the",
        "and",
        "said",
        "one",
        "two",
        "have",
        "come",
        "where",
        "laugh",
        // Regular multi-morphemic English stress tests.
        "children",
        "through",
        "queue",
        "unhelpfulness",
        "rediscovering",
        "reclassification",
        "microbiological",
        "internationalization",
        "hyperconnected",
        // English nonce words that should still look pronounceable.
        "glimmerthorn",
        "brindlewise",
        "sprockleton",
        "mindlecrate",
        // Taxonomic and dinosaur-heavy forms.
        "Tyrannosaurus",
        "Archaeopteryx",
        "Velociraptor",
        "Quetzalcoatlus",
        "Parasaurolophus",
        "Pachycephalosaurus",
        "Micropachycephalosaurus",
        "Coelophysis",
        "Yi",
        // Romance/Germanic real and nonce forms.
        "rendezvous",
        "mañana",
        "jalapeño",
        "desafortunadamente",
        "clarolumbre",
        "déshumanisation",
        "lumivrage",
        "brötchen",
        "Wiedervereinigung",
        "Sonnenklangerei",
        "Kraftwerk",
        "Pteranodon",
        "Łódź",
        "Dvořák",
        "São Paulo",
        // Classical and Indic script probes, including plausible nonce forms.
        "ventoribus",
        "praefulgeo",
        "ἄνθρωπος",
        "φιλοσοφία",
        "νεφελόφως",
        "कर्म",
        "धर्मक्षेत्र",
        "सुगमनिका",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn tongues_bin_path() -> PathBuf {
    PathBuf::from("target")
        .join("debug")
        .join(format!("tongues{}", env::consts::EXE_SUFFIX))
}

fn release_tongues_bin_path() -> PathBuf {
    let target_dir = env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    target_dir
        .join("release")
        .join(format!("tongues{}", env::consts::EXE_SUFFIX))
}

fn fmt_ms(duration: Duration) -> String {
    format!("{}ms", duration.as_millis())
}

fn clip(value: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (index, ch) in value.chars().enumerate() {
        if index >= max_chars {
            out.push_str("...");
            return out;
        }
        out.push(ch);
    }
    out
}

fn new_family_usage() -> &'static str {
    "Usage: cargo xtask new-family <family-slug>\n\nThe family slug must be lowercase kebab-case, for example:\n  sentence-boundary\n  allophone-realizer\n"
}

fn new_family(family: &str) -> Result<(), String> {
    validate_family_slug(family)?;

    let crate_name = format!("tongues-{family}");
    let crate_dir = PathBuf::from("crates").join(&crate_name);
    let config_dir = PathBuf::from("configs").join(family);
    let dataset_dir = PathBuf::from("datasets").join(family);
    let run_dir = PathBuf::from("runs").join(family);
    let model_dir = PathBuf::from("models").join(family);

    ensure_missing(&crate_dir)?;
    ensure_missing(&config_dir)?;

    fs::create_dir_all(crate_dir.join("src"))
        .map_err(|error| format!("creating {}: {error}", crate_dir.join("src").display()))?;
    fs::create_dir_all(&config_dir)
        .map_err(|error| format!("creating {}: {error}", config_dir.display()))?;
    create_placeholder_dir(&dataset_dir)?;
    create_placeholder_dir(&run_dir)?;
    create_placeholder_dir(&model_dir)?;

    write_file(&crate_dir.join("Cargo.toml"), &crate_manifest(&crate_name))?;
    write_file(&crate_dir.join("src/lib.rs"), &crate_lib_rs(family))?;
    write_file(&config_dir.join("default.toml"), "dataset_id = \"v0\"\n")?;
    add_workspace_member(&crate_dir)?;

    println!("Created {family} model family scaffold:");
    println!("  {}", crate_dir.display());
    println!("  {}", config_dir.join("default.toml").display());
    println!("  {}", dataset_dir.join(".gitkeep").display());
    println!("  {}", run_dir.join(".gitkeep").display());
    println!("  {}", model_dir.join(".gitkeep").display());
    println!();
    println!("Next steps:");
    println!("  cargo test -p {crate_name}");
    println!("  wire {family} into crates/tongues-cli when its CLI semantics are clear");

    Ok(())
}

fn validate_family_slug(family: &str) -> Result<(), String> {
    if family.is_empty() {
        return Err(format!("missing family slug\n\n{}", new_family_usage()));
    }
    let bytes = family.as_bytes();
    let starts_ok = bytes[0].is_ascii_lowercase() || bytes[0].is_ascii_digit();
    let ends_ok =
        bytes[bytes.len() - 1].is_ascii_lowercase() || bytes[bytes.len() - 1].is_ascii_digit();
    let chars_ok = bytes
        .iter()
        .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'-');
    if starts_ok && ends_ok && chars_ok {
        Ok(())
    } else {
        Err(format!(
            "family slug must be lowercase kebab-case: {family}\n\n{}",
            new_family_usage()
        ))
    }
}

fn ensure_missing(path: &Path) -> Result<(), String> {
    if path.exists() {
        Err(format!("{} already exists", path.display()))
    } else {
        Ok(())
    }
}

fn create_placeholder_dir(path: &Path) -> Result<(), String> {
    fs::create_dir_all(path).map_err(|error| format!("creating {}: {error}", path.display()))?;
    write_file(&path.join(".gitkeep"), "")
}

fn write_file(path: &Path, contents: &str) -> Result<(), String> {
    fs::write(path, contents).map_err(|error| format!("writing {}: {error}", path.display()))
}

fn add_workspace_member(crate_dir: &Path) -> Result<(), String> {
    let cargo_toml = Path::new("Cargo.toml");
    let text = fs::read_to_string(cargo_toml)
        .map_err(|error| format!("reading {}: {error}", cargo_toml.display()))?;
    let member = crate_dir.to_str().ok_or_else(|| {
        format!(
            "workspace member path is not UTF-8: {}",
            crate_dir.display()
        )
    })?;
    let entry = format!("    \"{member}\",\n");
    if text.contains(&entry) {
        return Ok(());
    }

    let anchor = "    \"crates/tongues-cli\",\n";
    let updated = text.replacen(anchor, &(entry + anchor), 1);
    if updated == text {
        return Err(format!(
            "workspace member anchor not found in {}",
            cargo_toml.display()
        ));
    }
    write_file(cargo_toml, &updated)
}

fn crate_manifest(crate_name: &str) -> String {
    format!(
        r#"[package]
name = "{crate_name}"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = {{ workspace = true }}
serde = {{ workspace = true }}
serde_json = {{ workspace = true }}
tongues-neural = {{ path = "../tongues-neural" }}
"#
    )
}

fn crate_lib_rs(family: &str) -> String {
    format!(
        r#"//! {family} model-family scaffold.

use std::fs;
use std::path::Path;

use anyhow::{{Context, Result}};
use serde::{{Deserialize, Serialize}};
use tongues_neural::{{write_manifest, ModelArtifactManifest}};

pub const FAMILY: &str = "{family}";
pub const ARCHITECTURE: &str = "unimplemented-family-template";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FamilyConfig {{
    pub dataset_id: String,
}}

impl Default for FamilyConfig {{
    fn default() -> Self {{
        Self {{
            dataset_id: "v0".to_string(),
        }}
    }}
}}

pub fn prepare_dataset(out: &Path, config: &FamilyConfig) -> Result<()> {{
    fs::create_dir_all(out).with_context(|| format!("creating {{}}", out.display()))?;
    fs::write(out.join("dataset_config.json"), serde_json::to_string_pretty(config)?)?;
    fs::write(
        out.join("README.md"),
        format!(
            "{{}} dataset scaffold. Add train/valid/test data here.\n",
            FAMILY
        ),
    )?;
    Ok(())
}}

/// Write metadata for a family that has not implemented a runnable model yet.
///
/// This deliberately does not create checkpoint or training-state files.
pub fn write_family_template(out: &Path, config: &FamilyConfig) -> Result<()> {{
    fs::create_dir_all(out).with_context(|| format!("creating {{}}", out.display()))?;
    fs::write(
        out.join("family_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    write_manifest(
        out,
        &ModelArtifactManifest::new(FAMILY, ARCHITECTURE, &config.dataset_id)
            .as_family_template(),
    )
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn default_config_names_v0_dataset() {{
        assert_eq!(FamilyConfig::default().dataset_id, "v0");
    }}
}}
"#
    )
}
