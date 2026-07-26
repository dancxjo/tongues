use serde::Deserialize;
use tongues_audio::{
    spectrogram, MelConfig, MelNormalization, MelScale, PadMode, SpectralDomain, SpectralScale,
    SpectrogramConfig, SpectrogramNormalization, SpectrogramOutput, StftConfig, Window,
};

#[derive(Debug, Deserialize)]
struct GoldenFixture {
    provenance: Provenance,
    absolute_tolerance: f32,
    samples: Vec<f32>,
    frames: usize,
    bins: usize,
    values_frames_by_bins: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct Provenance {
    project: String,
    tag: String,
    implementation: String,
    numpy: String,
    scipy: String,
    librosa: String,
}

#[test]
fn matches_coqui_v0_22_mel_tensor() {
    let fixture: GoldenFixture =
        serde_json::from_str(include_str!("fixtures/coqui-v0.22.0-mel.json"))
            .expect("valid golden fixture");
    assert_eq!(fixture.provenance.project, "coqui-ai/TTS");
    assert_eq!(fixture.provenance.tag, "v0.22.0");
    assert!(!fixture.provenance.implementation.is_empty());
    assert_eq!(fixture.provenance.numpy, "1.26.4");
    assert_eq!(fixture.provenance.scipy, "1.11.4");
    assert_eq!(fixture.provenance.librosa, "0.10.1");

    let config = SpectrogramConfig {
        sample_rate_hz: 8_000,
        stft: StftConfig {
            fft_size: 32,
            window_size: 32,
            hop_size: 8,
            center: true,
            pad_mode: PadMode::Reflect,
            window: Window::Hann,
        },
        output: SpectrogramOutput::Mel(MelConfig {
            bins: 8,
            min_frequency_hz: 0.0,
            max_frequency_hz: Some(4_000.0),
            scale: MelScale::Slaney,
            normalization: MelNormalization::Slaney,
        }),
        domain: SpectralDomain::Amplitude,
        scale: SpectralScale::Log10 {
            gain: 20.0,
            floor: 1.0e-8,
        },
        normalization: SpectrogramNormalization::Range {
            min_db: -100.0,
            reference_db: 20.0,
            max_norm: 4.0,
            symmetric: true,
            clipped: true,
        },
        preemphasis: Some(0.97),
    };
    let actual = spectrogram(&fixture.samples, &config).expect("native features");
    assert_eq!(actual.frames, fixture.frames);
    assert_eq!(actual.config.output_bins(), fixture.bins);
    assert_eq!(actual.values.len(), fixture.values_frames_by_bins.len());
    for (index, (&actual, &expected)) in actual
        .values
        .iter()
        .zip(&fixture.values_frames_by_bins)
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= fixture.absolute_tolerance,
            "Coqui parity mismatch at flat index {index}: actual={actual}, expected={expected}, tolerance={}",
            fixture.absolute_tolerance
        );
    }
}
