use std::path::Path;

use anyhow::{ensure, Context, Result};

use crate::{
    AudioChunk, AudioSink, SpeechModelCapabilities, SpeechModelFamily, SpeechSynthesisEngine,
    SpeechSynthesisRequest,
};

use super::{
    MbrolaDatabase, MbrolaDatabaseError, MbrolaPitchTarget, MbrolaProjector, PhoneTimedPlan,
};

const CROSSFADE_SAMPLES: usize = 32;
const MIN_PSOLA_GRAINS: usize = 2;

#[derive(Debug, Clone)]
pub struct NativeMbrolaRenderer {
    database: MbrolaDatabase,
    projector: MbrolaProjector,
}

impl NativeMbrolaRenderer {
    pub fn load(database_path: impl AsRef<Path>, mut projector: MbrolaProjector) -> Result<Self> {
        let database = MbrolaDatabase::load(database_path.as_ref())?;
        projector.inventory = database.phonemes().map(str::to_string).collect();
        projector.inventory.insert("_".into());
        Ok(Self {
            database,
            projector,
        })
    }

    pub fn database_path(&self) -> &Path {
        &self.database.path
    }

    pub fn projector(&self) -> &MbrolaProjector {
        &self.projector
    }

    pub fn render_plan(&self, plan: &PhoneTimedPlan) -> Result<Vec<f32>> {
        plan.validate()?;
        ensure!(
            !plan.phones.is_empty(),
            "cannot render an empty MBROLA plan"
        );
        let mut output = Vec::new();
        for (index, phone) in plan.phones.iter().enumerate() {
            if phone.symbol == "_" {
                output.extend(std::iter::repeat_n(
                    0.0,
                    duration_samples(phone.duration_ms, self.database.sample_rate_hz).max(1),
                ));
                continue;
            }
            let previous = index
                .checked_sub(1)
                .and_then(|index| plan.phones.get(index))
                .map_or("_", |phone| phone.symbol.as_str());
            let next = plan
                .phones
                .get(index + 1)
                .map_or("_", |phone| phone.symbol.as_str());
            let previous_unit = self.unit(previous, &phone.symbol)?;
            let next_unit = self.unit(&phone.symbol, next)?;
            let previous_split = previous_unit
                .diphone
                .halfseg_samples
                .min(previous_unit.samples.len());
            let next_split = next_unit
                .diphone
                .halfseg_samples
                .min(next_unit.samples.len());
            let right_half = &previous_unit.samples[previous_split..];
            let left_half = &next_unit.samples[..next_split];
            let mut centers = previous_unit
                .centers
                .iter()
                .copied()
                .filter(|center| *center >= previous_split)
                .map(|center| center - previous_split)
                .collect::<Vec<_>>();
            centers.extend(
                next_unit
                    .centers
                    .iter()
                    .copied()
                    .filter(|center| *center < next_split)
                    .map(|center| center + right_half.len()),
            );
            centers.sort_unstable();
            centers.dedup();
            let unit = assemble_unit(right_half, left_half, CROSSFADE_SAMPLES);
            ensure!(
                !unit.is_empty(),
                "MBROLA diphone material for phone `{}` is empty",
                phone.symbol
            );
            let rendered = psola_synthesize(
                &unit,
                &centers,
                phone.duration_ms,
                &phone.pitch_targets,
                self.database.sample_rate_hz,
                self.database.mbr_period,
            );
            append_smoothed(&mut output, rendered, CROSSFADE_SAMPLES);
        }
        ensure!(
            output.iter().all(|sample| sample.is_finite()),
            "native MBROLA renderer produced non-finite audio"
        );
        Ok(output)
    }

    pub fn render_pho(&self, source: &str) -> Result<Vec<f32>> {
        self.render_plan(&super::parse_pho(source)?)
    }

    fn unit(&self, left: &str, right: &str) -> Result<LoadedUnit<'_>> {
        let diphone = self.database.diphone(left, right).ok_or_else(|| {
            MbrolaDatabaseError::MissingDiphone {
                left: left.to_string(),
                right: right.to_string(),
                database: self.database.path.clone(),
            }
        })?;
        Ok(LoadedUnit {
            samples: self.database.samples_for_diphone(diphone)?,
            centers: self.database.frame_centers(diphone),
            diphone,
        })
    }
}

struct LoadedUnit<'a> {
    samples: Vec<f32>,
    centers: Vec<usize>,
    diphone: &'a super::MbrolaDiphone,
}

impl SpeechSynthesisEngine for NativeMbrolaRenderer {
    fn capabilities(&self) -> SpeechModelCapabilities {
        SpeechModelCapabilities {
            family: SpeechModelFamily::EndToEndSpeech,
            supports_named_speakers: false,
            supports_languages: false,
            supports_reference_audio: false,
            supports_voice_conversion: false,
            integrated_vocoder: true,
        }
    }

    fn sample_rate_hz(&self) -> u32 {
        self.database.sample_rate_hz
    }

    fn synthesize_plan_streaming(
        &mut self,
        request: &SpeechSynthesisRequest,
        sink: &mut dyn AudioSink,
    ) -> Result<()> {
        let (mut plan, _report) = self
            .projector
            .project(&request.plan)
            .context("failed to lower UtterancePlan to MBROLA phone timing")?;
        if let Some(scale) = request.options.length_scale {
            ensure!(
                scale.is_finite() && scale > 0.0,
                "MBROLA length scale must be finite and positive"
            );
            for phone in &mut plan.phones {
                phone.duration_ms = ((phone.duration_ms as f32 * scale).round() as u32).max(1);
            }
        }
        let pitch_scale = request.options.pitch_scale.unwrap_or(1.0);
        let pitch_shift = request.options.pitch_shift.unwrap_or(0.0);
        ensure!(
            pitch_scale.is_finite() && pitch_scale > 0.0 && pitch_shift.is_finite(),
            "MBROLA pitch scale/shift must be finite and pitch scale positive"
        );
        if pitch_scale != 1.0 || pitch_shift != 0.0 {
            for target in plan
                .phones
                .iter_mut()
                .flat_map(|phone| &mut phone.pitch_targets)
            {
                target.hz = target.hz * pitch_scale + pitch_shift;
                ensure!(
                    target.hz.is_finite() && target.hz > 0.0,
                    "MBROLA pitch control produced invalid F0"
                );
            }
        }
        let pcm = self.render_plan(&plan)?;
        sink.emit(AudioChunk {
            chunk_index: 0,
            is_final: true,
            pause_after_ms: 0,
            sample_rate_hz: self.sample_rate_hz(),
            pcm_mono_f32: pcm,
        })
    }
}

fn duration_samples(duration_ms: u32, sample_rate_hz: u32) -> usize {
    (u64::from(duration_ms) * u64::from(sample_rate_hz) / 1000) as usize
}

fn remove_dc(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    samples.iter().map(|sample| sample - mean).collect()
}

fn assemble_unit(right: &[f32], left: &[f32], radius: usize) -> Vec<f32> {
    let mut right = remove_dc(right);
    let left = remove_dc(left);
    let join = right.len();
    right.extend(left);
    smooth_join(&mut right, join, radius);
    right
}

fn append_smoothed(samples: &mut Vec<f32>, mut next: Vec<f32>, radius: usize) {
    if samples.is_empty() {
        samples.append(&mut next);
        return;
    }
    let join = samples.len();
    samples.append(&mut next);
    smooth_join(samples, join, radius);
}

fn smooth_join(samples: &mut [f32], join: usize, requested_radius: usize) {
    if join == 0 || join >= samples.len() {
        return;
    }
    let radius = requested_radius.min(join).min(samples.len() - join);
    let jump = samples[join] - samples[join - 1];
    if radius == 0 || !jump.is_finite() {
        return;
    }
    let half_jump = jump * 0.5;
    for index in 0..radius {
        let t = smoothstep((index + 1) as f32 / (radius + 1) as f32);
        samples[join - radius + index] += half_jump * t;
        samples[join + index] -= half_jump * (1.0 - t);
    }
}

fn smoothstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

fn psola_synthesize(
    input: &[f32],
    source_centers: &[usize],
    duration_ms: u32,
    targets: &[MbrolaPitchTarget],
    sample_rate_hz: u32,
    source_period: usize,
) -> Vec<f32> {
    let output_len = duration_samples(duration_ms, sample_rate_hz).max(1);
    if input.is_empty() {
        return vec![0.0; output_len];
    }
    let source_period = source_period.max(1);
    let grain_len = (source_period * 2).max(4);
    let marks = usable_centers(input.len(), source_centers, source_period);
    if input.len() < grain_len || marks.len() < MIN_PSOLA_GRAINS {
        return linear_resample(input, output_len);
    }
    let curve = PitchCurve::new(targets, sample_rate_hz as f32 / source_period as f32);
    let window = hann(grain_len);
    let half = grain_len / 2;
    let stretch = input.len() as f32 / output_len as f32;
    let mut output = vec![0.0; output_len];
    let mut weights = vec![0.0; output_len];
    let mut destination = target_period(&curve, 0, output_len, sample_rate_hz) * 0.5;
    while destination < output_len as f32 + half as f32 {
        let source_position =
            (destination * stretch).clamp(0.0, input.len().saturating_sub(1) as f32);
        overlap_add(
            input,
            &window,
            nearest_mark(&marks, source_position),
            destination.round() as isize,
            half,
            &mut output,
            &mut weights,
        );
        destination += target_period(
            &curve,
            destination.max(0.0).round() as usize,
            output_len,
            sample_rate_hz,
        )
        .max(1.0);
    }
    for (sample, weight) in output.iter_mut().zip(weights) {
        if weight > 1.0e-6 {
            *sample = (*sample / weight).clamp(-1.0, 1.0);
        }
    }
    output
}

fn linear_resample(input: &[f32], output_len: usize) -> Vec<f32> {
    if input.is_empty() || output_len == 0 {
        return Vec::new();
    }
    if output_len == 1 {
        return vec![input[0]];
    }
    let scale = (input.len() - 1) as f32 / (output_len - 1) as f32;
    (0..output_len)
        .map(|index| {
            let position = index as f32 * scale;
            let left = position.floor() as usize;
            let right = (left + 1).min(input.len() - 1);
            let fraction = position - left as f32;
            input[left] * (1.0 - fraction) + input[right] * fraction
        })
        .collect()
}

fn usable_centers(sample_len: usize, centers: &[usize], period: usize) -> Vec<usize> {
    let mut usable = centers
        .iter()
        .copied()
        .filter(|center| {
            center.saturating_sub(period) < sample_len && center + period <= sample_len
        })
        .collect::<Vec<_>>();
    usable.sort_unstable();
    usable.dedup();
    if usable.len() >= MIN_PSOLA_GRAINS {
        return usable;
    }
    let mut generated = Vec::new();
    let mut center = period;
    while center + period <= sample_len {
        generated.push(center);
        center += period;
    }
    if generated.is_empty() {
        generated.push(sample_len / 2);
    }
    generated
}

fn nearest_mark(marks: &[usize], target: f32) -> usize {
    marks
        .iter()
        .min_by(|left, right| {
            ((**left as f32) - target)
                .abs()
                .total_cmp(&((**right as f32) - target).abs())
        })
        .copied()
        .unwrap_or_default()
}

fn overlap_add(
    input: &[f32],
    window: &[f32],
    source_center: usize,
    destination_center: isize,
    half: usize,
    output: &mut [f32],
    weights: &mut [f32],
) {
    for (index, weight) in window.iter().copied().enumerate() {
        let source = source_center as isize + index as isize - half as isize;
        let destination = destination_center + index as isize - half as isize;
        if source >= 0
            && destination >= 0
            && (source as usize) < input.len()
            && (destination as usize) < output.len()
        {
            output[destination as usize] += input[source as usize] * weight;
            weights[destination as usize] += weight;
        }
    }
}

fn hann(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let phase = std::f32::consts::TAU * index as f32 / (len - 1).max(1) as f32;
            0.5 - 0.5 * phase.cos()
        })
        .collect()
}

fn target_period(
    curve: &PitchCurve,
    sample_index: usize,
    output_len: usize,
    sample_rate_hz: u32,
) -> f32 {
    sample_rate_hz as f32
        / curve
            .hz_at(sample_index, output_len)
            .clamp(40.0, sample_rate_hz as f32 / 2.0)
}

struct PitchCurve {
    neutral_hz: f32,
    targets: Vec<MbrolaPitchTarget>,
}

impl PitchCurve {
    fn new(targets: &[MbrolaPitchTarget], neutral_hz: f32) -> Self {
        let mut targets = targets.to_vec();
        targets.sort_by_key(|target| target.percent);
        Self {
            neutral_hz,
            targets,
        }
    }

    fn hz_at(&self, sample_index: usize, output_len: usize) -> f32 {
        let Some(first) = self.targets.first().copied() else {
            return self.neutral_hz;
        };
        let percent = sample_index as f32 * 100.0 / output_len.saturating_sub(1).max(1) as f32;
        if percent <= first.percent as f32 {
            return first.hz;
        }
        for pair in self.targets.windows(2) {
            if percent <= pair[1].percent as f32 {
                let width = (pair[1].percent - pair[0].percent).max(1) as f32;
                let fraction = (percent - pair[0].percent as f32) / width;
                return pair[0].hz * (1.0 - fraction) + pair[1].hz * fraction;
            }
        }
        self.targets
            .last()
            .map_or(self.neutral_hz, |target| target.hz)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn psola_is_finite_non_silent_and_matches_requested_duration() {
        let input = (0..1600)
            .map(|index| (std::f32::consts::TAU * 180.0 * index as f32 / 16_000.0).sin())
            .collect::<Vec<_>>();
        let output = psola_synthesize(
            &input,
            &[],
            250,
            &[MbrolaPitchTarget {
                percent: 50,
                hz: 130.0,
            }],
            16_000,
            80,
        );
        assert_eq!(output.len(), 4000);
        assert!(output.iter().all(|sample| sample.is_finite()));
        assert!(output.iter().any(|sample| sample.abs() > 0.01));
    }

    #[test]
    fn missing_diphone_is_actionable_and_never_substituted() {
        let error = MbrolaDatabaseError::MissingDiphone {
            left: "a".into(),
            right: "b".into(),
            database: PathBuf::from("tiny"),
        };
        assert_eq!(error.to_string(), "missing MBROLA diphone `a-b` in tiny");
    }

    #[test]
    fn test_generated_database_renders_without_external_executable() {
        let directory = tempfile::tempdir().unwrap();
        let voice = directory.path().join("tiny");
        std::fs::write(&voice, tiny_database()).unwrap();
        let projector = MbrolaProjector {
            voice: super::super::MbrolaVoiceMetadata {
                id: "tiny".into(),
                variety: "fixture".into(),
                baseline_hz: Some(120.0),
                pitch_range_hz: Some(30.0),
            },
            symbol_map: super::super::MbrolaSymbolMap::identity("tiny-identity"),
            inventory: Default::default(),
            timing: super::super::MbrolaTimingProfile::default(),
            control_baseline_hz: None,
            control_pitch_range_hz: None,
        };
        let renderer = NativeMbrolaRenderer::load(&voice, projector).unwrap();
        let plan = PhoneTimedPlan::new(vec![
            super::super::MbrolaPhone::new("h", 80),
            super::super::MbrolaPhone::new("@", 120).with_pitch_targets(vec![
                MbrolaPitchTarget {
                    percent: 0,
                    hz: 110.0,
                },
                MbrolaPitchTarget {
                    percent: 100,
                    hz: 145.0,
                },
            ]),
        ]);
        let audio = renderer.render_plan(&plan).unwrap();
        assert_eq!(audio.len(), 3_200);
        assert!(audio.iter().all(|sample| sample.is_finite()));
        assert!(audio.iter().any(|sample| sample.abs() > 0.01));
        assert_eq!(renderer.database_path(), voice);
    }

    fn tiny_database() -> Vec<u8> {
        let units = [("_", "h"), ("h", "@"), ("@", "_")];
        let period = 80usize;
        let frames = 8usize;
        let pitch_marks = units.len() * frames;
        let mut table = Vec::new();
        let mut raw = Vec::new();
        for (unit_index, (left, right)) in units.iter().enumerate() {
            table.extend_from_slice(left.as_bytes());
            table.push(0);
            table.extend_from_slice(right.as_bytes());
            table.push(0);
            table.extend_from_slice(&320_i16.to_le_bytes());
            table.push(frames as u8);
            table.push(frames as u8);
            for sample_index in 0..frames * period {
                let phase = std::f32::consts::TAU
                    * (150.0 + unit_index as f32 * 20.0)
                    * sample_index as f32
                    / 16_000.0;
                let sample = (phase.sin() * 0.5 * i16::MAX as f32) as i16;
                raw.extend_from_slice(&sample.to_le_bytes());
            }
        }
        let mut output = Vec::new();
        output.extend_from_slice(b"MBROLA");
        output.extend_from_slice(b"2.060");
        output.extend_from_slice(&(units.len() as i16).to_le_bytes());
        output.extend_from_slice(&0_u16.to_le_bytes());
        output.extend_from_slice(&(pitch_marks as i32).to_le_bytes());
        output.extend_from_slice(&(raw.len() as i32).to_le_bytes());
        output.extend_from_slice(&16_000_i16.to_le_bytes());
        output.push(period as u8);
        output.push(1);
        output.extend(table);
        output.extend(std::iter::repeat_n(0b1010_1010, pitch_marks.div_ceil(4)));
        output.extend(raw);
        output
    }
}
