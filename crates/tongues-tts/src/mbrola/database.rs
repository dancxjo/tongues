use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use thiserror::Error;

const MAGIC: &[u8; 6] = b"MBROLA";
const HEADER_LEN: usize = 27;
const VOICING_MASK: u8 = 2;
const VOICED: u8 = VOICING_MASK;

#[derive(Debug, Clone)]
pub struct MbrolaDatabase {
    pub path: PathBuf,
    pub version: String,
    pub sample_rate_hz: u32,
    pub mbr_period: usize,
    pub coding: u8,
    raw_offset: usize,
    pitch_marks: Vec<u8>,
    diphones: BTreeMap<(String, String), MbrolaDiphone>,
    phonemes: BTreeSet<String>,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MbrolaDiphone {
    pub left: String,
    pub right: String,
    pub pos_wave_samples: usize,
    pub halfseg_samples: usize,
    pub pos_pitch_mark: usize,
    pub logical_frames: usize,
    pub physical_frames: usize,
}

impl MbrolaDatabase {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, MbrolaDatabaseError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| MbrolaDatabaseError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_bytes(path.to_path_buf(), bytes)
    }

    pub fn from_bytes(path: PathBuf, bytes: Vec<u8>) -> Result<Self, MbrolaDatabaseError> {
        let mut cursor = Cursor::new(&bytes);
        if cursor.read_exact(6)? != MAGIC {
            return Err(MbrolaDatabaseError::BadMagic);
        }
        let version = String::from_utf8_lossy(cursor.read_exact(5)?).to_string();
        let diphone_count = cursor.read_i16_le()?;
        if diphone_count < 0 {
            return Err(MbrolaDatabaseError::InvalidHeader("negative diphone count"));
        }
        let diphone_count = diphone_count as usize;
        let old_mark_size = cursor.read_u16_le()?;
        let pitch_mark_count = if old_mark_size == 0 {
            nonnegative_i32(cursor.read_i32_le()?, "pitch-mark count")?
        } else {
            old_mark_size as usize
        };
        let raw_size = nonnegative_i32(cursor.read_i32_le()?, "raw byte count")?;
        let sample_rate = cursor.read_i16_le()?;
        if sample_rate <= 0 {
            return Err(MbrolaDatabaseError::InvalidHeader("invalid sample rate"));
        }
        let sample_rate_hz = sample_rate as u32;
        let mbr_period = cursor.read_u8()? as usize;
        if mbr_period == 0 {
            return Err(MbrolaDatabaseError::InvalidHeader("zero MBR period"));
        }
        let coding = cursor.read_u8()?;
        if coding != 1 {
            return Err(MbrolaDatabaseError::UnsupportedCoding(coding));
        }
        if cursor.pos != HEADER_LEN {
            return Err(MbrolaDatabaseError::InvalidHeader("internal header length"));
        }

        let mut pitch_cursor = 0;
        let mut wave_cursor = 0;
        let mut entries = BTreeMap::new();
        let mut phonemes = BTreeSet::new();
        let mut index = 0;
        while pitch_cursor != pitch_mark_count && index < diphone_count {
            let left = cursor.read_zstring()?;
            let right = cursor.read_zstring()?;
            let halfseg = cursor.read_i16_le()?;
            if halfseg < 0 {
                return Err(MbrolaDatabaseError::InvalidHeader(
                    "negative half-diphone split",
                ));
            }
            let logical_frames = cursor.read_u8()? as usize;
            let physical_frames = cursor.read_u8()? as usize;
            let diphone = MbrolaDiphone {
                left: left.clone(),
                right: right.clone(),
                pos_wave_samples: wave_cursor,
                halfseg_samples: halfseg as usize,
                pos_pitch_mark: pitch_cursor,
                logical_frames,
                physical_frames,
            };
            pitch_cursor = pitch_cursor
                .checked_add(logical_frames)
                .ok_or(MbrolaDatabaseError::SizeOverflow)?;
            wave_cursor = wave_cursor
                .checked_add(
                    physical_frames
                        .checked_mul(mbr_period)
                        .ok_or(MbrolaDatabaseError::SizeOverflow)?,
                )
                .ok_or(MbrolaDatabaseError::SizeOverflow)?;
            phonemes.insert(left.clone());
            phonemes.insert(right.clone());
            entries.insert((left, right), diphone);
            index += 1;
        }
        if pitch_cursor != pitch_mark_count {
            return Err(MbrolaDatabaseError::InvalidHeader(
                "diphone table and pitch-mark count disagree",
            ));
        }
        while index < diphone_count {
            let old_left = cursor.read_zstring()?;
            let old_right = cursor.read_zstring()?;
            let original = entries
                .get(&(old_left, old_right))
                .cloned()
                .ok_or(MbrolaDatabaseError::MissingReplacementSource)?;
            let left = cursor.read_zstring()?;
            let right = cursor.read_zstring()?;
            entries.insert(
                (left.clone(), right.clone()),
                MbrolaDiphone {
                    left: left.clone(),
                    right: right.clone(),
                    ..original
                },
            );
            phonemes.insert(left);
            phonemes.insert(right);
            index += 1;
        }

        let pitch_marks = cursor.read_exact(pitch_mark_count.div_ceil(4))?.to_vec();
        let raw_offset = cursor.pos;
        let raw_end = raw_offset
            .checked_add(raw_size)
            .ok_or(MbrolaDatabaseError::SizeOverflow)?;
        if raw_end > bytes.len() {
            return Err(MbrolaDatabaseError::UnexpectedEof);
        }
        Ok(Self {
            path,
            version,
            sample_rate_hz,
            mbr_period,
            coding,
            raw_offset,
            pitch_marks,
            diphones: entries,
            phonemes,
            bytes,
        })
    }

    pub fn phonemes(&self) -> impl Iterator<Item = &str> {
        self.phonemes.iter().map(String::as_str)
    }

    pub fn has_diphone(&self, left: &str, right: &str) -> bool {
        self.diphones
            .contains_key(&(left.to_string(), right.to_string()))
    }

    pub fn diphone(&self, left: &str, right: &str) -> Option<&MbrolaDiphone> {
        self.diphones.get(&(left.to_string(), right.to_string()))
    }

    pub fn samples_for_diphone(
        &self,
        diphone: &MbrolaDiphone,
    ) -> Result<Vec<f32>, MbrolaDatabaseError> {
        let sample_count = self
            .physical_frame_count(diphone)
            .checked_mul(self.mbr_period)
            .ok_or(MbrolaDatabaseError::SizeOverflow)?;
        let start = self
            .raw_offset
            .checked_add(
                diphone
                    .pos_wave_samples
                    .checked_mul(2)
                    .ok_or(MbrolaDatabaseError::SizeOverflow)?,
            )
            .ok_or(MbrolaDatabaseError::SizeOverflow)?;
        let end = start
            .checked_add(
                sample_count
                    .checked_mul(2)
                    .ok_or(MbrolaDatabaseError::SizeOverflow)?,
            )
            .ok_or(MbrolaDatabaseError::SizeOverflow)?;
        if end > self.bytes.len() {
            return Err(MbrolaDatabaseError::UnexpectedEof);
        }
        Ok(self.bytes[start..end]
            .chunks_exact(2)
            .map(|sample| i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32768.0)
            .collect())
    }

    pub fn frame_centers(&self, diphone: &MbrolaDiphone) -> Vec<usize> {
        let mut centers = Vec::with_capacity(diphone.logical_frames);
        let mut physical_frame = 1;
        let mut previous = VOICED;
        for logical_frame in 0..diphone.logical_frames {
            let mark = self.pitch_mark(diphone.pos_pitch_mark + logical_frame);
            if previous & VOICING_MASK == 0 && mark & VOICING_MASK != 0 {
                physical_frame += 1;
            }
            centers.push(physical_frame * self.mbr_period);
            physical_frame += 1;
            previous = mark;
        }
        centers
    }

    fn physical_frame_count(&self, diphone: &MbrolaDiphone) -> usize {
        let mut total = 1;
        let mut previous = VOICED;
        for logical_frame in 0..diphone.logical_frames {
            let mark = self.pitch_mark(diphone.pos_pitch_mark + logical_frame);
            if previous & VOICING_MASK == 0 && mark & VOICING_MASK != 0 {
                total += 1;
            }
            total += 1;
            previous = mark;
        }
        total = total.saturating_sub(1);
        if previous & VOICING_MASK == 0 {
            total += 1;
        }
        total
    }

    fn pitch_mark(&self, index: usize) -> u8 {
        self.pitch_marks
            .get(index / 4)
            .map(|byte| (byte >> (2 * (index % 4))) & 0x03)
            .unwrap_or_default()
    }
}

fn nonnegative_i32(value: i32, field: &'static str) -> Result<usize, MbrolaDatabaseError> {
    if value < 0 {
        return Err(MbrolaDatabaseError::InvalidHeader(field));
    }
    Ok(value as usize)
}

#[derive(Debug, Error)]
pub enum MbrolaDatabaseError {
    #[error("failed to read MBROLA database {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("not an MBROLA database")]
    BadMagic,
    #[error("unsupported MBROLA database coding {0}")]
    UnsupportedCoding(u8),
    #[error("invalid MBROLA database header: {0}")]
    InvalidHeader(&'static str),
    #[error("MBROLA database ended unexpectedly")]
    UnexpectedEof,
    #[error("MBROLA database size arithmetic overflowed")]
    SizeOverflow,
    #[error("replacement diphone source was missing")]
    MissingReplacementSource,
    #[error("missing MBROLA diphone `{left}-{right}` in {database}")]
    MissingDiphone {
        left: String,
        right: String,
        database: PathBuf,
    },
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], MbrolaDatabaseError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or(MbrolaDatabaseError::SizeOverflow)?;
        if end > self.bytes.len() {
            return Err(MbrolaDatabaseError::UnexpectedEof);
        }
        let value = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(value)
    }

    fn read_u8(&mut self) -> Result<u8, MbrolaDatabaseError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u16_le(&mut self) -> Result<u16, MbrolaDatabaseError> {
        let bytes = self.read_exact(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i16_le(&mut self) -> Result<i16, MbrolaDatabaseError> {
        let bytes = self.read_exact(2)?;
        Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i32_le(&mut self) -> Result<i32, MbrolaDatabaseError> {
        let bytes = self.read_exact(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_zstring(&mut self) -> Result<String, MbrolaDatabaseError> {
        let start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos] != 0 {
            self.pos += 1;
        }
        if self.pos >= self.bytes.len() {
            return Err(MbrolaDatabaseError::UnexpectedEof);
        }
        let value = String::from_utf8_lossy(&self.bytes[start..self.pos]).to_string();
        self.pos += 1;
        Ok(value)
    }
}
