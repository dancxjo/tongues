const globalAdvancedControls = ['--cpu', '--quiet', '--verbose'];

const controlDescriptions = {
    text: 'Text passed to the command as the primary input.',
    input: 'Input text, orthography, phoneme string, or raw source value for the command.',
    cursor: 'Current cursor prefix used for sentence-boundary inference.',
    buffer: 'Raw rolling UTF-8 text buffer used by head2phones inference.',
    wav: 'WAV file path consumed by the command.',
    refs: 'Glob or directory containing reference WAV files.',
    'style-vectors': 'JSONL file containing encoded StyleTTS2 style vectors.',
    model: 'Model bundle id or model selector value.',
    '--all': 'Apply the clean operation to both prepared data and model artifacts.',
    '--archive-dir': 'Root directory where cleaned artifacts are archived.',
    '--backend': 'Speech backend: Burn SpeedySpeech, Burn FastPitch, Burn VITS, StyleTTS2, ONNX voice models, or mock output.',
    '--batch-size': 'Mini-batch size used during training.',
    '--cache-dir': 'Directory used for downloaded or cached source data.',
    '--config': 'TOML configuration file for this module.',
    '--corpus': 'Emotion corpus to fetch; repeat it to fetch a selected subset.',
    '--cpu': 'Use CPU execution instead of CUDA where the backend supports both.',
    '--cuts-per-wav': 'Number of random emotion training cuts to sample from each WAV.',
    '--data': 'Prepared dataset directory read by train, eval, infer, or clean commands.',
    '--debug-pronunciation': 'Print pronunciation planning diagnostics while speaking.',
    '--diffusion-steps': 'StyleTTS2 diffusion step count; more steps can improve quality but run slower.',
    '--dropout': 'Dropout probability used while training.',
    '--durations': 'Comma-separated positive mel-frame durations, one per projected token.',
    '--dump': 'Existing decompressed MediaWiki XML dump to parse instead of downloading one.',
    '--embedding-scale': 'StyleTTS2 diffusion embedding scale.',
    '--emotion': 'Name of the target emotion to apply from the emotion signatures file.',
    '--emotion-signatures': 'JSON file containing emotion signature delta vectors.',
    '--emotion-strength': 'Multiplier applied to the selected emotion delta vector.',
    '--epochs': 'Maximum number of training epochs.',
    '--fail-on-guessed-pronunciation': 'Treat guessed pronunciations as errors instead of allowing synthesis.',
    '--force': 'Re-download model files even when they already appear to be present.',
    '--g2p2g-model': 'G2P2G model directory used by the discrepancy report.',
    '--head2phones-model': 'Head2phones model path used to parse text for StyleTTS2 discovery.',
    '--input': 'Input file or directory. Commands that accept multiple inputs can repeat this control.',
    '--labels': 'JSONL file mapping reference paths to emotion and speaker labels.',
    '--lang': 'Wiktionary language code override; can be repeated or comma-separated.',
    '--learning-rate': 'Optimizer learning rate.',
    '--limit': 'Maximum number of sampled words included in the report.',
    '--list': 'Print available choices and exit without downloading.',
    '--mask-policy': 'Masking policy for training: single-mask or variable curriculum.',
    '--max-chars': 'Maximum character count accepted by the command.',
    '--max-mask-rate': 'Maximum fraction of phones masked when variable masking is active.',
    '--max-rarity': 'Maximum OpenEPD rarity rank included in the default discrepancy sample.',
    '--max-tts-symbols': 'Maximum symbols per TTS chunk before text is split.',
    '--max-utterances': 'Limit utterances for smoke tests or smaller prepared datasets.',
    '--max-whisper-wer': 'Maximum word error rate allowed between Whisper text and the original transcript.',
    '--max-wiktionary-audio': 'Limit imported Wiktionary/Commons pronunciation audio rows.',
    '--mel-bins': 'Log-mel bin count before mean/std pooling.',
    '--method': 'Method used to compute emotion delta signatures.',
    '--min-cut-ms': 'Minimum duration for sampled emotion audio cuts.',
    '--max-cut-ms': 'Maximum duration for sampled emotion audio cuts.',
    '--model': 'Model directory read or updated by the command.',
    '--no-create': 'Do not recreate empty default directories after archiving cleaned artifacts.',
    '--no-download-wiktionary-audio': 'Use existing Wiktionary audio only; do not fetch missing files.',
    '--no-full-cut': 'Skip adding a full-length emotion cut for each source WAV.',
    '--no-g2p2g': 'Exclude the G2P2G model from discrepancy comparison.',
    '--no-tts-chunking': 'Disable automatic text chunking before StyleTTS2 synthesis.',
    '--no-whisper-transcripts': 'Keep original LibriSpeech transcript text instead of Whisper recasing.',
    '--no-wiktionary': 'Exclude the Wiktionary model from discrepancy comparison.',
    '--no-wiktionary-audio': 'Do not import Wiktionary/Commons pronunciation audio rows.',
    '--notation': 'Pronunciation representation: phones, phonemes, or both where supported.',
    '--num-samples': 'Number of generated samples or variants to produce.',
    '--ollama-max-chars': 'Maximum JSONL characters included in an Ollama verification prompt.',
    '--ollama-model': 'Ollama model name used for passive data verification.',
    '--ollama-rows': 'Maximum train rows sent to Ollama in one verification request.',
    '--ollama-strict': 'Fail the command when Ollama reports the scanned data is not sane.',
    '--ollama-url': 'Ollama server URL used for verification requests.',
    '--out': 'Output directory or output file written by the command.',
    '--out-dir': 'Output directory written by the command.',
    '--output': 'WAV output path written by `tongues speak`.',
    '--patience': 'Early-stopping patience measured in epochs without improvement.',
    '--pitch': 'Comma-separated normalized pitch values, one per projected token.',
    '--pitch-scale': 'Multiplier for normalized pitch conditioning.',
    '--pitch-shift': 'Offset added in normalized pitch-conditioning space.',
    '--prepare': 'Prepare or rebuild data before starting the training step.',
    '--previous': 'Previously parsed sentence shown to the sentence parser model.',
    '--quality': 'StyleTTS2 preset that chooses default synthesis quality and speed tradeoffs.',
    '--quiet': 'Silence status bars and diagnostic progress output.',
    '--raw': 'Treat input as the exact model source string, including control tags.',
    '--references-dir': 'Directory of WAV files used for empirical StyleTTS2 discovery randomness.',
    '--repair-control': 'ANSI control sequence emitted before a repaired streamed sentence.',
    '--run-id': 'Archive run id. If omitted, the CLI uses a unix-seconds id.',
    '--sample-rate-hz': 'Output sample rate in Hz.',
    '--seed': 'Random seed for reproducible shuffling, training, or sampling.',
    '--sight-words': 'Enable or disable extra training copies of matching English Dolch sight-word rows.',
    '--source': 'Refinement source: held-out discrepancies or the built-in sight-word list.',
    '--source-manifest': 'Source JSONL manifest with emotion labels and audio paths.',
    '--span-mask-prob': 'Probability weight for span masking during masked-phone training.',
    '--speaker': 'Named speaker declared by the selected voice model.',
    '--speaker-reference-strength': 'Voice reference strength from 0 to 1; higher keeps more speaker timbre.',
    '--speed': 'StyleTTS2 decoder speed multiplier.',
    '--split': 'Prepared data split to evaluate, usually train, valid, or test.',
    '--splits': 'Comma-separated prepared splits mined for refinement discrepancies.',
    '--strict': 'Exit non-zero when verification reports scanned data is not sane.',
    '--style-alpha': 'Raw StyleTTS2 alpha blend; higher uses more predicted speaker/timbre and less reference.',
    '--style-beta': 'Raw StyleTTS2 beta blend; higher uses more predicted style/prosody and less reference.',
    '--style-reference-strength': 'Style reference strength from 0 to 1; higher keeps more reference prosody.',
    '--style-seed': 'Seed for StyleTTS2 style diffusion.',
    '--style-wav': 'Reference WAV for style and prosody.',
    '--task': 'Task or direction to run, such as g2p, p2g, pronunciation, normalization, or all.',
    '--tier': 'StyleTTS2 discovery tier: diffusion, empirical reference styles, or broader random search.',
    '--timings': 'Emit word and audio timing metadata.',
    '--train-frac': 'Fraction of base words assigned to the training split during prepare.',
    '--training-set': 'Prepared sentence-parser row source used for training.',
    '--subset': 'Dataset subset to prepare, such as mini or train-clean-100.',
    '--valid-frac': 'Fraction of base words assigned to validation during prepare.',
    '--variety': 'Language or pronunciation variety tag from the linguistic variety registry.',
    '--verbose': 'Show status bars and diagnostic progress output.',
    '--verify-ollama': 'Ask Ollama to passively scan prepared training rows.',
    '--voice-wav': 'Reference WAV for speaker timbre.',
    '--wav': 'WAV file consumed by the command.',
    '--wait-for-prepare': 'Wait for an in-progress prepare in the data directory before training.',
    '--weight-decay': 'AdamW weight decay.',
    '--whisper-model': 'Whisper ggml model path used for transcript recasing and punctuation.',
    '--wiktionary-audio-data': 'Prepared Wiktionary dataset used to import single-word Commons audio.',
    '--wiktionary-model': 'Wiktionary model directory used by the discrepancy report.',
    '--word': 'Explicit word to include; repeat it to compare multiple words.',
    '--words-file': 'Path to a file containing additional words, one per line.',
    'display name': 'Human-readable model bundle name shown by the CLI.',
    bundle: 'Model bundle selected by the model menu.',
    'file presence': 'Whether expected model files exist locally.',
    id: 'Stable model bundle id.',
    kind: 'Model bundle category.',
    presence: 'Whether the model bundle assets are present locally.',
    'model category': 'Model category chosen in the interactive model menu.',
    'selected model': 'Currently selected LLM model bundle.',
    'selected voice model': 'Currently selected voice model bundle.',
};

const controlOptions = {
    '--backend': ['burn', 'fastpitch', 'vits', 'styletts2', 'onnx', 'mock'],
    '--corpus': ['ravdess', 'crema-d', 'tess', 'savee', 'emodb', 'iemocap'],
    '--mask-policy': ['variable', 'single'],
    '--method': ['speaker-neutral-delta'],
    '--notation': ['phones', 'phonemes', 'all'],
    '--quality': ['balanced', 'fast'],
    '--sight-words': ['true', 'false'],
    '--source': ['discrepancies', 'sight-words'],
    '--split': ['test', 'valid', 'train'],
    '--subset': ['mini', 'train-clean-100'],
    '--training-set': ['all', 'seams', 'naive-discrepancy'],
    'model category': ['LLM', 'Voice model'],
};

const pathDefaults = {
    g2p2g: {
        config: 'configs/g2p2g/default.toml',
        data: 'datasets/g2p2g/openepd-v0',
        model: 'models/g2p2g/openepd-v0',
        outData: 'datasets/g2p2g/openepd-v0',
        outModel: 'models/g2p2g/openepd-v0',
    },
    sentenceParser: {
        config: 'configs/sentence-parser/default.toml',
        data: 'datasets/sentence-parser/v0',
        model: 'models/sentence-parser/v0',
        outData: 'datasets/sentence-parser/v0',
        outModel: 'models/sentence-parser/v0',
    },
    head2phones: {
        config: 'configs/head2phones/default.toml',
        data: 'datasets/head2phones/v0',
        model: 'models/head2phones/v0',
        outData: 'datasets/head2phones/v0',
        outModel: 'models/head2phones/v0',
    },
    interpretation: {
        data: 'datasets/interpretation/mini-v0',
        model: 'models/interpretation/mini-v0',
        outData: 'datasets/interpretation/mini-v0',
        outModel: 'models/interpretation/mini-v0',
    },
    emotions: {
        data: 'datasets/emotions/v0',
        model: 'models/emotions/v0',
        outData: 'datasets/emotions/v0',
        outModel: 'models/emotions/v0',
    },
    wiktionary: {
        config: 'configs/wiktionary/default.toml',
        data: 'datasets/wiktionary/enwiktionary-2026-06-01-v0',
        model: 'models/wiktionary/enwiktionary-2026-06-01-v0-phones',
        outData: 'datasets/wiktionary/enwiktionary-2026-06-01-v0',
        outModel: 'models/wiktionary/enwiktionary-2026-06-01-v0-phones',
        cache: 'data/wiktionary',
    },
};

const commonDefaults = {
    '--archive-dir': 'archive',
    '--batch-size': '64',
    '--backend': 'burn',
    '--corpus': 'ravdess',
    '--cuts-per-wav': '8',
    '--diffusion-steps': '5',
    '--dropout': '0.1',
    '--embedding-scale': '1.0',
    '--emotion-signatures': 'emotion_signatures.json',
    '--emotion-strength': '1.0',
    '--epochs': '20',
    '--labels': 'labels.jsonl',
    '--learning-rate': '0.0003',
    '--limit': '250',
    '--mask-policy': 'variable',
    '--max-cut-ms': '3500',
    '--max-mask-rate': '0.4',
    '--max-rarity': '50000',
    '--max-tts-symbols': '180',
    '--max-whisper-wer': '0.35',
    '--mel-bins': '80',
    '--method': 'speaker-neutral-delta',
    '--min-cut-ms': '250',
    '--notation': 'phones',
    '--num-samples': '10',
    '--ollama-model': 'gemma3:4b',
    '--ollama-rows': '24',
    '--ollama-max-chars': '12000',
    '--ollama-url': 'http://localhost:11434',
    '--out-dir': 'outputs/styletts2-discovery',
    '--output': 'output.wav',
    '--patience': '5',
    '--quality': 'balanced',
    '--repair-control': '\\u001b[1A\\u001b[2K',
    '--sample-rate-hz': '24000',
    '--seed': '42',
    '--sight-words': 'true',
    '--source': 'discrepancies',
    '--source-manifest': 'style_vectors.jsonl',
    '--span-mask-prob': '0.15',
    '--speaker-reference-strength': '0.70',
    '--speed': '1.0',
    '--split': 'test',
    '--splits': 'valid,test',
    '--style-alpha': '0.30',
    '--style-beta': '0.10',
    '--style-reference-strength': '0.90',
    '--style-seed': '0',
    '--subset': 'mini',
    '--task': 'auto',
    '--tier': '1',
    '--train-frac': '0.8',
    '--training-set': 'all',
    '--valid-frac': '0.1',
    '--weight-decay': '0.0001',
    model: 'gemma4',
    refs: 'models/styletts2/en-us/reference_audio',
    'style-vectors': 'style_vectors.jsonl',
    text: 'hello world',
};

const numericControls = new Set([
    '--batch-size',
    '--cuts-per-wav',
    '--diffusion-steps',
    '--dropout',
    '--embedding-scale',
    '--emotion-strength',
    '--epochs',
    '--learning-rate',
    '--limit',
    '--max-cut-ms',
    '--max-mask-rate',
    '--max-rarity',
    '--max-tts-symbols',
    '--max-utterances',
    '--max-whisper-wer',
    '--max-wiktionary-audio',
    '--mel-bins',
    '--min-cut-ms',
    '--num-samples',
    '--ollama-rows',
    '--ollama-max-chars',
    '--patience',
    '--sample-rate-hz',
    '--seed',
    '--span-mask-prob',
    '--speaker-reference-strength',
    '--speed',
    '--style-alpha',
    '--style-beta',
    '--style-reference-strength',
    '--style-seed',
    '--tier',
    '--train-frac',
    '--valid-frac',
    '--weight-decay',
]);

const flagControls = new Set([
    '--cpu',
    '--quiet',
    '--verbose',
]);

const filePathControls = new Set([
    '--config',
    '--data',
    '--dump',
    '--emotion-signatures',
    '--g2p2g-model',
    '--head2phones-model',
    '--input',
    '--labels',
    '--model',
    '--source-manifest',
    '--style-wav',
    '--voice-wav',
    '--wav',
    '--whisper-model',
    '--wiktionary-audio-data',
    '--wiktionary-model',
    '--words-file',
    'refs',
    'style-vectors',
    'wav',
]);

const outputPathControls = new Set([
    '--archive-dir',
    '--cache-dir',
    '--out',
    '--out-dir',
    '--output',
]);

const moduleGuides = {
    Speech: {
        intro: 'Start here when you want immediate output from text. The speech pages are good smoke tests before preparing or training larger datasets.',
        firstRun: 'Try the defaults, generate a short WAV, then swap the voice or style sample once reference audio is available.',
    },
    G2P2G: {
        intro: 'G2P2G teaches the project to translate between spellings and pronunciations using OpenEPD-style lexical data.',
        firstRun: 'Prepare first, train second, then use infer or eval. The defaults keep data in datasets/g2p2g and models in models/g2p2g.',
    },
    'Sentence Parser': {
        intro: 'The sentence parser turns streaming text into stable sentence boundaries for downstream pronunciation and speech planning.',
        firstRun: 'Prepare a small text directory first. Training can run from the prepared dataset or prepare and train in one step.',
    },
    Head2Phones: {
        intro: 'Head2Phones predicts pronunciation from rolling text buffers, which is useful when text is still arriving.',
        firstRun: 'Use the default config, prepare a dataset, optionally scan rows with Ollama, then train the default model directory.',
    },
    Interpretation: {
        intro: 'Interpretation prepares and trains audio supervision for speech recognition style workflows.',
        firstRun: 'Use the mini subset first. It is the fastest way to verify downloads, features, transcripts, and training wiring.',
    },
    Emotions: {
        intro: 'Emotion commands prepare labeled audio cuts, train an emotion classifier, and classify WAV files.',
        firstRun: 'Fetch corpora or provide a style vector manifest, prepare cuts into datasets/emotions/v0, then train models/emotions/v0.',
    },
    Wiktionary: {
        intro: 'Wiktionary commands download pronunciation data, expand it into tasks, and train pronunciation models.',
        firstRun: 'Start with the default config and cache directory. Override languages only after the full default flow is clear.',
    },
    Utilities: {
        intro: 'Utilities manage model files, fetch source data, and generate project reports.',
        firstRun: 'Run models status or models fetch first so the runtime assets are present before synthesis or training.',
    },
    StyleTTS2: {
        intro: 'StyleTTS2 tools inspect and build reference styles, emotion signatures, and discovery samples for speech synthesis.',
        firstRun: 'Encode style references, compute emotion signatures, then use the speech page to apply those signatures.',
    },
    Legacy: {
        intro: 'Legacy pages are compatibility aliases for the G2P2G workflow.',
        firstRun: 'Prefer the G2P2G pages for new work; use these only when following older notes or scripts.',
    },
};

const commandPages = [
    {
        title: 'Speech Studio',
        path: '/speech',
        aliases: ['/styletts2'],
        command: 'tongues speak',
        group: 'Speech',
        summary: 'Generate speech with any registered backend, voice model, speaker embedding, and model-specific controls.',
        implemented: true,
    },
    {
        title: 'Pronunciation Demo',
        path: '/pronunciation-demo',
        command: 'tongues g2p2g infer / tongues wiktionary infer',
        group: 'Speech',
        summary: 'Try spelling-to-pronunciation, pronunciation-to-spelling, and Wiktionary pronunciation tasks.',
        implemented: true,
        page: 'pronunciation-demo',
    },
    {
        title: 'Speak',
        path: '/cli/speak',
        command: 'tongues speak',
        group: 'Speech',
        summary: 'Synthesize text into WAV output using the selected speech backend.',
        fields: [
            { name: 'text', description: 'Text to synthesize; stdin is used in the CLI when omitted.' },
            { name: '--output', description: 'WAV file path written by the CLI.' },
            { name: '--backend', options: ['burn', 'fastpitch', 'vits', 'onnx', 'styletts2', 'mock'], default: 'burn' },
            { name: '--variety' },
        ],
        advanced: ['--sample-rate-hz', '--speaker', '--voice-wav', '--style-wav', '--quality', '--diffusion-steps', '--speaker-reference-strength', '--style-reference-strength', '--style-alpha', '--style-beta', '--emotion-signatures', '--emotion', '--emotion-strength', '--embedding-scale', '--style-seed', '--speed', '--pitch-scale', '--pitch-shift', '--pitch', '--durations', { name: '--debug-pronunciation', type: 'flag' }, { name: '--timings', type: 'flag' }, '--max-tts-symbols', { name: '--no-tts-chunking', type: 'flag' }, { name: '--fail-on-guessed-pronunciation', type: 'flag' }],
    },
    {
        title: 'Phonemes',
        path: '/cli/phonemes',
        command: 'tongues phonemes',
        group: 'Speech',
        summary: 'Convert text into a broad IPA phoneme sequence.',
        fields: ['text'],
    },
    {
        title: 'Phones',
        path: '/cli/phones',
        command: 'tongues phones',
        group: 'Speech',
        summary: 'Convert text into a narrow IPA phone sequence.',
        fields: ['text'],
    },
    {
        title: 'G2P2G',
        path: '/g2p2g/prepare',
        command: 'tongues g2p2g prepare',
        group: 'G2P2G',
        summary: 'Prepare OpenEPD train, validation, and test splits.',
        fields: ['--config', '--out'],
        advanced: ['--input', '--train-frac', '--valid-frac', '--seed'],
    },
    {
        title: 'G2P2G Clean',
        path: '/g2p2g/clean',
        command: 'tongues g2p2g clean',
        group: 'G2P2G',
        summary: 'Archive selected G2P2G artifacts and recreate default directories.',
        fields: [{ name: '--all', type: 'flag' }, { name: '--data', type: 'flag' }, { name: '--model', type: 'flag' }],
        advanced: ['--archive-dir', '--run-id', { name: '--no-create', type: 'flag' }],
    },
    {
        title: 'G2P2G Train',
        path: '/g2p2g/train',
        command: 'tongues g2p2g train',
        group: 'G2P2G',
        summary: 'Train the G2P2G seq2seq model.',
        fields: ['--config', '--data', '--out', '--task'],
        advanced: ['--mask-policy', '--max-mask-rate', '--span-mask-prob', '--learning-rate', '--weight-decay', '--dropout', '--epochs', '--patience', '--batch-size', '--seed', { name: '--wait-for-prepare', type: 'flag' }],
    },
    {
        title: 'G2P2G Infer',
        path: '/g2p2g/infer',
        command: 'tongues g2p2g infer',
        group: 'G2P2G',
        summary: 'Run grapheme-to-phoneme or phoneme-to-grapheme inference.',
        fields: ['input', '--task', '--model'],
        advanced: ['--data'],
    },
    {
        title: 'G2P2G Eval',
        path: '/g2p2g/eval',
        command: 'tongues g2p2g eval',
        group: 'G2P2G',
        summary: 'Evaluate a trained G2P2G model on a prepared split.',
        fields: ['--model', '--split', '--data', '--task'],
    },
    {
        title: 'G2P2G Refine',
        path: '/g2p2g/refine',
        command: 'tongues g2p2g refine',
        group: 'G2P2G',
        summary: 'Fine-tune a G2P2G model on held-out discrepancies or sight words.',
        fields: ['--model', '--data', '--out', '--task'],
        advanced: ['--splits', '--source', '--learning-rate', '--weight-decay', '--epochs', '--patience', '--batch-size', '--seed'],
    },
    {
        title: 'G2P2G Repl',
        path: '/g2p2g/repl',
        command: 'tongues g2p2g repl',
        group: 'G2P2G',
        summary: 'Run an interactive G2P2G translation session.',
        fields: ['--task', '--model'],
        advanced: ['--data'],
    },
    {
        title: 'Sentence Parser',
        path: '/sentence-parser/prepare',
        command: 'tongues sentence-parser prepare',
        group: 'Sentence Parser',
        summary: 'Prepare sentence parser data from text files or directories.',
        fields: ['--config', '--input', '--out'],
    },
    {
        title: 'Sentence Parser Clean',
        path: '/sentence-parser/clean',
        command: 'tongues sentence-parser clean',
        group: 'Sentence Parser',
        summary: 'Archive selected sentence parser artifacts and recreate default directories.',
        fields: [{ name: '--all', type: 'flag' }, { name: '--data', type: 'flag' }, { name: '--model', type: 'flag' }],
        advanced: ['--archive-dir', '--run-id', { name: '--no-create', type: 'flag' }],
    },
    {
        title: 'Sentence Parser Train',
        path: '/sentence-parser/train',
        command: 'tongues sentence-parser train',
        group: 'Sentence Parser',
        summary: 'Train or scaffold the sentence parser model.',
        fields: ['--config', '--data', '--out', { name: '--prepare', type: 'flag' }, '--training-set'],
        advanced: ['--input', { name: '--wait-for-prepare', type: 'flag' }, '--learning-rate', '--weight-decay', '--dropout', '--batch-size', '--epochs', '--patience', '--seed'],
    },
    {
        title: 'Sentence Parser Eval',
        path: '/sentence-parser/eval',
        command: 'tongues sentence-parser eval',
        group: 'Sentence Parser',
        summary: 'Validate a sentence parser artifact scaffold.',
        fields: ['--model', '--split'],
    },
    {
        title: 'Sentence Parser Parse',
        path: '/sentence-parser/parse',
        command: 'tongues sentence-parser parse',
        group: 'Sentence Parser',
        summary: 'Parse a sentence into the speech syntax analysis shape.',
        fields: ['text', '--model'],
    },
    {
        title: 'Sentence Parser Infer',
        path: '/sentence-parser/infer',
        command: 'tongues sentence-parser infer',
        group: 'Sentence Parser',
        summary: 'Run cursor-time sentence-boundary seq2seq inference.',
        fields: ['cursor', '--model', '--previous'],
    },
    {
        title: 'Sentence Parser Stream',
        path: '/sentence-parser/stream',
        command: 'tongues sentence-parser stream',
        group: 'Sentence Parser',
        summary: 'Stream stdin through the cursor-time sentence parser.',
        fields: ['--model', '--repair-control'],
    },
    {
        title: 'Head2Phones',
        path: '/head2phones/prepare',
        command: 'tongues head2phones prepare',
        group: 'Head2Phones',
        summary: 'Prepare rolling head-chunk-to-phones training data.',
        fields: ['--config', '--input', '--out'],
        advanced: [{ name: '--verify-ollama', type: 'flag' }, '--ollama-model', '--ollama-url', '--ollama-rows', '--ollama-max-chars', { name: '--ollama-strict', type: 'flag' }],
    },
    {
        title: 'Head2Phones Clean',
        path: '/head2phones/clean',
        command: 'tongues head2phones clean',
        group: 'Head2Phones',
        summary: 'Archive selected head2phones artifacts and recreate default directories.',
        fields: [{ name: '--all', type: 'flag' }, { name: '--data', type: 'flag' }, { name: '--model', type: 'flag' }],
        advanced: ['--archive-dir', '--run-id', { name: '--no-create', type: 'flag' }],
    },
    {
        title: 'Head2Phones Verify',
        path: '/head2phones/verify',
        command: 'tongues head2phones verify',
        group: 'Head2Phones',
        summary: 'Passively verify prepared head2phones training rows with Ollama.',
        fields: ['--config', '--data'],
        advanced: ['--ollama-model', '--ollama-url', '--ollama-rows', '--ollama-max-chars', { name: '--strict', type: 'flag' }],
    },
    {
        title: 'Head2Phones Train',
        path: '/head2phones/train',
        command: 'tongues head2phones train',
        group: 'Head2Phones',
        summary: 'Train the rolling head-chunk-to-phones seq2seq model.',
        fields: ['--config', '--data', '--out', { name: '--prepare', type: 'flag' }],
        advanced: ['--input', { name: '--verify-ollama', type: 'flag' }, '--ollama-model', '--ollama-url', '--ollama-rows', '--ollama-max-chars', { name: '--ollama-strict', type: 'flag' }, { name: '--wait-for-prepare', type: 'flag' }, '--learning-rate', '--weight-decay', '--dropout', '--batch-size', '--epochs', '--patience', '--seed'],
    },
    {
        title: 'Head2Phones Infer',
        path: '/head2phones/infer',
        command: 'tongues head2phones infer',
        group: 'Head2Phones',
        summary: 'Run rolling-buffer head2phones inference.',
        fields: ['buffer', '--model', '--variety'],
    },
    {
        title: 'Interpretation',
        path: '/interpretation/prepare',
        command: 'tongues interpretation prepare',
        group: 'Interpretation',
        summary: 'Prepare LibriSpeech audio supervision data.',
        fields: ['--subset', '--out'],
        advanced: ['--max-utterances', '--wiktionary-audio-data', { name: '--no-wiktionary-audio', type: 'flag' }, '--max-wiktionary-audio', { name: '--no-download-wiktionary-audio', type: 'flag' }, '--whisper-model', { name: '--no-whisper-transcripts', type: 'flag' }, '--max-whisper-wer'],
    },
    {
        title: 'Interpretation Clean',
        path: '/interpretation/clean',
        command: 'tongues interpretation clean',
        group: 'Interpretation',
        summary: 'Archive selected interpretation artifacts and recreate default directories.',
        fields: [{ name: '--all', type: 'flag' }, { name: '--data', type: 'flag' }, { name: '--model', type: 'flag' }],
        advanced: ['--archive-dir', '--run-id', { name: '--no-create', type: 'flag' }],
    },
    {
        title: 'Interpretation Train',
        path: '/interpretation/train',
        command: 'tongues interpretation train',
        group: 'Interpretation',
        summary: 'Train the LibriSpeech ASR model.',
        fields: ['--data', '--out'],
        advanced: [{ name: '--wait-for-prepare', type: 'flag' }, '--epochs', '--batch-size', '--seed'],
    },
    {
        title: 'Interpretation Eval',
        path: '/interpretation/eval',
        command: 'tongues interpretation eval',
        group: 'Interpretation',
        summary: 'Evaluate a LibriSpeech ASR model.',
        fields: ['--model', '--data', '--split'],
    },
    {
        title: 'Interpretation Stream',
        path: '/interpretation/stream',
        command: 'tongues interpretation stream',
        group: 'Interpretation',
        summary: 'Stream a WAV file through the ASR model.',
        fields: ['--model', '--wav'],
    },
    {
        title: 'Emotions Prepare',
        path: '/emotions/prepare',
        command: 'tongues emotions prepare',
        group: 'Emotions',
        summary: 'Prepare labeled emotion WAV cuts from a style-vector or source manifest.',
        fields: ['--source-manifest', '--out'],
        advanced: ['--cuts-per-wav', '--min-cut-ms', '--max-cut-ms', { name: '--no-full-cut', type: 'flag' }, '--mel-bins', '--seed'],
    },
    {
        title: 'Emotions Train',
        path: '/emotions/train',
        command: 'tongues emotions train',
        group: 'Emotions',
        summary: 'Train the emotion classifier on prepared acoustic cuts.',
        fields: ['--data', '--out'],
        advanced: ['--epochs', '--batch-size', '--learning-rate', '--seed'],
    },
    {
        title: 'Emotions Eval',
        path: '/emotions/eval',
        command: 'tongues emotions eval',
        group: 'Emotions',
        summary: 'Evaluate an emotion classifier on a prepared split.',
        fields: ['--model', '--data', '--split'],
    },
    {
        title: 'Emotions Infer',
        path: '/emotions/infer',
        command: 'tongues emotions infer',
        group: 'Emotions',
        summary: 'Predict emotion probabilities for one WAV file.',
        fields: ['wav', '--model'],
    },
    {
        title: 'Wiktionary',
        path: '/wiktionary/prepare',
        command: 'tongues wiktionary prepare',
        group: 'Wiktionary',
        summary: 'Download and prepare Wiktionary pronunciation data.',
        fields: ['--config', '--out', '--cache-dir'],
        advanced: ['--dump', '--lang'],
    },
    {
        title: 'Wiktionary Clean',
        path: '/wiktionary/clean',
        command: 'tongues wiktionary clean',
        group: 'Wiktionary',
        summary: 'Archive selected Wiktionary artifacts and recreate default directories.',
        fields: [{ name: '--all', type: 'flag' }, { name: '--data', type: 'flag' }, { name: '--model', type: 'flag' }],
        advanced: ['--archive-dir', '--run-id', { name: '--no-create', type: 'flag' }],
    },
    {
        title: 'Wiktionary Train',
        path: '/wiktionary/train',
        command: 'tongues wiktionary train',
        group: 'Wiktionary',
        summary: 'Train a Wiktionary pronunciation seq2seq model.',
        fields: ['--config', '--data', '--out', '--task'],
        advanced: ['--dump', '--lang', '--notation', '--cache-dir', { name: '--prepare', type: 'flag' }, '--sight-words', { name: '--wait-for-prepare', type: 'flag' }, '--learning-rate', '--weight-decay', '--dropout', '--batch-size', '--epochs', '--patience', '--seed'],
    },
    {
        title: 'Wiktionary Infer',
        path: '/wiktionary/infer',
        command: 'tongues wiktionary infer',
        group: 'Wiktionary',
        summary: 'Run pronunciation and normalization tasks with a trained Wiktionary model.',
        fields: ['input', '--model', '--task', '--lang', '--notation'],
        advanced: ['--variety', { name: '--raw', type: 'flag' }],
    },
    {
        title: 'Models Menu',
        path: '/models/menu',
        command: 'tongues models menu',
        group: 'Utilities',
        summary: 'Choose the active model through the CLI menu.',
        docs: 'Opens the interactive model picker in the CLI. Use this when you want guided selection instead of passing a model id directly.',
        fields: [],
    },
    {
        title: 'Models List',
        path: '/models/list',
        command: 'tongues models list',
        group: 'Utilities',
        summary: 'List known model bundles.',
        docs: 'Lists every known model bundle, including its kind, id, display name, and whether the expected local files are present.',
        fields: [],
    },
    {
        title: 'Models Path',
        path: '/models/path',
        command: 'tongues models path',
        group: 'Utilities',
        summary: 'Print model paths and current selection.',
        fields: ['model'],
    },
    {
        title: 'Models Status',
        path: '/models/status',
        command: 'tongues models status',
        group: 'Utilities',
        summary: 'Show selected models and local file presence.',
        docs: 'Shows the selected model bundles and whether their expected local files are present. This command has no command-specific input controls.',
        fields: [],
    },
    {
        title: 'Models Use',
        path: '/models/use',
        command: 'tongues models use',
        group: 'Utilities',
        summary: 'Select the active LLM model.',
        fields: ['model'],
    },
    {
        title: 'Models Fetch',
        path: '/models/fetch',
        command: 'tongues models fetch',
        group: 'Utilities',
        summary: 'Fetch default runtime models or a named model.',
        fields: ['model'],
        advanced: [{ name: '--force', type: 'flag' }],
    },
    {
        title: 'Fetch Corpora',
        path: '/cli/fetch-corpora',
        command: 'tongues fetch-corpora',
        group: 'Utilities',
        summary: 'Download public emotion corpora for StyleTTS2 signatures.',
        fields: ['--out-dir', '--corpus'],
        advanced: [{ name: '--list', type: 'flag' }],
    },
    {
        title: 'Fetch CMUdict',
        path: '/cli/fetch-cmudict',
        command: 'tongues fetch-cmudict',
        group: 'Utilities',
        summary: 'Download CMUdict from GitHub.',
        fields: ['--out'],
    },
    {
        title: 'Discrepancies',
        path: '/cli/discrepancies',
        command: 'tongues discrepancies',
        group: 'Utilities',
        summary: 'Compare pronunciations from lexicons, rules, and trained models.',
        fields: ['--out', '--word', '--words-file'],
        advanced: ['--limit', '--max-rarity', { name: '--no-g2p2g', type: 'flag' }, { name: '--no-wiktionary', type: 'flag' }, '--g2p2g-model', '--wiktionary-model'],
    },
    {
        title: 'StyleTTS2 Discover',
        path: '/cli/styletts2/discover',
        command: 'tongues styletts2 discover',
        group: 'StyleTTS2',
        summary: 'Sample diffusion parameters and synthesize StyleTTS2 variants.',
        fields: ['text', '--out-dir', '--num-samples'],
        advanced: ['--head2phones-model', '--variety', '--seed', '--tier', '--references-dir'],
    },
    {
        title: 'StyleTTS2 Encode Style',
        path: '/cli/styletts2/encode-style',
        command: 'tongues styletts2 encode-style',
        group: 'StyleTTS2',
        summary: 'Batch-encode reference WAV files into StyleTTS2 style vectors.',
        fields: ['refs', '--out', '--labels'],
    },
    {
        title: 'StyleTTS2 Emotion Signatures',
        path: '/cli/styletts2/emotion-signatures',
        command: 'tongues styletts2 emotion-signatures',
        group: 'StyleTTS2',
        summary: 'Compute emotion delta signatures from encoded style vectors.',
        fields: ['style-vectors', '--out'],
        advanced: ['--method'],
    },
    {
        title: 'Legacy Prepare',
        path: '/cli/prepare',
        command: 'tongues prepare',
        group: 'Legacy',
        summary: 'Compatibility alias for G2P2G prepare.',
        fields: ['--out'],
        advanced: ['--input', '--train-frac', '--valid-frac', '--seed'],
    },
    {
        title: 'Legacy Train',
        path: '/cli/train',
        command: 'tongues train',
        group: 'Legacy',
        summary: 'Compatibility alias for G2P2G train.',
        fields: ['--data', '--out', '--task'],
        advanced: ['--mask-policy', '--max-mask-rate', '--span-mask-prob', '--learning-rate', '--weight-decay', '--dropout', '--epochs', '--patience', '--batch-size', '--seed'],
    },
    {
        title: 'Legacy Eval',
        path: '/cli/eval',
        command: 'tongues eval',
        group: 'Legacy',
        summary: 'Compatibility alias for G2P2G eval.',
        fields: ['--model', '--split', '--data', '--task'],
    },
    {
        title: 'Legacy Refine',
        path: '/cli/refine',
        command: 'tongues refine',
        group: 'Legacy',
        summary: 'Compatibility alias for G2P2G refine.',
        fields: ['--model', '--data', '--out', '--task'],
        advanced: ['--splits', '--source', '--learning-rate', '--weight-decay', '--epochs', '--patience', '--batch-size', '--seed'],
    },
    {
        title: 'Legacy Repl',
        path: '/cli/repl',
        command: 'tongues repl',
        group: 'Legacy',
        summary: 'Compatibility alias for G2P2G repl.',
        fields: ['--task', '--model'],
        advanced: ['--data'],
    },
    {
        title: 'Legacy Predict',
        path: '/cli/predict',
        command: 'tongues predict',
        group: 'Legacy',
        summary: 'Compatibility alias for G2P2G infer.',
        fields: ['input', '--task', '--model'],
        advanced: ['--data'],
    },
];

const byId = (id) => document.getElementById(id);
let activePage = null;
let activeJobId = null;
let activeJobSource = null;
let jobOutputLines = [];
let jobArtifacts = [];
let lastNavigationAt = 0;

document.addEventListener('DOMContentLoaded', async () => {
    renderNavigation();
    renderRoute();
    window.addEventListener('popstate', renderRoute);
    initJobs();
    await initPronunciationDemo();
    await initSpeechStudio();
});

function renderNavigation() {
    const nav = byId('primary-nav');
    const groups = [...new Set(commandPages.map((page) => page.group))];
    nav.innerHTML = groups.map((group) => {
        const links = commandPages
            .filter((page) => page.group === group)
            .map((page) => `<a href="${page.path}" data-route="${page.path}">${page.title}</a>`)
            .join('');
        return `<div class="nav-group"><div class="nav-heading">${group}</div>${links}</div>`;
    }).join('') + '<div class="nav-group"><div class="nav-heading">Runtime</div><a href="/jobs" data-route="/jobs">Background Jobs</a></div>';

    const handleNavActivation = (event) => {
        const link = event.target.closest('a[data-route]');
        if (!link) return;
        event.preventDefault();
        navigateTo(link.getAttribute('href'));
    };
    nav.addEventListener('pointerdown', handleNavActivation);
    nav.addEventListener('click', (event) => {
        if (Date.now() - lastNavigationAt < 250) {
            event.preventDefault();
            return;
        }
        handleNavActivation(event);
    });
    nav.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        handleNavActivation(event);
    });
}

function navigateTo(path) {
    if (!path) return;
    const current = normalizePath(window.location.pathname);
    const next = normalizePath(path);
    if (current === next) {
        return;
    }
    lastNavigationAt = Date.now();
    history.pushState({}, '', path);
    const active = document.activeElement;
    if (active && typeof active.blur === 'function') {
        active.blur();
    }
    renderRoute();
}

function renderRoute() {
    const path = normalizePath(window.location.pathname);
    const jobsRoute = path === '/jobs';
    const page = commandPages.find((candidate) => path === candidate.path)
        || commandPages.find((candidate) => (candidate.aliases || []).includes(path))
        || commandPages.find((candidate) => path.startsWith(candidate.path + '/'));
    const pronunciationRoute = page?.page === 'pronunciation-demo';

    byId('speech-page').classList.toggle('hidden', jobsRoute || !page?.implemented || pronunciationRoute);
    byId('pronunciation-demo-page').classList.toggle('hidden', !pronunciationRoute);
    byId('dashboard-page').classList.toggle('hidden', jobsRoute || Boolean(page));
    byId('skeleton-page').classList.toggle('hidden', jobsRoute || !page || page.implemented);
    byId('jobs-page').classList.toggle('hidden', !jobsRoute);

    document.querySelectorAll('[data-route]').forEach((link) => {
        link.classList.toggle('active', (page && link.dataset.route === page.path) || (jobsRoute && link.dataset.route === '/jobs'));
    });

    if (jobsRoute) {
        activePage = null;
        byId('page-kicker').textContent = 'Runtime';
        byId('page-title').textContent = 'Background Jobs';
        byId('page-summary').textContent = 'Check running commands, watch output, download artifacts, or cancel work.';
        byId('page-command').textContent = 'jobs';
        loadJobs();
        return;
    }

    if (!page) {
        activePage = null;
        renderDashboard();
        byId('page-kicker').textContent = 'Command surface';
        byId('page-title').textContent = 'Tongues Web';
        byId('page-summary').textContent = 'Pick a workflow. Each page starts with safe defaults and explains what the controls do.';
        byId('page-command').textContent = 'tongues';
        return;
    }

    byId('page-kicker').textContent = page.group;
    byId('page-title').textContent = page.title;
    byId('page-summary').textContent = page.summary;
    byId('page-command').textContent = page.command;
    activePage = page;

    if (!page.implemented) {
        renderSkeleton(page);
    }
}

function renderDashboard() {
    const grid = byId('dashboard-grid');
    grid.innerHTML = commandPages.map((page) => `
        <a class="command-card" href="${page.path}" data-dashboard-route="${page.path}">
            <span>${page.group}</span>
            <strong>${page.title}</strong>
            <small>${page.command}</small>
            <p>${escapeHtml(page.summary)}</p>
        </a>
    `).join('');

    grid.querySelectorAll('[data-dashboard-route]').forEach((link) => {
        const handleDashboardActivation = (event) => {
            event.preventDefault();
            navigateTo(link.getAttribute('href'));
        };
        link.addEventListener('pointerdown', handleDashboardActivation);
        link.addEventListener('click', (event) => {
            if (Date.now() - lastNavigationAt < 250) {
                event.preventDefault();
                return;
            }
            handleDashboardActivation(event);
        });
        link.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            handleDashboardActivation(event);
        });
    });
}

function renderSkeleton(page) {
    const starter = moduleGuides[page.group] || moduleGuides.Utilities;
    byId('command-preview').value = commandExample(page);
    byId('skeleton-doc').innerHTML = `
        <p>${escapeHtml(page.docs || page.summary)}</p>
        <p>${escapeHtml(starter.intro)}</p>
        <p class="cli-equivalent">CLI equivalent: <code>${escapeHtml(page.command)}</code></p>
    `;
    byId('side-panel-title').textContent = 'First Run';
    byId('side-panel-body').innerHTML = `
        <p>${escapeHtml(starter.firstRun)}</p>
        <div class="hint-list">
            <span>1. Check the defaults.</span>
            <span>2. Open Advanced only when tuning.</span>
            <span>3. Copy the command preview into a terminal.</span>
        </div>
    `;

    const fields = page.fields || [];
    const advancedFields = [...(page.advanced || []), ...globalAdvancedControls];
    byId('skeleton-fields').innerHTML = fields.length
        ? fields.map((field) => renderControl(field, page)).join('')
        : '<p class="empty-controls">This command has no command-specific controls.</p>';
    byId('skeleton-advanced-fields').innerHTML = advancedFields.map((field) => renderControl(field, page)).join('');
    byId('skeleton-advanced').classList.toggle('hidden', advancedFields.length === 0);
    byId('skeleton-notes').value = `${commandExample(page)}\n\n${page.summary}`;
    attachFilePickers(page);
}

function renderControl(field, page) {
    const control = normalizeControl(field, page);
    const helpText = control.description || controlDescriptions[control.name] || 'CLI control for this command.';
    const description = `<small>${escapeHtml(helpText)}</small>`;
    const type = control.type || 'text';

    if (type === 'flag') {
        return `
            <label class="checkbox-row control-checkbox">
                <input type="checkbox" data-control="${escapeHtml(control.name)}">
                <span>${escapeHtml(control.name)}</span>
                ${description}
            </label>
        `;
    }

    if (control.options) {
        const options = control.options.map((option) => {
            const value = typeof option === 'object' ? option.value : option;
            const label = typeof option === 'object' ? option.label : optionLabel(option);
            const selected = value === control.default ? ' selected' : '';
            return `<option value="${escapeHtml(value)}"${selected}>${escapeHtml(label)}</option>`;
        }).join('');
        return `
            <div class="form-group">
                <label>${escapeHtml(control.name)}</label>
                <select data-control="${escapeHtml(control.name)}">${options}</select>
                ${description}
            </div>
        `;
    }

    const value = control.default !== undefined && control.default !== null
        ? ` value="${escapeHtml(control.default)}"`
        : '';
    const placeholder = control.placeholder || control.default || control.name;
    const pathControl = isPathControl(control.name);
    const input = `<input type="${type}" data-control="${escapeHtml(control.name)}" placeholder="${escapeHtml(placeholder)}"${value}>`;
    const pathButton = pathControl
        ? `<button type="button" class="browse-button" data-browse-control="${escapeHtml(control.name)}">Browse</button>`
        : '';
    const pathBrowser = pathControl
        ? `<div class="file-browser" data-file-browser="${escapeHtml(control.name)}"></div>`
        : '';

    return `
        <div class="form-group ${pathControl ? 'path-control' : ''}">
            <label>${escapeHtml(control.name)}</label>
            ${pathControl ? `<div class="path-input-row">${input}${pathButton}</div>` : input}
            ${description}
            ${pathBrowser}
        </div>
    `;
}

function isPathControl(name) {
    return filePathControls.has(name) || outputPathControls.has(name);
}

function attachFilePickers(page) {
    document.querySelectorAll('#skeleton-page [data-browse-control]').forEach((button) => {
        button.addEventListener('click', async () => {
            const group = button.closest('.form-group');
            const input = group?.querySelector('[data-control]');
            const browser = group?.querySelector('[data-file-browser]');
            if (!input || !browser) return;
            const mode = outputPathControls.has(input.dataset.control) ? 'output' : 'input';
            const startPath = pickerStartPath(input.value || defaultForControl(input.dataset.control, page) || '');
            await loadFileBrowser(browser, input, startPath, mode);
        });
    });
}

function pickerStartPath(value) {
    const text = String(value || '').trim();
    if (!text) return '';
    if (text.includes('*')) return text.split('*')[0].replace(/\/[^/]*$/, '');
    if (/\.[A-Za-z0-9]{1,8}$/.test(text.split('/').pop() || '')) {
        return text.split('/').slice(0, -1).join('/');
    }
    return text;
}

async function loadFileBrowser(browser, input, path, mode) {
    browser.classList.add('open');
    browser.innerHTML = '<div class="file-browser-status">Loading files...</div>';
    const response = await fetch(`/api/files?path=${encodeURIComponent(path || '')}`);
    if (!response.ok) {
        browser.innerHTML = `<div class="file-browser-status">${escapeHtml(await response.text())}</div>`;
        return;
    }
    const data = await response.json();
    renderFileBrowser(browser, input, data, mode);
}

function renderFileBrowser(browser, input, data, mode) {
    const intro = mode === 'output'
        ? 'Existing files here can be downloaded; choose a path to reuse it.'
        : 'Choose an existing file or directory for this command.';
    const entries = (data.entries || []).map((entry) => {
        const meta = entry.kind === 'file' && entry.size !== null && entry.size !== undefined
            ? `<span>${formatBytes(entry.size)}</span>`
            : '<span>folder</span>';
        const download = entry.download_url
            ? `<a class="download-link" href="${entry.download_url}">Download</a>`
            : '';
        return `
            <div class="file-row" data-file-path="${escapeHtml(entry.path)}" data-file-kind="${escapeHtml(entry.kind)}">
                <button type="button" class="file-name">${entry.kind === 'dir' ? 'Folder' : 'File'} ${escapeHtml(entry.name)}</button>
                ${meta}
                ${download}
            </div>
        `;
    }).join('');
    const parent = data.parent
        ? `<button type="button" class="secondary-button file-parent" data-parent-path="${escapeHtml(data.parent)}">Up one folder</button>`
        : '';
    browser.innerHTML = `
        <div class="file-browser-header">
            <div>
                <strong>${escapeHtml(data.path || '.')}</strong>
                <small>${escapeHtml(intro)}</small>
            </div>
            <button type="button" class="secondary-button file-close">Close</button>
        </div>
        ${data.error ? `<div class="file-browser-status">${escapeHtml(data.error)}</div>` : ''}
        <div class="file-browser-actions">${parent}</div>
        <div class="file-browser-list">${entries || '<div class="file-browser-status">No files found here yet.</div>'}</div>
    `;
    browser.querySelector('.file-close')?.addEventListener('click', () => {
        browser.classList.remove('open');
        browser.innerHTML = '';
    });
    browser.querySelector('[data-parent-path]')?.addEventListener('click', (event) => {
        loadFileBrowser(browser, input, event.currentTarget.dataset.parentPath, mode);
    });
    browser.querySelectorAll('[data-file-path]').forEach((row) => {
        row.querySelector('.file-name')?.addEventListener('click', () => {
            input.value = row.dataset.filePath;
            input.dispatchEvent(new Event('input', { bubbles: true }));
            if (row.dataset.fileKind === 'dir') {
                loadFileBrowser(browser, input, row.dataset.filePath, mode);
            }
        });
    });
}

function formatBytes(bytes) {
    if (!Number.isFinite(bytes)) return '';
    if (bytes < 1024) return `${bytes} B`;
    const units = ['KB', 'MB', 'GB'];
    let value = bytes / 1024;
    let unit = units.shift();
    while (value >= 1024 && units.length) {
        value /= 1024;
        unit = units.shift();
    }
    return `${value.toFixed(value >= 10 ? 0 : 1)} ${unit}`;
}

function initJobs() {
    byId('run-command')?.addEventListener('click', startCurrentPageJob);
    byId('refresh-jobs')?.addEventListener('click', loadJobs);
    byId('cancel-job')?.addEventListener('click', cancelActiveJob);
    loadJobs();
}

async function startCurrentPageJob() {
    if (!activePage) return;
    const { command, args } = buildJobRequest(activePage);
    const response = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            label: activePage.title,
            command,
            args,
        }),
    });
    if (!response.ok) {
        alert(await response.text());
        return;
    }
    const data = await response.json();
    await loadJobs();
    selectJob(data.job_id);
}

function buildJobRequest(page) {
    const args = ['run', '--bin', 'tongues', '--'];
    const commandParts = page.command.split(/\s+/).slice(1);
    const globalFlags = [];
    const optionArgs = [];
    const positional = [];

    document.querySelectorAll('#skeleton-page [data-control]').forEach((node) => {
        const name = node.dataset.control;
        if (!name) return;
        if (node.type === 'checkbox') {
            if (!node.checked) return;
            if (globalAdvancedControls.includes(name)) {
                globalFlags.push(name);
            } else {
                optionArgs.push(name);
            }
            return;
        }
        const value = node.value.trim();
        if (!value) return;
        if (name.startsWith('--')) {
            optionArgs.push(name, value);
        } else {
            positional.push(value);
        }
    });

    args.push(...globalFlags, ...commandParts, ...optionArgs, ...positional);
    return { command: 'cargo', args };
}

async function loadJobs() {
    const response = await fetch('/api/jobs');
    if (!response.ok) return;
    const jobs = await response.json();
    renderJobList(jobs);
    if (!activeJobId && jobs.length > 0) {
        selectJob(jobs[0].id);
    }
}

function renderJobList(jobs) {
    const list = byId('job-list');
    if (!jobs.length) {
        list.innerHTML = '<div class="empty-controls">No background jobs yet.</div>';
        return;
    }
    list.innerHTML = jobs.map((job) => `
        <button type="button" class="job-item ${job.id === activeJobId ? 'active' : ''}" data-job-id="${job.id}">
            <span>${escapeHtml(job.label)}</span>
            <small>${escapeHtml(job.status)} · ${escapeHtml(job.progress.phase)}</small>
        </button>
    `).join('');
    list.querySelectorAll('[data-job-id]').forEach((button) => {
        button.addEventListener('click', () => selectJob(button.dataset.jobId));
    });
}

async function selectJob(jobId) {
    activeJobId = jobId;
    if (activeJobSource) {
        activeJobSource.close();
        activeJobSource = null;
    }
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
    if (response.ok) {
        const detail = await response.json();
        renderJobDetail(detail.summary, detail.output || [], detail.artifacts || []);
    }
    activeJobSource = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
    activeJobSource.onmessage = (message) => {
        const event = JSON.parse(message.data);
        applyJobEvent(event);
    };
    activeJobSource.onerror = () => {
        // EventSource reconnects automatically while the server is available.
    };
    loadJobs();
}

function applyJobEvent(event) {
    if (event.type === 'snapshot') {
        renderJobDetail(event.summary, event.output || [], jobArtifacts);
    } else if (event.type === 'output') {
        jobOutputLines.push(event);
        renderJobOutput();
    } else if (event.type === 'progress') {
        renderProgress(event.progress);
    } else if (event.type === 'status') {
        renderJobSummary(event.summary);
        refreshActiveJobArtifacts();
        loadJobs();
    }
}

function renderJobDetail(summary, output, artifacts = []) {
    jobOutputLines = output;
    jobArtifacts = artifacts;
    renderJobSummary(summary);
    renderJobArtifacts();
    renderJobOutput();
}

function renderJobSummary(summary) {
    byId('job-title').textContent = summary.label;
    byId('job-command').textContent = `${summary.command} ${summary.args.join(' ')}`;
    renderProgress(summary.progress, summary.status);
    byId('cancel-job').classList.toggle('hidden', summary.status !== 'running');
}

function renderProgress(progress, status = 'running') {
    const complete = ['succeeded', 'failed', 'canceled'].includes(status);
    const percent = progress.total ? Math.min(100, Math.round((progress.current || 0) / progress.total * 100)) : (complete ? 100 : 35);
    byId('job-progress-bar').style.width = `${percent}%`;
    byId('job-progress-bar').classList.toggle('indeterminate', !progress.total && !complete);
    byId('job-progress-label').textContent = progress.total
        ? `${progress.phase}: ${progress.current || 0} / ${progress.total}`
        : progress.phase;
}

function renderJobOutput() {
    const output = byId('job-output');
    output.textContent = jobOutputLines
        .slice(-500)
        .map((line) => `[${line.stream}] ${line.line}`)
        .join('\n');
    output.scrollTop = output.scrollHeight;
}

async function refreshActiveJobArtifacts() {
    if (!activeJobId) return;
    const response = await fetch(`/api/jobs/${encodeURIComponent(activeJobId)}`);
    if (!response.ok) return;
    const detail = await response.json();
    jobArtifacts = detail.artifacts || [];
    renderJobArtifacts();
}

function renderJobArtifacts() {
    const container = byId('job-artifacts');
    if (!container) return;
    if (!jobArtifacts.length) {
        container.innerHTML = '<div class="artifact-empty">Output files will appear here when the command writes them.</div>';
        return;
    }
    container.innerHTML = `
        <div class="artifact-title">Files</div>
        <div class="artifact-list">
            ${jobArtifacts.map((artifact) => {
                const size = artifact.size !== null && artifact.size !== undefined ? ` · ${formatBytes(artifact.size)}` : '';
                const action = artifact.download_url
                    ? `<a class="download-link" href="${artifact.download_url}">Download</a>`
                    : `<button type="button" class="secondary-button artifact-browse" data-artifact-path="${escapeHtml(artifact.path)}">Browse</button>`;
                return `
                    <div class="artifact-row">
                        <span>${artifact.kind === 'dir' ? 'Folder' : 'File'} ${escapeHtml(artifact.path)}${escapeHtml(size)}</span>
                        ${action}
                    </div>
                `;
            }).join('')}
        </div>
    `;
    container.querySelectorAll('[data-artifact-path]').forEach((button) => {
        button.addEventListener('click', () => {
            const url = `/api/files?path=${encodeURIComponent(button.dataset.artifactPath || '')}`;
            fetch(url)
                .then((response) => response.json())
                .then((data) => {
                    const files = (data.entries || [])
                        .filter((entry) => entry.download_url)
                        .map((entry) => `<a class="download-link" href="${entry.download_url}">${escapeHtml(entry.name)}</a>`)
                        .join('');
                    button.closest('.artifact-row').insertAdjacentHTML(
                        'afterend',
                        `<div class="artifact-directory-list">${files || 'No downloadable files in this folder yet.'}</div>`,
                    );
                });
        });
    });
}

async function cancelActiveJob() {
    if (!activeJobId) return;
    const response = await fetch(`/api/jobs/${encodeURIComponent(activeJobId)}/cancel`, { method: 'POST' });
    if (!response.ok) {
        alert(await response.text());
    }
}

function normalizeControl(field, page) {
    const control = typeof field === 'string' ? { name: field } : { ...field };
    if (!control.type && numericControls.has(control.name)) {
        control.type = 'number';
    }
    if (!control.type && flagControls.has(control.name)) {
        control.type = 'flag';
    }
    control.options = control.options || optionsForControl(control.name, page);
    const fallbackDefault = defaultForControl(control.name, page);
    if (control.default === undefined && fallbackDefault !== undefined) {
        control.default = fallbackDefault;
    }
    return control;
}

function optionsForControl(name, page) {
    if (name === '--task') return taskOptionsFor(page);
    if (name === 'model' && page.command === 'tongues models use') return ['gemma4', 'styletts2', 'voice-ljspeech-high'];
    if (name === 'model' && page.command === 'tongues models fetch') {
        return [
            { value: '', label: 'Default runtime models' },
            { value: 'gemma4', label: 'Gemma 4' },
            { value: 'styletts2', label: 'StyleTTS2' },
            { value: 'voice-ljspeech-high', label: 'Voice model en-US' },
        ];
    }
    return controlOptions[name];
}

function taskOptionsFor(page) {
    if (page.command.includes('g2p2g') || page.group === 'Legacy') return ['auto', 'g2p', 'p2g', 'both'];
    if (page.command.includes('wiktionary train')) {
        return ['all', 'orthography-to-phonemes', 'orthography-to-phones', 'phonetic-realization', 'find-etymology', 'normalize-phonology', 'lang'];
    }
    if (page.command.includes('wiktionary infer')) {
        return ['orthography-to-phones', 'orthography-to-phonemes', 'phones-to-orthography', 'phonemes-to-orthography', 'phonetic-realization', 'find-etymology', 'normalize'];
    }
    return ['auto'];
}

function defaultForControl(name, page) {
    const module = moduleDefaultsFor(page);
    if (name === '--config') return module?.config;
    if (name === '--data') return module?.data;
    if (name === '--model') return module?.model;
    if (name === '--cache-dir') return module?.cache;
    if (name === '--out') return outDefaultFor(page, module);
    if (name === '--g2p2g-model') return pathDefaults.g2p2g.model;
    if (name === '--wiktionary-model') return pathDefaults.wiktionary.model;
    if (name === '--wiktionary-audio-data') return pathDefaults.wiktionary.data;
    if (name === '--head2phones-model') return pathDefaults.head2phones.model;
    if (name === '--input' && page.command.includes('sentence-parser')) return 'data/texts';
    if (name === '--input' && page.command.includes('head2phones')) return 'data/texts';
    if (name === '--dump') return 'data/wiktionary/enwiktionary-latest-pages-articles.xml';
    if (name === '--lang') return 'eng';
    if (name === '--word') return 'example';
    if (name === '--words-file') return 'data/words.txt';
    if (name === '--wav') return 'samples/input.wav';
    if (name === 'wav') return 'samples/input.wav';
    if (name === 'cursor') return 'The first sentence starts here';
    if (name === 'buffer') return 'hello world';
    if (name === 'input') return inputDefaultFor(page);
    if (name === 'bundle') return 'gemma4';
    if (name === 'model' && page.command === 'tongues models fetch') return '';
    if (name === 'model category') return 'LLM';
    return commonDefaults[name];
}

function moduleDefaultsFor(page) {
    if (page.command.includes('g2p2g') || page.group === 'Legacy') return pathDefaults.g2p2g;
    if (page.command.includes('sentence-parser')) return pathDefaults.sentenceParser;
    if (page.command.includes('head2phones')) return pathDefaults.head2phones;
    if (page.command.includes('interpretation')) return pathDefaults.interpretation;
    if (page.command.includes('emotions')) return pathDefaults.emotions;
    if (page.command.includes('wiktionary')) return pathDefaults.wiktionary;
    return undefined;
}

function outDefaultFor(page, module) {
    if (page.command.includes(' prepare')) return module?.outData || commonDefaults['--out'];
    if (page.command.includes(' train')) return module?.outModel || commonDefaults['--out'];
    if (page.command.includes('refine')) return 'models/g2p2g/openepd-v0-refined';
    if (page.command.includes('fetch-cmudict')) return 'data/cmudict.dict';
    if (page.command.includes('discrepancies')) return 'docs/pronunciation-discrepancies.md';
    if (page.command.includes('encode-style')) return 'style_vectors.jsonl';
    if (page.command.includes('emotion-signatures')) return 'emotion_signatures.json';
    return commonDefaults['--out'];
}

function inputDefaultFor(page) {
    if (page.command.includes('phonemes') || page.command.includes('phones')) return 'hello world';
    if (page.command.includes('wiktionary')) return 'example';
    if (page.command.includes('g2p2g') || page.group === 'Legacy') return 'example';
    return 'hello world';
}

function commandExample(page) {
    const controls = [...(page.fields || []), ...(page.advanced || [])]
        .map((field) => normalizeControl(field, page))
        .filter((control) => control.default !== undefined && control.default !== '' && control.type !== 'flag')
        .slice(0, 5);
    const args = controls.map((control) => {
        if (control.name.startsWith('--')) return `${control.name} ${quoteArg(control.default)}`;
        return quoteArg(control.default);
    });
    return [page.command, ...args].join(' ');
}

function quoteArg(value) {
    const text = String(value);
    return /\s/.test(text) ? `"${text.replaceAll('"', '\\"')}"` : text;
}

function optionLabel(option) {
    return String(option)
        .split('-')
        .map((part) => part ? part.charAt(0).toUpperCase() + part.slice(1) : part)
        .join(' ');
}

function escapeHtml(value) {
    return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function normalizePath(path) {
    if (path.length > 1 && path.endsWith('/')) return path.slice(0, -1);
    return path;
}

async function initPronunciationDemo() {
    const form = byId('pronunciation-form');
    if (!form) return;

    const familySelect = byId('pronunciation-family');
    const modelSelect = byId('pronunciation-model');
    const taskSelect = byId('pronunciation-task');
    const langSelect = byId('pronunciation-lang');
    const varietySelect = byId('pronunciation-variety');
    const notationSelect = byId('pronunciation-notation');
    const input = byId('pronunciation-input');
    const rawInput = byId('pronunciation-raw');
    const cpuInput = byId('pronunciation-cpu');
    const submit = byId('pronunciation-submit');
    const result = byId('pronunciation-result');
    const output = byId('pronunciation-output');
    const command = byId('pronunciation-command');
    const source = byId('pronunciation-source');
    const sourceBlock = byId('pronunciation-source-block');

    let metadata = {
        models: [],
        languages: [],
        varieties: [],
        wiktionary_tasks: [],
        g2p2g_tasks: [],
        notations: [],
    };

    const fillOptions = (select, options, selectedValue) => {
        select.innerHTML = options.map((option) => {
            const disabled = option.available === false ? ' disabled' : '';
            const suffix = option.available === false ? ' (missing files)' : '';
            const selected = option.value === selectedValue || option.path === selectedValue ? ' selected' : '';
            return `<option value="${escapeHtml(option.value || option.path)}"${selected}${disabled}>${escapeHtml(option.label + suffix)}</option>`;
        }).join('');
    };

    const syncFamilyControls = () => {
        const family = familySelect.value;
        const wiktionary = family === 'wiktionary';
        document.querySelectorAll('.wiktionary-only').forEach((node) => {
            node.classList.toggle('hidden', !wiktionary);
        });

        const models = metadata.models
            .filter((model) => model.family === family)
            .map((model) => ({ ...model, value: model.path }));
        fillOptions(modelSelect, models, models.find((model) => model.available)?.path || models[0]?.path || '');
        fillOptions(taskSelect, wiktionary ? metadata.wiktionary_tasks : metadata.g2p2g_tasks);
        if (wiktionary) {
            fillOptions(langSelect, metadata.languages, metadata.languages[0]?.value || '');
            fillOptions(varietySelect, metadata.varieties, '');
            fillOptions(notationSelect, metadata.notations, 'phones');
        }
        input.placeholder = wiktionary ? 'example, kæt, <SEGMENT> water-bottle, or raw tagged input' : 'example or ˈfɑɹ.kəl';
    };

    try {
        const response = await fetch('/api/pronunciation-demo/models');
        if (!response.ok) throw new Error(await response.text());
        metadata = await response.json();
        syncFamilyControls();
    } catch (error) {
        modelSelect.innerHTML = `<option>${escapeHtml(error.message)}</option>`;
        submit.disabled = true;
    }

    familySelect.addEventListener('change', syncFamilyControls);
    taskSelect.addEventListener('change', () => {
        const task = taskSelect.value;
        if (familySelect.value === 'g2p2g') {
            input.value = task === 'p2g' ? 'ˈfɑɹ.kəl' : 'example';
        } else if (task.includes('to-orthography') || task.includes('phonology')) {
            input.value = task.startsWith('phonemes') ? 'wʌn' : 'wʌn';
        } else if (task === 'phonetic-realization') {
            input.value = 'wʌn';
        } else if (task === 'segment-compound') {
            input.value = 'water-bottle';
        } else if (task === 'pronounce-segments') {
            input.value = 'water + bottle';
        } else if (task === 'verify-pronunciation') {
            input.value = 'one => wʌn';
        } else {
            input.value = 'example';
        }
    });

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (!input.value.trim()) return;
        submit.classList.add('loading');
        submit.disabled = true;
        result.classList.add('hidden');

        try {
            const payload = {
                family: familySelect.value,
                model: modelSelect.value,
                input: input.value,
                task: taskSelect.value,
                lang: langSelect.value,
                variety: varietySelect.value,
                notation: notationSelect.value,
                raw: rawInput.checked,
                cpu: cpuInput.checked,
            };
            const response = await fetch('/api/pronunciation-demo/infer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!response.ok) throw new Error(await response.text());
            const data = await response.json();
            output.textContent = data.output || '(empty output)';
            command.textContent = ['cargo', ...data.command].map(quoteArg).join(' ');
            source.textContent = data.source || '';
            sourceBlock.classList.toggle('hidden', !data.source);
            result.classList.remove('hidden');
        } catch (error) {
            output.textContent = error.message;
            command.textContent = '';
            source.textContent = '';
            sourceBlock.classList.add('hidden');
            result.classList.remove('hidden');
        } finally {
            submit.classList.remove('loading');
            submit.disabled = false;
        }
    });
}

async function initSpeechStudio() {
    const emotionSelect = byId('emotion');
    const strengthInput = byId('emotion_strength');
    const strengthVal = byId('strength-val');
    const emotionDetail = byId('emotion-detail');
    const strengthPresets = byId('strength-presets');
    const voiceSelect = byId('voice_sample');
    const styleSelect = byId('style_sample');
    const voicePreview = byId('voice-preview');
    const stylePreview = byId('style-preview');
    const blendMode = byId('blend_mode');
    const quietInput = byId('quiet');
    const verboseInput = byId('verbose');
    const form = byId('synth-form');
    const btn = byId('submit-btn');
    const resultContainer = byId('result-container');
    const audioPlayer = byId('audio-player');
    const runtimeState = byId('speech-runtime-state');
    const runtimeDetail = byId('speech-runtime-detail');
    const reloadRuntimeButton = byId('reload-speech-runtime');
    const backendSelect = byId('backend');
    const modelSelect = byId('speech_model');
    const modelDetail = byId('speech-model-detail');
    const speakerSelect = byId('speaker');
    const speakerDetail = byId('speaker-detail');
    const speakerControl = byId('speaker-control');

    if (!form) return;

    const emotions = new Map();
    const samples = new Map();
    let speechModels = [];

    const renderRuntime = (runtime) => {
        const state = runtime.state || (runtime.busy ? 'busy' : 'idle');
        runtimeState.dataset.state = state;
        runtimeState.textContent = state;

        const details = [`${runtime.device || 'unknown'} device`];
        if (Number.isInteger(runtime.active) && Number.isInteger(runtime.queued)) {
            details.push(`${runtime.active} active · ${runtime.queued} queued · ${runtime.capacity} max`);
        }
        if ((runtime.loaded || []).length > 0) {
            details.push(`loaded: ${runtime.loaded.join(', ')}`);
        }
        const failures = Object.entries(runtime.failed || {});
        if (failures.length > 0) {
            details.push(failures.map(([engine, error]) => `${engine}: ${error}`).join(' · '));
        } else if (state === 'idle') {
            details.push('models load on first request');
        }
        runtimeDetail.textContent = details.join(' · ');
    };

    const loadRuntime = async () => {
        const response = await fetch('/api/speech/runtime', { cache: 'no-store' });
        if (!response.ok) throw new Error(await response.text());
        const runtime = await response.json();
        renderRuntime(runtime);
        return runtime;
    };

    reloadRuntimeButton.addEventListener('click', async () => {
        reloadRuntimeButton.classList.add('loading');
        reloadRuntimeButton.disabled = true;
        try {
            const response = await fetch('/api/speech/runtime/reload', { method: 'POST' });
            if (!response.ok) throw new Error(await response.text());
            renderRuntime(await response.json());
        } catch (error) {
            runtimeState.dataset.state = 'failed';
            runtimeState.textContent = 'failed';
            runtimeDetail.textContent = `Reload failed: ${error.message}`;
        } finally {
            reloadRuntimeButton.classList.remove('loading');
            reloadRuntimeButton.disabled = false;
            loadRuntime().catch((error) => {
                runtimeDetail.textContent = `Runtime status unavailable: ${error.message}`;
            });
        }
    });

    const numericControls = [
        ['diffusion_steps', 'diffusion-steps-val', 0],
        ['speed', 'speed-val', 2],
        ['speaker_reference_strength', 'speaker-strength-val', 2],
        ['style_reference_strength', 'style-strength-val', 2],
        ['style_alpha', 'alpha-val', 2],
        ['style_beta', 'beta-val', 2],
        ['embedding_scale', 'embedding-scale-val', 2],
        ['noise_scale', 'noise-scale-val', 2],
        ['duration_noise_scale', 'duration-noise-scale-val', 2],
        ['pitch_scale', 'pitch-scale-val', 2],
        ['pitch_shift', 'pitch-shift-val', 2],
    ];

    const setStrength = (value) => {
        strengthInput.value = value.toFixed(2);
        strengthVal.textContent = value.toFixed(2);
    };

    const formatEmotionName = (name) => name.charAt(0).toUpperCase() + name.slice(1);

    const formatDuration = (durationMs) => {
        if (!durationMs) return '';
        const seconds = durationMs / 1000;
        return ` (${seconds.toFixed(1)} s)`;
    };

    const updateEmotionDetail = () => {
        const selected = emotions.get(emotionSelect.value);
        if (!selected) {
            emotionDetail.textContent = 'No emotion signature selected';
            return;
        }

        const stats = selected.stats || {};
        const sampleCount = stats.sample_count || 0;
        const speakerCount = stats.n_speakers || 0;
        emotionDetail.textContent = `${selected.dims || selected.vector.length} dims · ${speakerCount} speakers · ${sampleCount} samples`;
    };

    const updatePreview = (select, audio) => {
        const sample = samples.get(select.value);
        if (!sample) {
            audio.removeAttribute('src');
            audio.classList.add('empty');
            return;
        }
        audio.src = sample.audio_url;
        audio.classList.remove('empty');
    };

    const syncBlendMode = () => {
        const styleTts2 = backendSelect.value === 'styletts2';
        const raw = blendMode.value === 'raw';
        document.querySelectorAll('.blend-strength').forEach((node) => node.classList.toggle('hidden', !styleTts2 || raw));
        document.querySelectorAll('.blend-raw').forEach((node) => node.classList.toggle('hidden', !styleTts2 || !raw));
    };

    numericControls.forEach(([inputId, outputId, precision]) => {
        const input = byId(inputId);
        const output = byId(outputId);
        const sync = () => {
            const value = Number(input.value);
            output.textContent = precision === 0 ? String(value) : value.toFixed(precision);
        };
        input.addEventListener('input', sync);
        sync();
    });

    strengthInput.addEventListener('input', (event) => {
        setStrength(parseFloat(event.target.value));
    });

    emotionSelect.addEventListener('change', () => {
        const selected = emotions.get(emotionSelect.value);
        if (selected && selected.recommended_strength) {
            setStrength(selected.recommended_strength.normal || 0.65);
        }
        updateEmotionDetail();
    });

    strengthPresets.addEventListener('click', (event) => {
        const button = event.target.closest('button[data-preset]');
        const selected = emotions.get(emotionSelect.value);
        if (!button || !selected || !selected.recommended_strength) return;
        const value = selected.recommended_strength[button.dataset.preset];
        if (typeof value === 'number') {
            setStrength(value);
        }
    });

    voiceSelect.addEventListener('change', () => updatePreview(voiceSelect, voicePreview));
    styleSelect.addEventListener('change', () => updatePreview(styleSelect, stylePreview));
    blendMode.addEventListener('change', syncBlendMode);
    byId('quality').addEventListener('change', (event) => {
        const diffusionSteps = byId('diffusion_steps');
        diffusionSteps.value = event.target.value === 'fast' ? '2' : '5';
        diffusionSteps.dispatchEvent(new Event('input'));
    });
    quietInput.addEventListener('change', () => {
        if (quietInput.checked) verboseInput.checked = false;
    });
    verboseInput.addEventListener('change', () => {
        if (verboseInput.checked) quietInput.checked = false;
    });
    syncBlendMode();

    const loadEmotions = async () => {
        const res = await fetch('/api/emotions');
        const data = await res.json();

        if (data.error) {
            emotionDetail.textContent = data.error;
            return;
        }

        if (data.emotions && data.emotions.length > 0) {
            data.emotions.forEach((em) => {
                if (!Array.isArray(em.vector)) return;
                emotions.set(em.name, em);
                const option = document.createElement('option');
                option.value = em.name;
                option.textContent = formatEmotionName(em.name);
                emotionSelect.appendChild(option);
            });
            emotionDetail.textContent = `${data.emotions.length} emotion signatures loaded`;
        } else {
            emotionDetail.textContent = 'No emotion signatures found';
        }
    };

    const loadSamples = async () => {
        const res = await fetch('/api/styletts2-samples');
        const data = await res.json();
        if (data.error) {
            const option = document.createElement('option');
            option.textContent = data.error;
            option.disabled = true;
            voiceSelect.appendChild(option.cloneNode(true));
            styleSelect.appendChild(option);
            return;
        }

        data.samples.forEach((sample) => {
            samples.set(sample.id, sample);
            const voiceOption = document.createElement('option');
            voiceOption.value = sample.id;
            voiceOption.textContent = `${sample.label}${formatDuration(sample.duration_ms)}`;
            const styleOption = voiceOption.cloneNode(true);
            voiceSelect.appendChild(voiceOption);
            styleSelect.appendChild(styleOption);
        });

        const defaults = data.defaults || {};
        if (samples.has(defaults.voice)) voiceSelect.value = defaults.voice;
        if (samples.has(defaults.style)) styleSelect.value = defaults.style;
        updatePreview(voiceSelect, voicePreview);
        updatePreview(styleSelect, stylePreview);
    };

    const selectedModel = () => speechModels.find((model) => (
        model.backend === backendSelect.value && model.model === modelSelect.value
    ));

    const backendLabels = {
        styletts2: 'StyleTTS2',
        burn: 'Burn SpeedySpeech + HiFi-GAN',
        fastpitch: 'Burn FastPitch + HiFi-GAN',
        vits: 'Burn VITS',
        onnx: 'ONNX voice',
        mock: 'Mock',
    };

    const renderBackends = () => {
        const previous = backendSelect.value;
        const backends = [...new Set(speechModels.map((model) => model.backend))];
        backendSelect.innerHTML = '';
        backends.forEach((backend) => {
            const option = document.createElement('option');
            option.value = backend;
            option.textContent = backendLabels[backend] || backend;
            backendSelect.appendChild(option);
        });
        if (backends.includes(previous)) {
            backendSelect.value = previous;
        } else if (backends.includes('burn')) {
            backendSelect.value = 'burn';
        }
        backendSelect.disabled = backends.length < 2;
    };

    const renderSpeakerEmbeddings = (model) => {
        speakerSelect.innerHTML = '<option value="">No speaker embedding</option>';
        speakerSelect.disabled = true;
        speakerControl.classList.add('hidden');
        speakerDetail.textContent = 'This model does not expose speaker embeddings.';

        const values = model?.speakers?.values || {};
        if (values.support !== 'listed') return;
        const speakers = values.values || [];
        speakerControl.classList.remove('hidden');
        speakers.forEach((speaker) => {
            const option = document.createElement('option');
            option.value = speaker.id;
            option.textContent = speaker.numeric_id == null
                ? speaker.label
                : `${speaker.label} · embedding ${speaker.numeric_id}`;
            speakerSelect.appendChild(option);
        });
        speakerSelect.disabled = speakers.length === 0;
        if (speakers.some((speaker) => speaker.id === 'p225')) {
            speakerSelect.value = 'p225';
        }
        speakerDetail.textContent = `${speakers.length} learned embeddings from ${model.display_name || model.model}`;
    };

    const renderModelControls = () => {
        const backend = backendSelect.value || 'styletts2';
        const model = selectedModel();
        document.querySelectorAll('[data-speech-backends]').forEach((node) => {
            const backends = node.dataset.speechBackends.split(/\s+/);
            node.classList.toggle('hidden', !backends.includes(backend));
        });
        byId('speed-control').classList.toggle('hidden', !model?.speed);
        byId('seed-control').classList.toggle('hidden', !model?.seed);
        byId('pitch-scale-control').classList.toggle('hidden', !model?.pitch?.scale);
        byId('pitch-shift-control').classList.toggle('hidden', !model?.pitch?.shift);
        byId('pitch-values-control').classList.toggle('hidden', !model?.pitch?.explicit_values);
        byId('durations-control').classList.toggle('hidden', !model?.durations);
        renderSpeakerEmbeddings(model);
        syncBlendMode();

        if (!model) {
            modelDetail.textContent = `No models are registered for ${backend}.`;
            btn.disabled = true;
            return;
        }
        btn.disabled = !model.installed;
        const rate = model.output?.sample_rate_hz;
        const description = [model.model, rate ? `${rate} Hz` : null].filter(Boolean).join(' · ');
        modelDetail.textContent = model.installed
            ? `${description} · installed`
            : `${description} · ${model.error || 'model files are not installed'}`;
    };

    const renderModels = () => {
        const backend = backendSelect.value || 'styletts2';
        const models = speechModels.filter((model) => model.backend === backend);
        const previous = modelSelect.value;
        modelSelect.innerHTML = '';
        models.forEach((model) => {
            const option = document.createElement('option');
            option.value = model.model;
            option.textContent = `${model.display_name || model.model}${model.installed ? '' : ' · not installed'}`;
            modelSelect.appendChild(option);
        });
        const preferred = models.find((model) => model.model === previous)
            || models.find((model) => model.selected && model.installed)
            || models.find((model) => model.installed)
            || models.find((model) => model.selected)
            || models[0];
        if (preferred) modelSelect.value = preferred.model;
        modelSelect.disabled = models.length < 2;
        renderModelControls();
    };

    const loadModels = async () => {
        const res = await fetch('/api/speech/models');
        if (!res.ok) throw new Error(await res.text());
        speechModels = await res.json();
        renderBackends();
        renderModels();
    };

    const parseNumberList = (inputId, label, { positiveIntegers = false } = {}) => {
        const source = byId(inputId).value.trim();
        if (!source) return null;
        const parts = source.split(',').map((part) => part.trim());
        const values = parts.map(Number);
        const valid = parts.every(Boolean) && (
            positiveIntegers
                ? values.every((value) => Number.isSafeInteger(value) && value > 0)
                : values.every(Number.isFinite)
        );
        if (!valid) {
            throw new Error(
                positiveIntegers
                    ? `${label} must contain comma-separated positive integers.`
                    : `${label} must contain comma-separated numbers.`,
            );
        }
        return values;
    };

    const loadVarieties = async () => {
        const varietySelect = byId('variety');
        const res = await fetch('/api/linguistic/varieties');
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        varietySelect.innerHTML = '';
        (data.varieties || []).forEach((variety) => {
            const option = document.createElement('option');
            option.value = variety.value;
            option.textContent = variety.label;
            varietySelect.appendChild(option);
        });
        if ((data.varieties || []).some((variety) => variety.value === data.default)) {
            varietySelect.value = data.default;
        }
    };

    await Promise.all([
        loadEmotions().catch((err) => {
            console.error('Failed to load emotions', err);
            emotionDetail.textContent = 'Failed to load emotion signatures';
        }),
        loadSamples().catch((err) => {
            console.error('Failed to load StyleTTS2 samples', err);
        }),
        loadModels().catch((err) => {
            console.error('Failed to load speech models', err);
            modelDetail.textContent = 'Failed to load speech model inventory';
            speakerDetail.textContent = 'Failed to load speaker embeddings';
        }),
        loadVarieties().catch((err) => {
            console.error('Failed to load linguistic varieties', err);
            byId('variety').innerHTML = '<option value="">Variety registry unavailable</option>';
        }),
        loadRuntime().catch((err) => {
            console.error('Failed to load speech runtime', err);
            runtimeState.dataset.state = 'failed';
            runtimeState.textContent = 'unavailable';
            runtimeDetail.textContent = `Runtime status unavailable: ${err.message}`;
        }),
    ]);

    backendSelect.addEventListener('change', () => {
        renderModels();
        loadRuntime().catch((err) => {
            console.error('Failed to refresh speech runtime', err);
        });
    });
    modelSelect.addEventListener('change', renderModelControls);

    form.addEventListener('submit', async (event) => {
        event.preventDefault();

        const text = byId('text').value;
        const emotion = emotionSelect.value;
        const selectedEmotion = emotions.get(emotion);
        const strength = parseFloat(strengthInput.value);

        if (!text.trim()) return;

        btn.classList.add('loading');
        btn.disabled = true;
        resultContainer.classList.add('hidden');
        loadRuntime().catch(() => {});
        const runtimePoll = window.setInterval(() => {
            loadRuntime().catch(() => {});
        }, 750);

        try {
            const backend = backendSelect.value || 'styletts2';
            const model = selectedModel();
            const payload = {
                text,
                cpu: byId('cpu').checked,
                quiet: quietInput.checked,
                verbose: verboseInput.checked,
                variety: byId('variety').value || null,
                backend,
                model: modelSelect.value || null,
                speaker: speakerControl.classList.contains('hidden') ? null : (speakerSelect.value || null),
                speed: Number(byId('speed').value),
                max_tts_symbols: Number(byId('max_tts_symbols').value),
                no_tts_chunking: byId('no_tts_chunking').checked,
                debug_pronunciation: byId('debug_pronunciation').checked,
                timings: byId('timings').checked,
                fail_on_guessed_pronunciation: byId('fail_on_guessed_pronunciation').checked,
            };

            if (model?.seed) {
                payload.seed = Number(byId('synthesis_seed').value || 0);
            }
            if (backend === 'mock') {
                payload.sample_rate_hz = Number(byId('sample_rate_hz').value);
            }
            if (backend === 'onnx' || backend === 'vits') {
                payload.noise_scale = Number(byId('noise_scale').value);
                payload.duration_noise_scale = Number(byId('duration_noise_scale').value);
            }
            if (model?.pitch?.scale) {
                payload.pitch_scale = Number(byId('pitch_scale').value);
            }
            if (model?.pitch?.shift) {
                payload.pitch_shift = Number(byId('pitch_shift').value);
            }
            if (model?.pitch?.explicit_values) {
                payload.pitch = parseNumberList('pitch_values', 'Per-token pitch');
            }
            if (model?.durations) {
                payload.durations = parseNumberList(
                    'durations',
                    'Per-token durations',
                    { positiveIntegers: true },
                );
            }
            if (backend === 'styletts2') {
                payload.voice_sample = voiceSelect.value || null;
                payload.style_sample = styleSelect.value || null;
                payload.emotion = emotion || null;
                payload.emotion_vector = selectedEmotion ? selectedEmotion.vector : null;
                payload.emotion_strength = emotion ? strength : null;
                payload.quality = byId('quality').value;
                payload.diffusion_steps = Number(byId('diffusion_steps').value);
                payload.embedding_scale = Number(byId('embedding_scale').value);
                if (blendMode.value === 'raw') {
                    payload.style_alpha = Number(byId('style_alpha').value);
                    payload.style_beta = Number(byId('style_beta').value);
                } else {
                    payload.speaker_reference_strength = Number(byId('speaker_reference_strength').value);
                    payload.style_reference_strength = Number(byId('style_reference_strength').value);
                }
            }

            const response = await fetch('/api/speak', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                const textErr = await response.text();
                throw new Error(textErr);
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);

            audioPlayer.src = url;
            resultContainer.classList.remove('hidden');
            audioPlayer.play().catch((error) => console.log('Auto-play prevented', error));
        } catch (err) {
            alert(`Synthesis Error: ${err.message}`);
        } finally {
            window.clearInterval(runtimePoll);
            btn.classList.remove('loading');
            renderModelControls();
            loadRuntime().catch((err) => {
                console.error('Failed to refresh speech runtime', err);
            });
        }
    });
}
