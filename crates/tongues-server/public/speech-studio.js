(function speechStudioModule(root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    if (root) root.SpeechStudio = api;
}(typeof window !== 'undefined' ? window : null, function buildSpeechStudio() {
    'use strict';

    const browser = typeof window !== 'undefined' ? window : null;
    const state = {
        discovery: null,
        pathKey: '',
        presetId: '',
        values: new Map(),
        samples: [],
        sampleDefaults: {},
        emotions: [],
        audioInputCapabilities: null,
        languageRoutingCapabilities: null,
        audioUrl: null,
        speechRun: null,
        restoringSpeechSelection: false,
        runtimeTimer: null,
        runtimePollController: null,
        runtimePollGeneration: 0,
        verificationGeneration: 0,
        catalogDiscovery: null,
        catalogView: 'ready',
        catalogRequestGeneration: 0,
        workflow: 'speak',
        compareResults: new Map(),
        comparePreferred: '',
        userRecipes: [],
        selectedStage: 'generator',
        jobStreams: new Map(),
        liveProviders: [],
        liveMessages: [],
        liveConversation: { schema_version: 1, messages: [] },
        liveSessionId: null,
        liveDurableEvents: [],
        liveJournal: [],
        liveTurn: null,
        liveGeneration: 0,
        liveSynthesisTail: Promise.resolve(),
        liveAudioContext: null,
        liveNextAudioTime: 0,
        liveSources: new Set(),
        livePlaybackPending: new Map(),
        liveSegments: new Map(),
        liveAudioBuffers: [],
        liveGenerated: '',
        liveCommitted: '',
        liveSpoken: '',
        liveFinalTokenAt: 0,
        liveFirstAudioAt: 0,
        browserMicSocket: null,
        browserMicContext: null,
        browserMicStream: null,
        browserMicSource: null,
        browserMicNode: null,
        browserMicGain: null,
        browserMicSequence: 0,
        browserMicStartFrame: 0,
        conversationMicSocket: null,
        conversationMicContext: null,
        conversationMicStream: null,
        conversationMicSource: null,
        conversationMicNode: null,
        conversationMicStartedAt: 0,
        conversationFirstPartialAt: 0,
        conversationCommittedAt: 0,
        duplexRequest: null,
    };
    const VERIFICATION_CONCURRENCY = 1;
    const DEFAULT_COMPARISON_CONCURRENCY = 2;
    const JOB_OUTPUT_LIMIT = 1_000;
    const USER_RECIPES_KEY = 'tongues.speech.user-recipes.v1';
    const CATALOG_PAGE_SIZE = 32;
    const CATALOG_CACHE_NAME = 'tongues.speech.catalog.v4';
    const CATALOG_CACHE_TTL_MS = 5 * 60 * 1000;
    const catalogPageCache = new Map();
    const LIVE_CONVERSATION_SCHEMA_VERSION = 1;
    const LIVE_TERMINAL_MESSAGE_STATES = new Set(['completed', 'stopped', 'failed']);
    const SPEECH_SAMPLES = [
        ['ar', 'ara', 'Arabic', 'العربية', ['مرحبًا بالعالم. هذا مثال قصير لتحويل النص إلى كلام.']],
        ['am', 'amh', 'Amharic', 'አማርኛ', ['ሰላም ለዓለም። ይህ አጭር የንግግር ምሳሌ ነው።']],
        ['az', 'aze', 'Azerbaijani', 'Azərbaycanca', ['Salam dünya. Bu, nitq sintezi üçün qısa bir nümunədir.']],
        ['bn', 'ben', 'Bengali', 'বাংলা', ['হ্যালো পৃথিবী। এটি কথ্য ভাষার একটি ছোট নমুনা।']],
        ['bg', 'bul', 'Bulgarian', 'Български', ['Здравей, свят. Това е кратък пример за синтез на реч.']],
        ['ca', 'cat', 'Catalan', 'Català', ['Hola, món. Aquest és un exemple breu de síntesi de veu.']],
        ['zh', 'cmn', 'Chinese (Mandarin)', '中文（普通话）', ['你好，世界。这是一段简短的语音合成示例。', '今天天气很好，我们一起去公园散步吧。']],
        ['hr', 'hrv', 'Croatian', 'Hrvatski', ['Pozdrav, svijete. Ovo je kratak primjer sinteze govora.']],
        ['cs', 'ces', 'Czech', 'Čeština', ['Ahoj světe. Toto je krátká ukázka syntézy řeči.']],
        ['da', 'dan', 'Danish', 'Dansk', ['Hej verden. Dette er et kort eksempel på talesyntese.']],
        ['nl', 'nld', 'Dutch', 'Nederlands', ['Hallo wereld. Dit is een kort voorbeeld van spraaksynthese.']],
        ['en', 'eng', 'English', 'English', [
            'All human beings are born free and equal in dignity and rights.',
            'The quick brown fox jumps over the lazy dog.',
            'Welcome to Tongues. Choose a voice, then listen to this sample.',
        ]],
        ['et', 'est', 'Estonian', 'Eesti', ['Tere, maailm. See on lühike kõnesünteesi näide.']],
        ['fi', 'fin', 'Finnish', 'Suomi', ['Hei maailma. Tämä on lyhyt esimerkki puhesynteesistä.']],
        ['fr', 'fra', 'French', 'Français', ['Bonjour le monde. Voici un court exemple de synthèse vocale.', 'Portez ce vieux whisky au juge blond qui fume.']],
        ['ka', 'kat', 'Georgian', 'ქართული', ['გამარჯობა, მსოფლიო. ეს მეტყველების სინთეზის მოკლე მაგალითია.']],
        ['de', 'deu', 'German', 'Deutsch', ['Hallo Welt. Dies ist ein kurzes Beispiel für Sprachsynthese.', 'Franz jagt im komplett verwahrlosten Taxi quer durch Bayern.']],
        ['el', 'ell', 'Greek', 'Ελληνικά', ['Γεια σου, κόσμε. Αυτό είναι ένα σύντομο παράδειγμα σύνθεσης ομιλίας.']],
        ['gu', 'guj', 'Gujarati', 'ગુજરાતી', ['નમસ્તે દુનિયા. આ વાણી સંશ્લેષણનું ટૂંકું ઉદાહરણ છે.']],
        ['he', 'heb', 'Hebrew', 'עברית', ['שלום עולם. זוהי דוגמה קצרה לסינתזת דיבור.']],
        ['hi', 'hin', 'Hindi', 'हिन्दी', ['नमस्ते दुनिया। यह वाक् संश्लेषण का एक छोटा उदाहरण है।']],
        ['hu', 'hun', 'Hungarian', 'Magyar', ['Helló, világ! Ez egy rövid példa a beszédszintézisre.']],
        ['is', 'isl', 'Icelandic', 'Íslenska', ['Halló heimur. Þetta er stutt dæmi um talgervingu.']],
        ['id', 'ind', 'Indonesian', 'Bahasa Indonesia', ['Halo dunia. Ini adalah contoh singkat sintesis ucapan.']],
        ['ga', 'gle', 'Irish', 'Gaeilge', ['Dia duit, a dhomhain. Seo sampla gairid de shintéis cainte.']],
        ['it', 'ita', 'Italian', 'Italiano', ['Ciao mondo. Questo è un breve esempio di sintesi vocale.', 'Quel vituperabile xenofobo zelante assaggia il whisky ed esclama.']],
        ['ja', 'jpn', 'Japanese', '日本語', ['こんにちは、世界。これは音声合成の短い例です。', '今日はいい天気ですね。一緒に散歩しましょう。']],
        ['kn', 'kan', 'Kannada', 'ಕನ್ನಡ', ['ನಮಸ್ಕಾರ ಜಗತ್ತು. ಇದು ಭಾಷಣ ಸಂಶ್ಲೇಷಣೆಯ ಒಂದು ಚಿಕ್ಕ ಉದಾಹರಣೆ.']],
        ['ko', 'kor', 'Korean', '한국어', ['안녕하세요, 세상. 이것은 음성 합성의 짧은 예입니다.']],
        ['lv', 'lav', 'Latvian', 'Latviešu', ['Sveika, pasaule. Šis ir īss runas sintēzes piemērs.']],
        ['lt', 'lit', 'Lithuanian', 'Lietuvių', ['Labas, pasauli. Tai trumpas kalbos sintezės pavyzdys.']],
        ['ms', 'msa', 'Malay', 'Bahasa Melayu', ['Hai dunia. Ini ialah contoh ringkas sintesis pertuturan.']],
        ['ml', 'mal', 'Malayalam', 'മലയാളം', ['നമസ്കാരം ലോകമേ. ഇത് സംഭാഷണ സംശ്ലേഷണത്തിന്റെ ഒരു ചെറിയ ഉദാഹരണമാണ്.']],
        ['mr', 'mar', 'Marathi', 'मराठी', ['नमस्कार जग. हे वाक् संश्लेषणाचे एक छोटे उदाहरण आहे.']],
        ['ne', 'nep', 'Nepali', 'नेपाली', ['नमस्ते संसार। यो वाक् संश्लेषणको छोटो उदाहरण हो।']],
        ['no', 'nor', 'Norwegian', 'Norsk', ['Hei, verden. Dette er et kort eksempel på talesyntese.']],
        ['fa', 'fas', 'Persian', 'فارسی', ['سلام دنیا. این یک نمونه کوتاه از تبدیل متن به گفتار است.']],
        ['pl', 'pol', 'Polish', 'Polski', ['Witaj, świecie. To jest krótki przykład syntezy mowy.']],
        ['pt', 'por', 'Portuguese', 'Português', ['Olá, mundo. Este é um breve exemplo de síntese de fala.', 'A rápida raposa marrom salta sobre o cão preguiçoso.']],
        ['pa', 'pan', 'Punjabi', 'ਪੰਜਾਬੀ', ['ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਦੁਨੀਆ। ਇਹ ਬੋਲੀ ਸੰਸਲੇਸ਼ਣ ਦੀ ਇੱਕ ਛੋਟੀ ਮਿਸਾਲ ਹੈ।']],
        ['ro', 'ron', 'Romanian', 'Română', ['Salut, lume. Acesta este un scurt exemplu de sinteză vocală.']],
        ['ru', 'rus', 'Russian', 'Русский', ['Здравствуй, мир. Это короткий пример синтеза речи.', 'Съешь ещё этих мягких французских булок, да выпей чаю.']],
        ['sr', 'srp', 'Serbian', 'Српски', ['Здраво, свете. Ово је кратак пример синтезе говора.']],
        ['sk', 'slk', 'Slovak', 'Slovenčina', ['Ahoj, svet. Toto je krátka ukážka syntézy reči.']],
        ['sl', 'slv', 'Slovenian', 'Slovenščina', ['Pozdravljen, svet. To je kratek primer sinteze govora.']],
        ['es', 'spa', 'Spanish', 'Español', ['Hola, mundo. Este es un breve ejemplo de síntesis de voz.', 'El veloz murciélago hindú comía feliz cardillo y kiwi.']],
        ['sw', 'swa', 'Swahili', 'Kiswahili', ['Habari, dunia. Huu ni mfano mfupi wa usanisi wa sauti.']],
        ['sv', 'swe', 'Swedish', 'Svenska', ['Hej världen. Det här är ett kort exempel på talsyntes.']],
        ['ta', 'tam', 'Tamil', 'தமிழ்', ['வணக்கம் உலகமே. இது பேச்சுத் தொகுப்பின் ஒரு சிறிய எடுத்துக்காட்டு.']],
        ['te', 'tel', 'Telugu', 'తెలుగు', ['నమస్కారం ప్రపంచం. ఇది ప్రసంగ సంశ్లేషణకు ఒక చిన్న ఉదాహరణ.']],
        ['th', 'tha', 'Thai', 'ไทย', ['สวัสดีชาวโลก นี่คือตัวอย่างสั้น ๆ ของการสังเคราะห์เสียงพูด']],
        ['tr', 'tur', 'Turkish', 'Türkçe', ['Merhaba dünya. Bu, konuşma sentezinin kısa bir örneğidir.']],
        ['uk', 'ukr', 'Ukrainian', 'Українська', ['Привіт, світе. Це короткий приклад синтезу мовлення.']],
        ['ur', 'urd', 'Urdu', 'اردو', ['سلام دنیا۔ یہ تقریری ترکیب کی ایک مختصر مثال ہے۔']],
        ['vi', 'vie', 'Vietnamese', 'Tiếng Việt', ['Xin chào thế giới. Đây là một ví dụ ngắn về tổng hợp giọng nói.']],
        ['cy', 'cym', 'Welsh', 'Cymraeg', ['Helo, fyd. Dyma enghraifft fer o synthesis lleferydd.']],
    ].map(([code, iso3, name, nativeName, texts]) => ({
        code, iso3, name, nativeName, texts,
    }));
    const WORKFLOWS = {
        speak: {
            path: '/speech',
            label: 'Speak',
            summary: 'Generate and export speech from a complete, verified recipe.',
        },
        live: {
            path: '/speech/live',
            label: 'Live',
            summary: 'Hear a streamed Ollama response while its later text is still being generated.',
        },
        compose: {
            path: '/speech/compose',
            label: 'Compose',
            summary: 'Inspect and assemble contract-valid speech pipelines.',
        },
        compare: {
            path: '/speech/compare',
            label: 'Compare',
            summary: 'Listen to several complete recipes using one shared prompt.',
        },
        catalog: {
            path: '/speech/catalog',
            label: 'Catalog',
            summary: 'Find ready voices, installable model families, and developer components.',
        },
        operate: {
            path: '/speech/operate',
            label: 'Operate',
            summary: 'Inspect runtime state, verification, jobs, failures, and evidence.',
        },
    };

    const pathKey = (path) => `${path.backend}::${path.model}`;
    function createLiveConversation(snapshot = {}) {
        const messages = structuredClone(snapshot.messages || []);
        if (
            snapshot.schema_version != null
            && snapshot.schema_version !== LIVE_CONVERSATION_SCHEMA_VERSION
        ) {
            throw new Error(
                `Live conversation schema ${snapshot.schema_version} is unsupported.`,
            );
        }
        const ids = new Set();
        for (const message of messages) {
            if (
                !message.id
                || !message.turn_id
                || !['user', 'assistant'].includes(message.role)
                || typeof message.content !== 'string'
                || !ids.add(message.id)
            ) throw new Error('Live conversation contains an invalid or duplicate message.');
            if (
                message.role === 'assistant'
                && !['pending', 'streaming', ...LIVE_TERMINAL_MESSAGE_STATES]
                    .includes(message.status)
            ) throw new Error(`Assistant message ${message.id} has an invalid state.`);
        }
        return { schema_version: LIVE_CONVERSATION_SCHEMA_VERSION, messages };
    }
    function beginLiveConversationTurn(conversation, turnId, userText) {
        if (conversation.messages.some((message) => message.turn_id === turnId)) {
            throw new Error(`Live turn ${turnId} already exists.`);
        }
        conversation.messages.push(
            {
                id: `${turnId}:user`,
                turn_id: turnId,
                role: 'user',
                content: userText,
                status: 'completed',
            },
            {
                id: `${turnId}:assistant`,
                turn_id: turnId,
                role: 'assistant',
                content: '',
                status: 'pending',
            },
        );
        return `${turnId}:assistant`;
    }
    function updateLiveAssistant(conversation, turnId, content) {
        const message = conversation.messages.find((candidate) => (
            candidate.id === `${turnId}:assistant`
        ));
        if (!message || LIVE_TERMINAL_MESSAGE_STATES.has(message.status)) return false;
        message.content = content;
        message.status = 'streaming';
        return true;
    }
    function finishLiveAssistant(conversation, turnId, status) {
        if (!LIVE_TERMINAL_MESSAGE_STATES.has(status)) {
            throw new Error(`Invalid terminal assistant state ${status}.`);
        }
        const message = conversation.messages.find((candidate) => (
            candidate.id === `${turnId}:assistant`
        ));
        if (!message || LIVE_TERMINAL_MESSAGE_STATES.has(message.status)) return false;
        message.status = status === 'completed' && !message.content ? 'failed' : status;
        return true;
    }
    function liveProviderMessages(conversation) {
        return conversation.messages
            .filter((message) => (
                message.role === 'user'
                || (message.role === 'assistant' && message.status === 'completed')
            ))
            .map(({ role, content }) => ({ role, content }));
    }
    const compositionGenerator = (composition) => (
        composition?.pipeline?.end_to_end || composition?.pipeline?.acoustic_model || ''
    );
    const selectedComposition = () => (
        state.discovery?.compositions?.find((composition) => composition.id === state.pathKey)
    );
    const selectedPath = () => {
        const composition = selectedComposition();
        if (!composition) {
            return state.discovery?.paths?.find((path) => pathKey(path) === state.pathKey);
        }
        const legacy = state.discovery?.paths?.find((path) => (
            path.backend === composition.backend && path.model === composition.model
        )) || {};
        return {
            ...legacy,
            ...(composition.capabilities || {}),
            ...composition,
            id: composition.model,
            complete: true,
            acoustic_model: composition.pipeline.acoustic_model,
            vocoder: composition.pipeline.vocoder,
            voice_model: composition.pipeline.end_to_end,
        };
    };
    const listedValues = (capability) => (
        capability?.support === 'listed' ? (capability.values || []) : []
    );
    const normalizedLanguageCode = (value) => String(value || '')
        .trim()
        .toLowerCase()
        .replaceAll('_', '-')
        .split('-')[0];
    const SAMPLE_LANGUAGE_ALIASES = {
        az: ['azj'],
        fa: ['pes'],
        ms: ['zsm'],
        no: ['nob', 'nno'],
        zh: ['zho'],
    };
    const arrayValues = (value) => (Array.isArray(value) ? value : []);
    function pathLanguageCodes(path) {
        const values = [
            ...arrayValues(path?.linguistic_coverage?.languages),
            ...arrayValues(path?.languages?.values),
            ...arrayValues(path?.catalog).flatMap((entry) => arrayValues(entry.languages)),
            ...varietiesForPath(path).flatMap((variety) => [variety.id, variety.language]),
        ];
        return new Set(values.map(normalizedLanguageCode).filter(Boolean));
    }
    function pathSupportsSampleLanguage(path, language) {
        if (!path || !language) return true;
        const sample = typeof language === 'string'
            ? SPEECH_SAMPLES.find((candidate) => candidate.code === language || candidate.iso3 === language)
            : language;
        if (!sample) return false;
        const codes = pathLanguageCodes(path);
        return [sample.code, sample.iso3, ...(SAMPLE_LANGUAGE_ALIASES[sample.code] || [])]
            .some((code) => codes.has(code));
    }
    function sampleLanguagesForDiscovery(discovery) {
        const paths = availableCompositions(discovery).map(
            (composition) => pathForComposition(composition, discovery),
        ).filter(Boolean);
        return SPEECH_SAMPLES.filter((language) => (
            paths.some((path) => pathSupportsSampleLanguage(path, language))
        ));
    }
    const comparisonSpeaker = (path, recipe = {}) => {
        if (recipe.speaker) return recipe.speaker;
        if (!path?.speakers?.required) return null;
        const speakers = listedValues(path.speakers.values);
        return speakers.find((speaker) => speaker.id === 'p225')?.id
            || speakers[0]?.id
            || null;
    };
    async function mapWithConcurrency(items, concurrency, callback) {
        const limit = Math.max(1, Math.min(items.length, Math.floor(concurrency) || 1));
        let next = 0;
        const workers = Array.from({ length: limit }, async () => {
            while (next < items.length) {
                const index = next;
                next += 1;
                await callback(items[index], index);
            }
        });
        await Promise.all(workers);
    }
    const availablePaths = (discovery, includeTest = false) => (
        (discovery?.paths || []).filter((path) => includeTest || path.backend !== 'mock')
    );
    const selectInitialPath = (discovery) => {
        const paths = availablePaths(discovery);
        return paths.find((path) => path.selected && path.runnable)
            || paths.find((path) => path.runnable)
            || paths.find((path) => path.selected)
            || paths[0]
            || null;
    };
    const availableCompositions = (discovery, includeTest = false) => (
        (discovery?.compositions || []).filter((composition) => (
            includeTest || composition.backend !== 'mock'
        ))
    );
    const referenceControlDefault = (field, defaults = {}) => ({
        source_audio: defaults.style,
        target_audio: defaults.voice,
        voice_sample: defaults.voice,
        style_sample: defaults.style,
    }[field]);
    const controlDefault = (control, defaults = {}) => (
        control?.default ?? referenceControlDefault(control?.field, defaults)
    );
    const defaultControlValues = (path, defaults = {}) => Object.fromEntries(
        (path?.controls || []).flatMap((control) => {
            const value = controlDefault(control, defaults);
            return value == null || value === '' ? [] : [[control.field, value]];
        }),
    );
    const selectInitialComposition = (discovery) => {
        const compositions = availableCompositions(discovery);
        return compositions.find((composition) => composition.selected && composition.runnable)
            || compositions.find((composition) => composition.runnable)
            || compositions.find((composition) => composition.selected)
            || compositions[0]
            || null;
    };
    const compatibilityFor = (discovery, from, to) => (
        (discovery?.compatibility || []).find((edge) => (
            edge.from_component_id === from && edge.to_component_id === to
        ))
    );
    const varietiesForPath = (path) => listedValues(path?.varieties);
    const controlsForPath = (path, group) => (
        (path?.controls || []).filter((control) => control.group === group)
    );
    const pendingVerificationIds = (discovery) => (
        [...new Set((discovery?.verification_ids || []).filter(Boolean))]
    );
    const mergeUnique = (current, incoming, key) => {
        const merged = [...(current || [])];
        const seen = new Set(merged.map(key));
        for (const item of incoming || []) {
            const itemKey = key(item);
            if (seen.has(itemKey)) continue;
            seen.add(itemKey);
            merged.push(item);
        }
        return merged;
    };
    const mergeUpdated = (current, incoming, key) => {
        const incomingByKey = new Map((incoming || []).map((item) => [key(item), item]));
        const merged = (current || []).map((item) => incomingByKey.get(key(item)) || item);
        const seen = new Set((current || []).map(key));
        for (const item of incoming || []) {
            const itemKey = key(item);
            if (seen.has(itemKey)) continue;
            seen.add(itemKey);
            merged.push(item);
        }
        return merged;
    };
    const mergeDiscovery = (current, incoming) => {
        if (!current || !incoming?.page || incoming.page.cursor === 0) return incoming;
        return {
            ...current,
            ...incoming,
            paths: mergeUnique(current.paths, incoming.paths, pathKey),
            components: mergeUnique(current.components, incoming.components, (item) => item.id),
            compositions: mergeUnique(
                current.compositions,
                incoming.compositions,
                (item) => item.id,
            ),
            compatibility: mergeUnique(
                current.compatibility,
                incoming.compatibility,
                (item) => `${item.from_component_id}\0${item.to_component_id}`,
            ),
            presets: mergeUnique(current.presets, incoming.presets, (item) => item.id),
            verification_ids: pendingVerificationIds({
                verification_ids: [
                    ...(current.verification_ids || []),
                    ...(incoming.verification_ids || []),
                ],
            }),
            error: incoming.error || current.error,
        };
    };
    const mergeInventoryDiscovery = (current, incoming) => {
        if (!current) return incoming;
        if (!incoming) return current;
        return {
            ...current,
            schema_version: incoming.schema_version || current.schema_version,
            paths: mergeUpdated(current.paths, incoming.paths, pathKey),
            components: mergeUpdated(current.components, incoming.components, (item) => item.id),
            compositions: mergeUpdated(
                current.compositions,
                incoming.compositions,
                (item) => item.id,
            ),
            compatibility: mergeUpdated(
                current.compatibility,
                incoming.compatibility,
                (item) => `${item.from_component_id}\0${item.to_component_id}`,
            ),
            presets: mergeUpdated(current.presets, incoming.presets, (item) => item.id),
            verification_ids: pendingVerificationIds({
                verification_ids: [
                    ...(current.verification_ids || []),
                    ...(incoming.verification_ids || []),
                ],
            }),
            error: incoming.error || current.error,
        };
    };
    const mergeSelectedResultIntoDiscovery = (current, discovery, composition) => {
        if (!composition) return current;
        const path = (discovery?.paths || []).find((candidate) => (
            candidate.backend === composition.backend && candidate.model === composition.model
        ));
        const componentIds = new Set([
            composition.pipeline?.input,
            composition.pipeline?.projector,
            composition.pipeline?.acoustic_model,
            composition.pipeline?.vocoder,
            composition.pipeline?.end_to_end,
            composition.pipeline?.output,
            ...(composition.pipeline?.conditioners || []),
        ].filter(Boolean));
        const selectedCatalogIds = new Set((path?.catalog || []).map((entry) => entry.id));
        const selected = {
            ...discovery,
            paths: path ? [path] : [],
            components: (discovery?.components || []).filter(
                (component) => componentIds.has(component.id),
            ),
            compositions: [composition],
            compatibility: (discovery?.compatibility || []).filter((edge) => (
                componentIds.has(edge.from_component_id) && componentIds.has(edge.to_component_id)
            )),
            presets: (discovery?.presets || []).filter(
                (preset) => preset.composition_id === composition.id,
            ),
            verification_ids: (discovery?.verification_ids || []).filter(
                (id) => selectedCatalogIds.has(id),
            ),
        };
        return mergeInventoryDiscovery(current, selected);
    };
    const preservesVerificationProgress = (current, updated) => {
        if (!current) return true;
        const currentPending = new Set(pendingVerificationIds(current));
        return pendingVerificationIds(updated).every((id) => currentPending.has(id));
    };

    function parseNumberArray(source, positiveIntegers = false) {
        if (!String(source || '').trim()) return null;
        const parts = String(source).split(',').map((part) => part.trim());
        const values = parts.map(Number);
        const valid = parts.every(Boolean) && (positiveIntegers
            ? values.every((value) => Number.isSafeInteger(value) && value > 0)
            : values.every(Number.isFinite));
        if (!valid) {
            throw new Error(positiveIntegers
                ? 'Use comma-separated positive whole numbers.'
                : 'Use comma-separated finite numbers.');
        }
        return values;
    }

    function applyControlValues(path, values, payload) {
        const declared = new Set((path.controls || []).map((control) => control.field));
        for (const control of path.controls || []) {
            const value = values.get(control.field);
            if (value == null || value === '') continue;
            if (control.field === 'device') {
                if (value === 'cpu') payload.cpu = true;
                if (String(value).startsWith('cuda:')) {
                    payload.cuda_device = Number(String(value).split(':')[1]);
                }
                continue;
            }
            if (control.field === 'blend_mode') continue;
            if (
                ['style_alpha', 'style_beta'].includes(control.field)
                && values.get('blend_mode') !== 'raw'
            ) continue;
            if (
                ['speaker_reference_strength', 'style_reference_strength'].includes(control.field)
                && values.get('blend_mode') === 'raw'
            ) continue;
            if (control.kind === 'number') {
                const numeric = Number(value);
                if (!Number.isFinite(numeric)) throw new Error(`${control.label} must be a number.`);
                payload[control.field] = numeric;
            } else if (control.kind === 'number_array') {
                const parsed = parseNumberArray(value);
                if (parsed) payload[control.field] = parsed;
            } else if (control.kind === 'positive_integer_array') {
                const parsed = parseNumberArray(value, true);
                if (parsed) payload[control.field] = parsed;
            } else if (control.kind === 'boolean') {
                payload[control.field] = Boolean(value);
            } else {
                payload[control.field] = value;
            }
        }
        return declared;
    }

    function buildPayload(path, values, context = {}) {
        if (!path?.complete || !path?.runnable) {
            throw new Error(path?.unavailable_reason || 'Select a complete, ready synthesis path.');
        }
        const payload = {
            text: String(context.text || ''),
            variety: context.variety || varietiesForPath(path)[0]?.id || null,
        };
        if (path.pipeline) payload.pipeline = path.pipeline;
        else {
            payload.backend = path.backend;
            payload.model = path.model;
        }
        if (!payload.text.trim() && path.family !== 'voice_conversion') {
            throw new Error('Enter text to synthesize.');
        }
        if (path.speakers?.required && !context.speaker) {
            throw new Error('This model requires a speaker.');
        }
        if (context.speaker) payload.speaker = context.speaker;

        const declared = applyControlValues(path, values, payload);
        if (payload.emotion && context.emotionVector) {
            payload.emotion_vector = context.emotionVector;
        }
        for (const field of Object.keys(payload)) {
            if (
                !['text', 'pipeline', 'backend', 'model', 'variety', 'speaker', 'emotion_vector'].includes(field)
                && !declared.has(field)
                && !['cpu', 'cuda_device'].includes(field)
            ) {
                delete payload[field];
            }
        }
        return payload;
    }

    function studioShell() {
        const workflowTabs = Object.values(WORKFLOWS).map((workflow) => `
            <a href="${workflow.path}" data-studio-route="${workflow.path}"
                role="tab" aria-selected="false">${workflow.label}</a>
        `).join('');
        return `
            <main class="glass-panel speech-studio">
                <nav class="studio-workflows" role="tablist" aria-label="Speech Studio workflows">
                    ${workflowTabs}
                    <a href="/speech-dataflow.html" aria-label="Open speech dataflow editor">Dataflow</a>
                    <a href="/wavedeck.html" aria-label="Open speech timeline editor">Timeline</a>
                </nav>

                <section id="speech-workflow-speak" class="studio-workflow" data-workflow="speak"
                    aria-labelledby="speak-heading">
                    <div class="workflow-heading">
                        <div>
                            <p class="eyebrow">Quick synthesis</p>
                            <h2 id="speak-heading">Turn text into speech</h2>
                            <p>Choose a complete recipe, adjust only the controls it supports, and generate audio.</p>
                        </div>
                    </div>
                    <form id="synth-form" novalidate>
                        <div id="speech-error" class="inline-error hidden" role="alert" tabindex="-1"></div>
                        <div class="sample-picker" aria-labelledby="speech-samples-heading">
                            <div>
                                <p class="eyebrow" id="speech-samples-heading">Sample library</p>
                                <strong>Try a language this recipe can speak</strong>
                            </div>
                            <div class="form-group">
                                <label for="speech-language">Language</label>
                                <select id="speech-language"></select>
                                <small id="speech-language-detail">Loading compatible languages…</small>
                            </div>
                            <div class="form-group">
                                <label for="speech-sample">Sample text</label>
                                <select id="speech-sample"></select>
                                <small>Selecting a sample fills the text below.</small>
                            </div>
                        </div>
                        <div class="form-group">
                            <label for="text">Text to speak</label>
                            <textarea id="text" required>All human beings are born free and equal in dignity and rights. They are endowed with reason and conscience and should act towards one another in a spirit of brotherhood.</textarea>
                        </div>
                        <div class="speak-choice-grid">
                            <div class="form-group">
                                <label for="speech-voice">Voice or language</label>
                                <input id="speech-voice" type="search" list="speech-voice-options"
                                    autocomplete="off" placeholder="Search a voice, language, or model">
                                <datalist id="speech-voice-options"></datalist>
                                <small>Searchable complete speech paths only; technical components are in Compose.</small>
                            </div>
                            <div class="form-group">
                                <label for="speech-preset">Recipe</label>
                                <select id="speech-preset"></select>
                                <small id="synthesis-path-detail">Loading recipes…</small>
                            </div>
                        </div>
                        <div class="controls-grid">
                            <div class="form-group" id="variety-control">
                                <label for="variety">Language or variety</label>
                                <select id="variety"></select>
                                <div id="fixed-variety" class="fixed-value hidden"></div>
                                <small id="variety-detail"></small>
                            </div>
                            <div class="form-group hidden" id="speaker-control">
                                <label for="speaker">Voice or speaker</label>
                                <input id="speaker" type="search" list="speaker-options"
                                    autocomplete="off" placeholder="Search named speakers">
                                <datalist id="speaker-options"></datalist>
                                <small id="speaker-detail"></small>
                            </div>
                        </div>
                        <section id="speech-recipe-summary" class="recipe-summary" aria-live="polite"></section>
                        <div id="speech-controls-basic" class="controls-grid"></div>
                        <details class="advanced-section">
                            <summary>Advanced synthesis controls</summary>
                            <div id="speech-controls-advanced" class="controls-grid"></div>
                        </details>
                        <details class="advanced-section">
                            <summary>Expert token controls</summary>
                            <p class="expert-warning">Token arrays are checked against the selected
                            model projection before synthesis.</p>
                            <div id="speech-controls-expert" class="controls-grid"></div>
                        </details>
                        <details id="speech-developer" class="advanced-section">
                            <summary>Developer and testing</summary>
                            <div id="speech-controls-developer" class="controls-grid"></div>
                            <button id="select-mock-path" type="button" class="secondary-button hidden">
                                Use deterministic test path
                            </button>
                        </details>
                        <div class="action-bar split-actions">
                            <button type="submit" id="submit-btn" disabled>
                                <span class="btn-text">Generate speech</span>
                                <div class="spinner"></div>
                            </button>
                            <button id="show-pipeline" type="button" class="secondary-button">Show pipeline</button>
                            <button id="add-current-to-compare" type="button" class="secondary-button">Add to Compare</button>
                        </div>
                        <div id="speech-submit-status" role="status" aria-live="polite"></div>
                    </form>
                    <section id="result-container" class="result-panel hidden"
                        aria-labelledby="speech-result-heading">
                        <div class="speech-section-heading">
                            <div>
                                <span id="speech-result-state" class="runtime-badge" data-state="ready">Completed</span>
                                <h2 id="speech-result-heading">Speech result</h2>
                            </div>
                            <a id="speech-download" class="secondary-button"
                                download="tongues-speech.wav">Download WAV</a>
                            <a id="speech-run-link" class="secondary-button hidden"
                                href="/speech">Permanent run link</a>
                        </div>
                        <audio id="audio-player" controls></audio>
                        <dl id="speech-result-metadata" class="metadata-grid"></dl>
                        <details id="speech-result-diagnostics" class="advanced-section hidden">
                            <summary>Additional details</summary>
                            <pre id="speech-diagnostics-output" class="source-preview"></pre>
                        </details>
                    </section>
                </section>

                <section id="speech-workflow-live" class="studio-workflow hidden"
                    data-workflow="live" aria-labelledby="live-heading">
                    <div class="workflow-heading">
                        <div>
                            <p class="eyebrow">Streaming conversation</p>
                            <h2 id="live-heading">Hear the answer while it is written</h2>
                            <p>The server streams generated text and synthesized phrase audio together; the browser keeps those audio segments on one Web Audio timeline.</p>
                        </div>
                        <span id="live-state" class="runtime-badge" data-state="ready">Ready</span>
                    </div>
                    <div id="live-error" class="inline-error hidden" role="alert" tabindex="-1"></div>
                    <section class="live-controls" aria-label="Live conversation controls">
                        <div class="form-group">
                            <label for="live-provider">Text provider</label>
                            <select id="live-provider"></select>
                            <small id="live-provider-detail">Checking Ollama…</small>
                        </div>
                        <div class="form-group">
                            <label for="live-model">LLM model</label>
                            <select id="live-model"></select>
                        </div>
                        <div class="form-group">
                            <label for="live-recipe">Speech recipe</label>
                            <select id="live-recipe"></select>
                            <small>The recipe supplies language, script, and normalization instructions.</small>
                        </div>
                        <div class="form-group">
                            <label for="live-instructions">Response instructions</label>
                            <input id="live-instructions" type="text"
                                placeholder="For example: Tell a vivid story in four paragraphs.">
                        </div>
                    </section>
                    <div class="live-frontiers" aria-label="Turn frontiers">
                        <div><span>Generated</span><strong id="live-generated-count">0</strong></div>
                        <div><span>Planned</span><strong id="live-planned-count">0</strong></div>
                        <div><span>Spoken</span><strong id="live-spoken-count">0</strong></div>
                    </div>
                    <section id="live-conversation" class="live-conversation"
                        aria-label="Conversation" aria-live="polite">
                        <p class="live-empty">Start a turn to watch generation, planning, and playback move independently.</p>
                    </section>
                    <section class="browser-mic-probe" aria-labelledby="conversation-mic-heading">
                        <h3 id="conversation-mic-heading">Live microphone recognition</h3>
                        <p id="live-asr-text" class="live-generating">No recognition yet.</p>
                        <button id="live-mic-start" type="button">Start microphone conversation</button>
                        <button id="live-mic-stop" type="button" class="secondary-button" disabled>Stop microphone</button>
                        <label><input id="live-barge-in" type="checkbox" checked> Committed external speech may interrupt playback</label>
                        <p>Microphone activity and unstable text never become user turns. Target-like echo is ignored; a committed external segment cancels and restarts at a safe turn boundary.</p>
                        <dl id="live-latencies" class="metadata-grid" aria-live="polite"></dl>
                    </section>
                    <form id="live-form" class="live-composer">
                        <label class="sr-only" for="live-message">Message</label>
                        <input id="live-message" type="text" required
                            placeholder="Ask for a story, explanation, dialogue, or translation.">
                        <div class="action-bar split-actions">
                            <button id="live-send" type="submit">Send</button>
                            <button id="live-stop" type="button" class="danger-button" disabled>Stop</button>
                            <button id="live-replay" type="button" class="secondary-button" disabled>Replay turn</button>
                            <a id="live-download" class="secondary-button hidden"
                                download="tongues-live-turn.wav">Download turn</a>
                            <a href="/studio/graphs/starter%3Alive_conversation">Open conversation graph</a>
                            <a id="live-session-link" class="hidden" href="/sessions/new/correct">Inspect durable session evidence</a>
                        </div>
                    </form>
                    <details class="advanced-section">
                        <summary>Turn journal and exact artifacts</summary>
                        <pre id="live-journal" class="source-preview">No turn events yet.</pre>
                    </details>
                </section>

                <section id="speech-workflow-compose" class="studio-workflow hidden"
                    data-workflow="compose" aria-labelledby="compose-heading">
                    <div class="workflow-heading">
                        <div>
                            <p class="eyebrow">Typed patch bay</p>
                            <h2 id="compose-heading">Compose a speech pipeline</h2>
                            <p>Each selectable connection is filtered by the runtime’s executable contract graph.</p>
                        </div>
                        <span id="compose-validity" class="runtime-badge" data-state="loading">Checking</span>
                    </div>
                    <div id="compose-error" class="inline-error hidden" role="alert" tabindex="-1"></div>
                    <div class="pipeline-toolbar">
                        <div class="form-group">
                            <label for="compose-recipe-name">Recipe name</label>
                            <input id="compose-recipe-name" type="text" placeholder="My speech recipe">
                        </div>
                        <div class="compose-recipe-actions">
                            <button id="duplicate-recipe" type="button" class="secondary-button">Duplicate</button>
                            <button id="save-recipe" type="button">Save recipe</button>
                            <button id="restore-recipe" type="button" class="secondary-button">Restore</button>
                            <button id="delete-recipe" type="button" class="secondary-button" disabled>Delete saved copy</button>
                        </div>
                    </div>
                    <section class="pipeline-workbench" aria-label="Speech synthesis pipeline">
                        <div class="pipeline-stage" role="group" tabindex="0" data-stage="input">
                            <span class="pipeline-stage-label">Input</span>
                            <strong>Text</strong>
                            <small>Tongues linguistic plan</small>
                        </div>
                        <span class="pipeline-connector" aria-hidden="true">→</span>
                        <div class="pipeline-stage" role="group" tabindex="0" data-stage="projector">
                            <label class="pipeline-stage-label" for="pipeline-projector">Projector</label>
                            <select id="pipeline-projector"></select>
                            <small id="pipeline-projector-detail"></small>
                        </div>
                        <span class="pipeline-connector" aria-hidden="true">→</span>
                        <div class="pipeline-generator-stack">
                            <div class="pipeline-stage pipeline-conditioning" role="group" tabindex="0" data-stage="conditioner">
                                <span class="pipeline-stage-label">Conditioning</span>
                                <strong id="pipeline-conditioning-name">Model controls</strong>
                                <small id="pipeline-conditioning-detail">Speaker, language, style, and prosody</small>
                            </div>
                            <span class="pipeline-branch" aria-hidden="true">↓</span>
                            <div class="pipeline-stage" role="group" tabindex="0" data-stage="generator">
                                <label class="pipeline-stage-label" for="pipeline-generator">Acoustic model</label>
                                <select id="pipeline-generator"></select>
                                <small id="pipeline-generator-detail"></small>
                            </div>
                        </div>
                        <span class="pipeline-connector" aria-hidden="true">→</span>
                        <div class="pipeline-stage" id="pipeline-vocoder-stage" role="group" tabindex="0" data-stage="vocoder">
                            <label class="pipeline-stage-label" for="pipeline-vocoder">Vocoder</label>
                            <select id="pipeline-vocoder"></select>
                            <small id="pipeline-vocoder-detail"></small>
                        </div>
                        <span class="pipeline-connector" aria-hidden="true">→</span>
                        <div class="pipeline-stage" role="group" tabindex="0" data-stage="output">
                            <span class="pipeline-stage-label">Output</span>
                            <strong>WAV audio</strong>
                            <small>Playback and download</small>
                        </div>
                    </section>
                    <div class="compose-detail-grid">
                        <section id="pipeline-stage-inspector" class="stage-inspector" aria-live="polite"></section>
                        <details class="model-card">
                            <summary>Recipe provenance, readiness, and license</summary>
                            <section id="speech-model-card" aria-live="polite"></section>
                        </details>
                    </div>
                    <div class="form-group">
                        <label for="compose-cli">Exact CLI representation</label>
                        <div class="copy-row">
                            <code id="compose-cli"></code>
                            <button id="copy-compose-cli" type="button" class="secondary-button">Copy</button>
                        </div>
                    </div>
                    <div class="action-bar split-actions">
                        <button id="test-pipeline" type="button">Test pipeline</button>
                        <button id="open-pipeline-in-speak" type="button" class="secondary-button">Open in Speak</button>
                        <button id="add-pipeline-to-compare" type="button" class="secondary-button">Add to Compare</button>
                    </div>
                    <div id="compose-test-status" role="status" aria-live="polite"></div>
                    <audio id="compose-audio" class="hidden" controls></audio>
                </section>

                <section id="speech-workflow-compare" class="studio-workflow hidden"
                    data-workflow="compare" aria-labelledby="compare-heading">
                    <div class="workflow-heading">
                        <div>
                            <p class="eyebrow">Listening test</p>
                            <h2 id="compare-heading">Compare complete recipes</h2>
                            <p>Text-to-speech candidates receive the same prompt. Voice-conversion candidates use their configured source audio, and every result remains tied to its exact controls.</p>
                        </div>
                    </div>
                    <div id="compare-error" class="inline-error hidden" role="alert" tabindex="-1"></div>
                    <div class="form-group">
                        <label for="compare-text">Shared prompt</label>
                        <textarea id="compare-text" required></textarea>
                    </div>
                    <fieldset class="compare-fieldset">
                        <legend>Recipes to compare</legend>
                        <div id="compare-candidates" class="compare-candidates"></div>
                    </fieldset>
                    <label class="checkbox-row">
                        <input id="compare-blind" type="checkbox">
                        <span>Blind listening mode<small>Hide identities until results are revealed.</small></span>
                    </label>
                    <div class="action-bar split-actions">
                        <button id="generate-all" type="button">Generate all</button>
                        <button id="reveal-comparison" type="button" class="secondary-button">Reveal identities</button>
                        <button id="save-preferred" type="button" class="secondary-button" disabled>Save preferred recipe</button>
                    </div>
                    <div id="compare-status" role="status" aria-live="polite"></div>
                    <div id="compare-results" class="compare-results"></div>
                </section>

                <section id="speech-workflow-catalog" class="studio-workflow hidden"
                    data-workflow="catalog" aria-labelledby="model-catalog-heading">
                    <div class="workflow-heading">
                        <div>
                            <p class="eyebrow">Capability discovery</p>
                            <h2 id="model-catalog-heading">Catalog <span id="component-count"></span></h2>
                            <p>Start with usable speech paths. Expand model families only when you need them.</p>
                        </div>
                    </div>
                    <div class="catalog-view-tabs" role="tablist" aria-label="Catalog view">
                        <button type="button" role="tab" data-catalog-view="ready" aria-selected="true">Ready now</button>
                        <button type="button" role="tab" data-catalog-view="downloadable" aria-selected="false">Available to fetch</button>
                        <button type="button" role="tab" data-catalog-view="components" aria-selected="false">Developer components</button>
                    </div>
                    <div class="catalog-filters" aria-label="Catalog filters">
                        <div class="form-group">
                            <label for="catalog-search">Search models and languages</label>
                            <input id="catalog-search" type="search"
                                placeholder="Language, script, voice, architecture, or capability">
                        </div>
                        <div class="form-group">
                            <label for="catalog-family">Family</label>
                            <select id="catalog-family">
                                <option value="">All families</option>
                                <option value="mms">MMS VITS</option>
                                <option value="styletts2">StyleTTS2</option>
                                <option value="other">Other native families</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="catalog-license">License</label>
                            <select id="catalog-license">
                                <option value="">All licenses</option>
                                <option value="CC-BY-NC-4.0">CC-BY-NC-4.0</option>
                                <option value="MIT">MIT</option>
                                <option value="NOASSERTION">Metadata unavailable</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="catalog-capability">Capability</label>
                            <select id="catalog-capability">
                                <option value="">All capabilities</option>
                                <option value="speech">Text to speech</option>
                                <option value="voice_conversion">Voice conversion or cloning</option>
                                <option value="vocoder">Vocoder</option>
                                <option value="developer">Trainer or developer component</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="catalog-verification">Verification</label>
                            <select id="catalog-verification">
                                <option value="">Any verification state</option>
                                <option value="verified">Verified</option>
                                <option value="pending">Pending or changed</option>
                                <option value="failed">Failed or unavailable</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="catalog-device">Device support</label>
                            <select id="catalog-device">
                                <option value="">Any supported device</option>
                                <option value="cpu">CPU</option>
                                <option value="cuda">CUDA</option>
                            </select>
                        </div>
                    </div>
                    <p id="catalog-status" class="catalog-status" role="status" aria-live="polite"></p>
                    <div id="component-inventory" class="component-inventory"></div>
                </section>

                <section id="speech-workflow-operate" class="studio-workflow hidden"
                    data-workflow="operate" aria-labelledby="operate-heading">
                    <div class="workflow-heading">
                        <div>
                            <p class="eyebrow">Runtime and evidence</p>
                            <h2 id="operate-heading">Operate Speech Studio</h2>
                            <p>Runtime capabilities, verification, activity, and expert evidence share one state model.</p>
                        </div>
                    </div>
                    <section class="speech-runtime-panel" aria-labelledby="speech-runtime-heading">
                        <div class="speech-section-heading">
                            <div>
                                <span id="speech-runtime-state" class="runtime-badge" data-state="loading">Checking</span>
                                <h2 id="speech-runtime-heading">Resident runtime</h2>
                            </div>
                            <button id="reload-speech-runtime" type="button" class="secondary-button">Reload models</button>
                        </div>
                        <dl id="speech-runtime-grid" class="metadata-grid" aria-live="polite"></dl>
                        <div id="speech-runtime-errors" class="inline-error hidden" role="alert"></div>
                    </section>
                    <section class="operate-section" aria-labelledby="verification-heading">
                        <div class="speech-section-heading">
                            <div>
                                <h2 id="verification-heading">Verification</h2>
                                <p>Verify only installed models that changed since their last evidence check.</p>
                            </div>
                            <button id="verify-all-models" type="button">Verify changed models</button>
                        </div>
                        <p id="verification-status" role="status" aria-live="polite"></p>
                    </section>
                    <section class="operate-section" aria-labelledby="activity-heading">
                        <div class="speech-section-heading">
                            <div>
                                <h2 id="activity-heading">Jobs and activity</h2>
                                <p>Human-readable work first; commands, logs, and artifacts remain in details.</p>
                            </div>
                            <button id="refresh-operate-jobs" type="button" class="secondary-button">Refresh</button>
                        </div>
                        <div id="operate-jobs" class="operate-jobs" aria-live="polite"></div>
                    </section>
                    <details class="operate-labs">
                        <summary>Interpretation evidence</summary>
                        <section class="duplex-panel" aria-labelledby="duplex-heading">
                    <div class="speech-section-heading">
                        <div>
                            <h2 id="duplex-heading">Interpretation decisions and predictive timeline</h2>
                            <p class="page-doc">
                                The concise timeline stays visible first. Expand an evidence target
                                to inspect alternatives, sources, conflicts, confidence, revisions,
                                backend degradation, and output consequences.
                                Predicted text remains visible but never becomes playable client audio.
                            </p>
                        </div>
                        <div class="duplex-actions">
                            <button id="run-duplex-preview" type="button" class="secondary-button">Preview timeline</button>
                            <button id="replay-duplex-journal" type="button" class="secondary-button">Replay journal</button>
                        </div>
                    </div>
                    <div class="controls-grid">
                        <div class="form-group">
                            <label for="duplex-mock-acoustics">Mock acoustic chunks</label>
                            <textarea id="duplex-mock-acoustics"
                                placeholder="Optional transcript chunks, one line per chunk"></textarea>
                            <small>Leave blank to preview the prompt lines as direct text chunks.</small>
                        </div>
                        <div class="form-group">
                            <label for="duplex-journal-path">Saved journal path</label>
                            <input id="duplex-journal-path" type="text"
                                placeholder="target/duplex/oracle-chunks.journal.json">
                            <small>Replay a saved journal relative to the repository root.</small>
                        </div>
                    </div>
                    <div id="duplex-error" class="inline-error hidden" role="alert"></div>
                    <div id="duplex-summary" class="duplex-summary hidden"></div>
                    <section id="duplex-evidence" class="duplex-evidence hidden"
                        aria-labelledby="duplex-evidence-heading">
                        <h3 id="duplex-evidence-heading" tabindex="-1">Evidence chain</h3>
                        <div id="duplex-evidence-status" role="status" aria-live="polite"></div>
                        <div id="duplex-evidence-targets"></div>
                        <nav id="duplex-evidence-pages" aria-label="Interpretation evidence pages"></nav>
                    </section>
                    <ol id="duplex-timeline" class="duplex-timeline hidden"></ol>
                        </section>
                    </details>
                </section>
            </main>`;
    }

    const byId = (id) => document.getElementById(id);

    function workflowForPath(pathname) {
        const normalized = String(pathname || '/speech').replace(/\/+$/, '') || '/speech';
        return Object.entries(WORKFLOWS).find(([, workflow]) => workflow.path === normalized)?.[0]
            || 'speak';
    }

    function setWorkflow(pathname, { focus = false } = {}) {
        const workflow = WORKFLOWS[workflowForPath(pathname)] ? workflowForPath(pathname) : 'speak';
        state.workflow = workflow;
        if (typeof document === 'undefined') return workflow;
        if (workflow !== 'operate') stopOperateJobStreams();
        document.querySelectorAll('[data-workflow]').forEach((section) => {
            section.classList.toggle('hidden', section.dataset.workflow !== workflow);
        });
        document.querySelectorAll('[data-studio-route]').forEach((link) => {
            const selected = workflowForPath(link.dataset.studioRoute) === workflow;
            link.classList.toggle('active', selected);
            link.setAttribute('aria-selected', String(selected));
            if (selected) link.setAttribute('aria-current', 'page');
            else link.removeAttribute('aria-current');
        });
        if (workflow === 'compare') {
            if (byId('compare-text') && !byId('compare-text').value) {
                byId('compare-text').value = byId('text')?.value || '';
            }
            renderCompareCandidates();
        }
        if (workflow === 'operate') {
            refreshOperateJobs().catch((error) => {
                if (byId('operate-jobs')) {
                    byId('operate-jobs').textContent = `Activity unavailable: ${error.message}`;
                }
            });
        }
        if (focus) {
            const heading = byId(`speech-workflow-${workflow}`)?.querySelector('h2');
            heading?.setAttribute('tabindex', '-1');
            heading?.focus();
        }
        return workflow;
    }

    function navigateWorkflow(workflow) {
        const path = WORKFLOWS[workflow]?.path || WORKFLOWS.speak.path;
        const current = speechSelectionFromLocation();
        const query = new URLSearchParams();
        if (current.recipeId) query.set('recipe', current.recipeId);
        if (current.compositionId) query.set('composition', current.compositionId);
        if (current.modelId) query.set('model', current.modelId);
        if (current.runId) query.set('run', current.runId);
        const destination = `${path}${query.size ? `?${query}` : ''}`;
        if (
            browser?.history
            && `${browser.location?.pathname}${browser.location?.search}` !== destination
        ) {
            browser.history.pushState({}, '', destination);
            browser.dispatchEvent(new PopStateEvent('popstate'));
        } else {
            setWorkflow(path, { focus: true });
        }
    }

    function speechSelectionFromLocation(locationLike = browser?.location) {
        const params = new URLSearchParams(locationLike?.search || '');
        return {
            runId: params.get('run') || '',
            recipeId: params.get('recipe') || '',
            compositionId: params.get('composition') || '',
            modelId: params.get('model') || '',
        };
    }

    function syncSpeechSelectionUrl({ runId = '' } = {}) {
        if (!browser?.history || !browser.location) return '';
        const url = new URL(browser.location.href);
        const path = selectedPath();
        if (state.presetId) url.searchParams.set('recipe', state.presetId);
        else url.searchParams.delete('recipe');
        if (state.pathKey) url.searchParams.set('composition', state.pathKey);
        else url.searchParams.delete('composition');
        if (path?.model) url.searchParams.set('model', path.model);
        else url.searchParams.delete('model');
        if (runId) {
            state.speechRun = {
                id: runId,
                compositionId: state.pathKey,
                recipeId: state.presetId,
            };
        }
        const matchingRun = state.speechRun
            && state.speechRun.compositionId === state.pathKey
            && state.speechRun.recipeId === state.presetId
            ? state.speechRun.id
            : '';
        if (matchingRun) url.searchParams.set('run', matchingRun);
        else url.searchParams.delete('run');
        const destination = `${url.pathname}${url.search}`;
        browser.history.replaceState({ workflow: state.workflow }, '', destination);
        return destination;
    }

    function markSpeechSelectionChanged() {
        state.speechRun = null;
        byId('speech-run-link')?.classList.add('hidden');
        syncSpeechSelectionUrl();
    }

    function loadUserRecipes() {
        try {
            const parsed = JSON.parse(browser?.localStorage?.getItem(USER_RECIPES_KEY) || '[]');
            state.userRecipes = Array.isArray(parsed) ? parsed.filter((recipe) => (
                recipe && typeof recipe.id === 'string' && typeof recipe.compositionId === 'string'
            )) : [];
        } catch (_error) {
            state.userRecipes = [];
        }
        return state.userRecipes;
    }

    function persistUserRecipes() {
        browser?.localStorage?.setItem(USER_RECIPES_KEY, JSON.stringify(state.userRecipes));
    }

    function deleteUserRecipe(recipes, id) {
        const index = recipes.findIndex((recipe) => recipe.id === id);
        if (index < 0) return { recipes, deleted: null };
        return {
            recipes: recipes.filter((_recipe, candidateIndex) => candidateIndex !== index),
            deleted: recipes[index],
        };
    }

    function selectedVariety(path = selectedPath()) {
        const varieties = varietiesForPath(path);
        return varieties.length === 1 ? varieties[0].id : (byId('variety')?.value || varieties[0]?.id);
    }

    function selectedSpeaker(path = selectedPath()) {
        if (!path || byId('speaker-control')?.classList.contains('hidden')) return null;
        return byId('speaker')?.value.trim() || null;
    }

    function controlSnapshot(path = selectedPath()) {
        const controls = {};
        for (const control of path?.controls || []) {
            const value = state.values.get(control.field);
            if (value != null && value !== '') controls[control.field] = value;
        }
        return controls;
    }

    function recipeSnapshot(name = '', path = selectedPath()) {
        if (!path) return null;
        return {
            id: `user/${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            name: name.trim() || `${path.display_name} copy`,
            builtIn: false,
            compositionId: state.pathKey,
            pipeline: path.pipeline || null,
            backend: path.backend,
            model: path.model,
            variety: selectedVariety(path),
            speaker: selectedSpeaker(path),
            controls: controlSnapshot(path),
            updatedAt: new Date().toISOString(),
        };
    }

    function restoreRecipeValues(values, path, recipe) {
        for (const control of path?.controls || []) values.delete(control.field);
        for (const [field, value] of Object.entries(recipe?.controls || {})) {
            values.set(field, value);
        }
        const recipePathId = path.id || path.model;
        const varietyKey = `variety:${recipePathId}`;
        const speakerKey = `speaker:${recipePathId}`;
        if (recipe?.variety) values.set(varietyKey, recipe.variety);
        else values.delete(varietyKey);
        if (recipe?.speaker) values.set(speakerKey, recipe.speaker);
        else values.delete(speakerKey);
        return values;
    }

    function applyRecipe(recipe) {
        if (!recipe) return false;
        const composition = (state.discovery?.compositions || []).find(
            (candidate) => candidate.id === recipe.compositionId,
        );
        if (!composition) return false;
        const path = pathForComposition(composition);
        if (!path) return false;
        state.pathKey = composition.id;
        state.presetId = recipe.id;
        restoreRecipeValues(state.values, path, recipe);
        renderPathSelector();
        renderSelectedPath();
        if (byId('compose-recipe-name')) byId('compose-recipe-name').value = recipe.name;
        return true;
    }

    function shellQuote(value) {
        const source = String(value ?? '');
        return /^[A-Za-z0-9_./:+-]+$/.test(source)
            ? source
            : `'${source.replaceAll("'", "'\\''")}'`;
    }

    function cliRepresentation(path, values = state.values, context = {}) {
        if (!path) return '';
        const parts = ['tongues', 'speak'];
        const payload = {};
        applyControlValues(path, values, payload);
        if (payload.cpu) parts.push('--cpu');
        if (Number.isInteger(payload.cuda_device)) {
            parts.push('--cuda-device', String(payload.cuda_device));
        }
        const text = context.text ?? (
            typeof document !== 'undefined' ? byId('text')?.value : ''
        );
        if (text?.trim()) parts.push(shellQuote(text.trim()));
        parts.push('--backend', shellQuote(path.backend));
        if (path.model) parts.push('--model', shellQuote(path.model));
        if (path.cli_vocoder) parts.push('--vocoder', shellQuote(path.cli_vocoder));
        const variety = context.variety ?? (
            typeof document !== 'undefined'
                ? selectedVariety(path)
                : varietiesForPath(path)[0]?.id
        );
        if (variety) parts.push('--variety', shellQuote(variety));
        const speaker = context.speaker ?? (
            typeof document !== 'undefined' ? selectedSpeaker(path) : null
        );
        if (speaker) parts.push('--speaker', shellQuote(speaker));
        for (const control of path.controls || []) {
            const value = payload[control.field];
            if (value == null || value === '' || control.field === 'device') continue;
            const flag = `--${control.field.replaceAll('_', '-')}`;
            if (control.kind === 'boolean') {
                if (value) parts.push(flag);
            } else {
                parts.push(flag, shellQuote(Array.isArray(value) ? value.join(',') : value));
            }
        }
        return parts.join(' ');
    }

    function showError(message, target = byId('speech-error')) {
        target.textContent = message;
        target.classList.remove('hidden');
        if (target.id === 'speech-error') target.focus();
    }

    function clearError(target = byId('speech-error')) {
        target.textContent = '';
        target.classList.add('hidden');
    }

    function setStoredValue(field, value) {
        state.values.set(field, value);
    }

    function controlInput(control) {
        const wrapper = document.createElement('div');
        wrapper.className = 'form-group speech-control';
        wrapper.dataset.control = control.field;
        const label = document.createElement('label');
        label.htmlFor = `speech-control-${control.field}`;
        label.textContent = control.label;
        const id = label.htmlFor;
        let input;

        if (control.kind === 'select') {
            input = document.createElement('select');
            for (const choice of control.options || []) {
                const option = document.createElement('option');
                option.value = choice.value;
                option.textContent = choice.label;
                input.appendChild(option);
            }
        } else if (control.kind === 'boolean') {
            input = document.createElement('input');
            input.type = 'checkbox';
            wrapper.classList.add('control-checkbox');
        } else if (control.kind === 'reference_audio' || control.kind === 'emotion') {
            input = document.createElement('select');
            const empty = document.createElement('option');
            empty.value = '';
            empty.textContent = control.kind === 'emotion' ? 'None / neutral' : 'Default reference';
            input.appendChild(empty);
            const source = control.kind === 'emotion' ? state.emotions : state.samples;
            for (const item of source) {
                const option = document.createElement('option');
                option.value = item.id || item.name;
                option.textContent = item.label || item.name || item.id;
                input.appendChild(option);
            }
        } else {
            input = document.createElement('input');
            if (control.kind === 'number') {
                input.type = 'number';
                if (control.min != null) input.min = control.min;
                if (control.max != null) input.max = control.max;
                if (control.step != null) input.step = control.step;
            } else {
                input.type = 'text';
                input.placeholder = control.kind === 'positive_integer_array'
                    ? '4, 7, 3'
                    : '0.1, -0.2, 0.0';
            }
        }
        input.id = id;
        input.name = control.field;
        const stored = state.values.get(control.field);
        const initial = stored ?? controlDefault(control, state.sampleDefaults);
        const optionValues = input.options ? Array.from(input.options, (item) => item.value) : [];
        if (control.kind === 'boolean') {
            input.checked = Boolean(initial);
        } else if (initial != null && optionValues.includes(String(initial))) {
            input.value = String(initial);
        } else if (initial != null && control.kind !== 'select') {
            input.value = String(initial);
        } else if (control.kind === 'select' && input.options.length) {
            input.value = input.options[0].value;
        }
        setStoredValue(control.field, control.kind === 'boolean' ? input.checked : input.value);
        input.addEventListener('input', () => {
            setStoredValue(control.field, control.kind === 'boolean' ? input.checked : input.value);
            syncBlendControls();
            markSpeechSelectionChanged();
        });
        const help = document.createElement('small');
        help.textContent = [control.help, control.unit ? `Unit: ${control.unit}.` : '']
            .filter(Boolean).join(' ');
        wrapper.append(label, input, help);
        if (control.kind === 'reference_audio') {
            const preview = document.createElement('audio');
            preview.controls = true;
            preview.className = 'preview-audio empty';
            input.addEventListener('change', () => {
                const sample = state.samples.find((item) => item.id === input.value);
                if (sample) {
                    preview.src = sample.audio_url;
                    preview.classList.remove('empty');
                } else {
                    preview.removeAttribute('src');
                    preview.classList.add('empty');
                }
            });
            wrapper.appendChild(preview);
        }
        return wrapper;
    }

    function syncBlendControls() {
        const raw = state.values.get('blend_mode') === 'raw';
        for (const field of ['style_alpha', 'style_beta']) {
            document.querySelector(`[data-control="${field}"]`)?.classList.toggle('hidden', !raw);
        }
        for (const field of ['speaker_reference_strength', 'style_reference_strength']) {
            document.querySelector(`[data-control="${field}"]`)?.classList.toggle('hidden', raw);
        }
    }

    function renderControls(path) {
        for (const group of ['basic', 'advanced', 'expert', 'developer']) {
            const target = byId(`speech-controls-${group}`);
            target.replaceChildren(...controlsForPath(path, group).map(controlInput));
        }
        syncBlendControls();
    }

    function renderPathSelector() {
        const select = byId('speech-preset');
        const liveSelect = byId('live-recipe');
        const sampleLanguage = byId('speech-language')?.value || '';
        const supportsSelectedLanguage = (composition) => (
            !sampleLanguage
            || pathSupportsSampleLanguage(
                pathForComposition(composition, state.discovery),
                sampleLanguage,
            )
        );
        select.replaceChildren();
        liveSelect?.replaceChildren();
        select.appendChild(new Option('Current pipeline', 'custom'));
        liveSelect?.appendChild(new Option('Current pipeline', 'custom'));
        for (const preset of state.discovery.presets || []) {
            const composition = state.discovery.compositions.find(
                (candidate) => candidate.id === preset.composition_id,
            );
            if (preset.developer || !composition?.runnable || !supportsSelectedLanguage(composition)) {
                continue;
            }
            const option = new Option(
                preset.display_name,
                preset.id,
            );
            select.appendChild(option);
            liveSelect?.appendChild(new Option(preset.display_name, preset.id));
        }
        for (const recipe of state.userRecipes) {
            const composition = state.discovery.compositions.find(
                (candidate) => candidate.id === recipe.compositionId,
            );
            if (!composition || !supportsSelectedLanguage(composition)) continue;
            select.appendChild(new Option(`${recipe.name} · saved`, recipe.id));
            liveSelect?.appendChild(new Option(`${recipe.name} · saved`, recipe.id));
        }
        const matchingPreset = (state.discovery.presets || []).find((preset) => (
            preset.composition_id === state.pathKey && preset.id === state.presetId
        ));
        const matchingUserRecipe = state.userRecipes.find(
            (recipe) => recipe.compositionId === state.pathKey && recipe.id === state.presetId,
        );
        select.value = matchingUserRecipe?.id || matchingPreset?.id || 'custom';
        if (liveSelect) liveSelect.value = select.value;
        const voiceInput = byId('speech-voice');
        const voiceOptions = byId('speech-voice-options');
        voiceOptions.replaceChildren();
        for (const composition of availableCompositions(state.discovery).filter(supportsSelectedLanguage)) {
            const path = (state.discovery.paths || []).find((candidate) => (
                candidate.backend === composition.backend && candidate.model === composition.model
            ));
            const option = document.createElement('option');
            option.value = composition.display_name;
            option.dataset.compositionId = composition.id;
            const languages = [...new Set((path?.catalog || []).flatMap(
                (entry) => entry.languages || [],
            ))];
            option.label = [
                languages.join(', '),
                composition.backend,
                composition.runnable ? 'ready' : 'unavailable',
            ].filter(Boolean).join(' · ');
            voiceOptions.appendChild(option);
        }
        const selected = selectedComposition();
        if (voiceInput && document.activeElement !== voiceInput) {
            voiceInput.value = selected?.display_name || selectedPath()?.display_name || '';
        }
        const mock = (state.discovery.compositions || []).find((path) => path.backend === 'mock');
        byId('select-mock-path').classList.toggle('hidden', !mock);
    }

    function renderSpeechSamples({ replaceText = false } = {}) {
        const languageSelect = byId('speech-language');
        const sampleSelect = byId('speech-sample');
        if (!languageSelect || !sampleSelect || !state.discovery) return;
        const languages = sampleLanguagesForDiscovery(state.discovery);
        const previousLanguage = languageSelect.value;
        languageSelect.replaceChildren(...languages.map((language) => (
            new Option(
                language.name === language.nativeName
                    ? language.name
                    : `${language.name} · ${language.nativeName}`,
                language.code,
            )
        )));
        const selectedLanguage = languages.find((language) => language.code === previousLanguage)
            || languages.find((language) => pathSupportsSampleLanguage(selectedPath(), language))
            || languages.find((language) => language.code === 'en')
            || languages[0];
        languageSelect.value = selectedLanguage?.code || '';
        languageSelect.disabled = languages.length < 2;
        byId('speech-language-detail').textContent = languages.length
            ? `${languages.length} sampled languages have a discovered speech path. Voice and recipe choices are filtered to this language.`
            : 'No discovered speech path matches the sample library.';
        sampleSelect.replaceChildren(...(selectedLanguage?.texts || []).map((text, index) => (
            new Option(
                `${index + 1}. ${text.length > 72 ? `${text.slice(0, 69)}…` : text}`,
                String(index),
            )
        )));
        sampleSelect.disabled = !selectedLanguage?.texts?.length;
        if (replaceText && selectedLanguage?.texts?.length) {
            byId('text').value = selectedLanguage.texts[0];
        }
    }

    function componentById(id) {
        return (state.discovery.components || []).find((component) => component.id === id);
    }

    function componentOptions(select, stage, selectedId, compatibleWith = null) {
        select.replaceChildren();
        const candidates = (state.discovery.components || []).filter((component) => (
            component.stage === stage
        ));
        for (const component of candidates) {
            const option = new Option(component.display_name, component.id);
            let reason = '';
            if (compatibleWith) {
                const edge = compatibilityFor(
                    state.discovery,
                    compatibleWith.from === 'selected' ? selectedId : compatibleWith.from,
                    compatibleWith.to === 'component' ? component.id : compatibleWith.to,
                );
                if (!edge?.compatible) reason = edge?.reason || 'No compatible executable connection is registered.';
            }
            if (!component.runnable && component.id !== selectedId) {
                reason ||= component.explanation || 'Component is not ready for runtime synthesis.';
            }
            option.disabled = Boolean(reason);
            option.title = reason;
            if (reason) option.textContent += ` — ${reason}`;
            select.appendChild(option);
        }
        if (![...select.options].some((option) => option.value === selectedId)) {
            select.appendChild(new Option(selectedId, selectedId));
        }
        select.value = selectedId;
    }

    function renderComposition(path) {
        const pipeline = path.pipeline;
        if (!pipeline) return;
        const generatorId = pipeline.end_to_end || pipeline.acoustic_model;
        const generator = componentById(generatorId);
        const projector = componentById(pipeline.projector);
        const generatorSelect = byId('pipeline-generator');
        generatorSelect.replaceChildren();
        const generatorIds = new Set((state.discovery.compositions || []).map(compositionGenerator));
        for (const component of state.discovery.components || []) {
            if (!['acoustic_model', 'end_to_end'].includes(component.stage)) continue;
            const option = new Option(component.display_name, component.id);
            const executable = generatorIds.has(component.id);
            option.disabled = !executable;
            if (!executable) {
                option.title = component.explanation || 'No executable composition is registered.';
                option.textContent += ` — ${option.title}`;
            }
            generatorSelect.appendChild(option);
        }
        if (![...generatorSelect.options].some((option) => option.value === generatorId)) {
            generatorSelect.appendChild(new Option(generatorId, generatorId));
        }
        generatorSelect.value = generatorId;

        const projectorSelect = byId('pipeline-projector');
        projectorSelect.replaceChildren();
        for (const component of (state.discovery.components || []).filter(
            (candidate) => candidate.stage === 'projector',
        )) {
            const edge = compatibilityFor(state.discovery, component.id, generatorId);
            const option = new Option(component.display_name, component.id);
            option.disabled = !edge?.compatible;
            option.title = edge?.reason || 'No compatible projector contract is registered.';
            if (option.disabled) option.textContent += ` — ${option.title}`;
            projectorSelect.appendChild(option);
        }
        projectorSelect.value = pipeline.projector;
        byId('pipeline-projector-detail').textContent = (
            compatibilityFor(state.discovery, pipeline.projector, generatorId)?.reason
            || projector?.explanation
            || 'Checkpoint-bound linguistic projection.'
        );

        const endToEnd = Boolean(pipeline.end_to_end);
        const vocoderStage = byId('pipeline-vocoder-stage');
        vocoderStage.classList.toggle('pipeline-stage-spanned', endToEnd);
        byId('pipeline-generator').previousElementSibling.textContent = endToEnd
            ? 'End-to-end model'
            : 'Acoustic model';
        byId('pipeline-generator-detail').textContent = endToEnd
            ? 'This block spans acoustic generation and waveform decoding.'
            : (generator?.produces?.[0]?.summary || 'Produces features for a compatible vocoder.');
        const vocoder = byId('pipeline-vocoder');
        if (endToEnd) {
            vocoder.replaceChildren(new Option(`Integrated in ${generator?.display_name || generatorId}`, 'integrated'));
            vocoder.disabled = true;
            byId('pipeline-vocoder-detail').textContent = 'Waveform decoding is integrated into this model.';
        } else {
            vocoder.disabled = false;
            componentOptions(vocoder, 'vocoder', pipeline.vocoder, {
                from: generatorId,
                to: 'component',
            });
            byId('pipeline-vocoder-detail').textContent = (
                compatibilityFor(state.discovery, generatorId, pipeline.vocoder)?.reason
                || 'Only exact contract matches can be selected.'
            );
            if (pipeline.vocoder?.includes('standardiz')) {
                byId('pipeline-vocoder').previousElementSibling.textContent = 'Adapter + vocoder';
            } else {
                byId('pipeline-vocoder').previousElementSibling.textContent = 'Vocoder';
            }
        }
        const conditioners = (pipeline.conditioners || []).map((id) => (
            componentById(id)?.display_name || id
        ));
        byId('pipeline-conditioning-name').textContent = conditioners.join(', ')
            || 'Built-in model conditioning';
        byId('pipeline-conditioning-detail').textContent = [
            path.speakers?.required ? 'speaker required' : null,
            path.languages?.required ? 'model language required' : null,
            path.reference_audio?.speaker ? 'reference voice' : null,
            path.styles?.reference_audio ? 'reference style' : null,
        ].filter(Boolean).join(' · ') || 'Variety and synthesis controls';
        renderStageInspector(path, state.selectedStage);
    }

    function stageComponent(path, stage) {
        const pipeline = path?.pipeline || {};
        const id = {
            input: pipeline.input,
            projector: pipeline.projector,
            conditioner: pipeline.conditioners?.[0],
            generator: pipeline.end_to_end || pipeline.acoustic_model,
            vocoder: pipeline.vocoder || (pipeline.end_to_end ? pipeline.end_to_end : null),
            output: pipeline.output,
        }[stage];
        return id ? componentById(id) : null;
    }

    function contractList(contracts, fallback) {
        if (!(contracts || []).length) return `<li>${escapeHtml(fallback)}</li>`;
        return contracts.map((contract) => `
            <li><strong>${escapeHtml(contract.kind)}</strong> · ${escapeHtml(contract.summary || contract.key)}</li>
        `).join('');
    }

    function renderStageInspector(path, stage = 'generator') {
        const target = byId('pipeline-stage-inspector');
        if (!target || !path) return;
        state.selectedStage = stage;
        document.querySelectorAll('.pipeline-stage[data-stage]').forEach((element) => {
            element.classList.toggle('selected', element.dataset.stage === stage);
        });
        const component = stageComponent(path, stage);
        const pipeline = path.pipeline || {};
        const generatorId = pipeline.end_to_end || pipeline.acoustic_model;
        let ownership = 'This stage is selected independently when an executable replacement exists.';
        if (stage === 'projector') {
            ownership = compatibilityFor(state.discovery, pipeline.projector, generatorId)?.reason
                || 'The projector is owned by the selected checkpoint and cannot be substituted independently.';
        } else if (stage === 'vocoder' && pipeline.end_to_end) {
            ownership = 'Waveform decoding is integrated into the end-to-end checkpoint.';
        } else if (stage === 'vocoder' && pipeline.vocoder?.includes('standardiz')) {
            ownership = 'The named standardizer is an explicit part of this adapter and vocoder stage.';
        }
        const statuses = component?.statuses || path.statuses || [];
        const audioInput = stage === 'input' ? state.audioInputCapabilities : null;
        const sourceKinds = (audioInput?.source_kinds || [])
            .map((kind) => kind.replaceAll('_', ' '))
            .join(', ');
        const inputDiscovery = audioInput ? `
            <h4>Recognition input discovery</h4>
            <p>${escapeHtml(sourceKinds || 'No source adapters reported.')}</p>
            <p>${escapeHtml(
                (audioInput.devices || []).length
                    ? (audioInput.devices || []).map((device) => (
                        `${device.display_name}${device.is_default ? ' (default)' : ''}`
                    )).join(', ')
                    : (audioInput.device_discovery_error || 'No local microphone devices reported.')
            )}</p>
            ${renderLanguageRoutingControls()}
            <div class="browser-mic-probe">
                <label>VAD
                    <select id="browser-mic-vad">
                        <option value="web_rtc">WebRTC</option>
                        <option value="energy">Energy baseline</option>
                    </select>
                </label>
                <details>
                    <summary>Segmentation controls</summary>
                    <div class="browser-mic-controls">
                        <label>Speech start (ms)<input id="browser-mic-speech-start" type="number" min="10" step="10" value="30"></label>
                        <label>Acoustic end (ms)<input id="browser-mic-acoustic-end" type="number" min="10" step="10" value="300"></label>
                        <label>Segment close (ms)<input id="browser-mic-segment-end" type="number" min="10" step="10" value="800"></label>
                        <label>Pre-roll (ms)<input id="browser-mic-pre-roll" type="number" min="0" step="10" value="200"></label>
                        <label>Minimum speech (ms)<input id="browser-mic-minimum-speech" type="number" min="0" step="10" value="250"></label>
                        <label>Maximum segment (ms)<input id="browser-mic-maximum-segment" type="number" min="100" step="100" value="30000"></label>
                    </div>
                </details>
                <details>
                    <summary>Server audio cleanup</summary>
                    <div class="browser-mic-controls">
                        <label><input id="browser-cleanup-dc" type="checkbox" checked> DC removal</label>
                        <label><input id="browser-cleanup-gate" type="checkbox" checked> Noise gate</label>
                        <label><input id="browser-cleanup-gain" type="checkbox" checked> Gain control</label>
                        <label><input id="browser-cleanup-low-pass" type="checkbox"> Low-pass filter</label>
                        <label><input id="browser-cleanup-echo" type="checkbox"> Echo cancellation</label>
                        <label><input id="browser-cleanup-separation" type="checkbox"> Source separation baseline</label>
                    </div>
                    <p>Browser-native echo cancellation, noise suppression, and gain control remain disabled for this test; selected stages run on the Tongues server.</p>
                </details>
                <button id="browser-mic-start" type="button">Test this browser’s microphone</button>
                <button id="browser-mic-stop" type="button" class="secondary-button" disabled>Stop test</button>
                <meter id="browser-mic-level" min="0" max="1" value="0" aria-label="Browser microphone level"></meter>
                <div class="browser-cleanup-comparison">
                    <label>Raw <meter id="browser-mic-raw-level" min="0" max="1" value="0"></meter></label>
                    <label>Server processed <meter id="browser-mic-processed-level" min="0" max="1" value="0"></meter></label>
                </div>
                <p id="browser-cleanup-status">Raw and processed levels will use the same server cleanup pipeline as recognition.</p>
                <p id="browser-mic-status" aria-live="polite">Uses this browser device, not the server host microphone.</p>
                <ol id="browser-mic-events" class="browser-mic-events" aria-label="Utterance lifecycle events"></ol>
            </div>
        ` : '';
        target.innerHTML = `
            <p class="eyebrow">${escapeHtml(stage.replaceAll('_', ' '))}</p>
            <h3>${escapeHtml(component?.display_name || stage)}</h3>
            <p>${escapeHtml(component?.explanation || ownership)}</p>
            <div class="status-badges">${statuses.map(
                (status) => `<span class="status-badge">${escapeHtml(status)}</span>`,
            ).join('')}</div>
            <h4>Accepted contract</h4>
            <ul>${contractList(component?.accepts, stage === 'input' ? 'User text' : 'No separate input contract')}</ul>
            <h4>Emitted contract</h4>
            <ul>${contractList(component?.produces, stage === 'output' ? 'Downloadable WAV audio' : 'No separate output contract')}</ul>
            <h4>Ownership and compatibility</h4>
            <p>${escapeHtml(ownership)}</p>
            ${inputDiscovery}
        `;
        if (audioInput) {
            byId('browser-mic-start')?.addEventListener('click', () => {
                startBrowserMicProbe().catch((error) => {
                    setBrowserMicStatus(`Browser microphone failed: ${error.message}`, true);
                });
            });
            byId('browser-mic-stop')?.addEventListener('click', () => {
                stopBrowserMicProbe();
            });
            byId('recognition-language-mode')?.addEventListener('change', updateLanguageRoutingLabels);
            updateLanguageRoutingLabels();
        }
    }

    function renderLanguageRoutingControls() {
        const capabilities = state.languageRoutingCapabilities;
        if (!capabilities) return '<p>Language routing capabilities unavailable.</p>';
        const languages = [...new Set(
            (capabilities.asr_providers || []).flatMap((provider) => provider.languages || []),
        )].sort();
        const detectorReady = (capabilities.detectors || []).some((detector) => detector.installed);
        const policy = capabilities.default_switch_policy || {};
        return `
            <div class="language-routing-controls">
                <h4>Recognition language routing</h4>
                <label>Mode
                    <select id="recognition-language-mode">
                        <option value="fixed">Fixed language</option>
                        <option value="detect" ${detectorReady ? '' : 'disabled'}>Auto-detect${detectorReady ? '' : ' (detector model not installed)'}</option>
                    </select>
                </label>
                <label><span id="recognition-language-label">Fixed language</span>
                    <select id="recognition-language-fixed">
                        ${languages.map((language) => `<option value="${escapeAttribute(language)}" ${language === 'en' ? 'selected' : ''}>${escapeHtml(language)}</option>`).join('')}
                    </select>
                </label>
                <label>Minimum confidence
                    <input id="recognition-language-confidence" type="number" min="0" max="1" step="0.05" value="${escapeAttribute(policy.minimum_confidence ?? 0.65)}">
                </label>
                <label>Minimum evidence (ms)
                    <input id="recognition-language-evidence" type="number" min="0" step="50" value="${escapeAttribute(policy.minimum_evidence_ms ?? 300)}">
                </label>
                <label>Switch margin
                    <input id="recognition-language-margin" type="number" min="0" max="1" step="0.05" value="${escapeAttribute(policy.switch_margin ?? 0.15)}">
                </label>
                <p>${detectorReady
                    ? 'Detection evidence remains visible if ASR routing falls back.'
                    : 'Auto-detect fails closed until the advertised detector model is installed; fixed-language routing remains available.'}</p>
                <p id="recognition-language-status" class="recognition-language-status">No segment has been routed yet.</p>
            </div>
        `;
    }

    function updateLanguageRoutingLabels() {
        const detecting = byId('recognition-language-mode')?.value === 'detect';
        const label = byId('recognition-language-label');
        if (label) label.textContent = detecting ? 'Fallback language' : 'Fixed language';
        const status = byId('recognition-language-status');
        if (status && !status.dataset.routed) {
            status.textContent = detecting
                ? 'Waiting for a finalized segment to identify.'
                : `Pinned to ${byId('recognition-language-fixed')?.value || 'en'}.`;
        }
    }

    function setBrowserMicStatus(message, failed = false) {
        const status = byId('browser-mic-status');
        if (status) {
            status.textContent = message;
            status.dataset.state = failed ? 'failed' : 'ready';
        }
    }

    function appendBrowserMicEvent(message) {
        const events = byId('browser-mic-events');
        if (!events) return;
        const item = document.createElement('li');
        item.textContent = message;
        events.prepend(item);
        while (events.children.length > 6) events.lastElementChild?.remove();
    }

    function browserMicSocketUrl() {
        const scheme = browser.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${scheme}//${browser.location.host}/api/audio-input/browser-stream`;
    }

    async function startBrowserMicProbe() {
        if (!browser?.isSecureContext) {
            throw new Error(
                'microphone capture requires a secure browser context; use trusted HTTPS or open http://localhost:3000 through an SSH port-forward',
            );
        }
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error('this browser does not expose getUserMedia');
        }
        stopBrowserMicProbe(false);
        const start = byId('browser-mic-start');
        const stop = byId('browser-mic-stop');
        const segmentation = {
            frame_ms: 10,
            speech_start_ms: Number(byId('browser-mic-speech-start')?.value || 30),
            acoustic_end_silence_ms: Number(byId('browser-mic-acoustic-end')?.value || 300),
            segment_end_silence_ms: Number(byId('browser-mic-segment-end')?.value || 800),
            minimum_speech_ms: Number(byId('browser-mic-minimum-speech')?.value || 250),
            pre_roll_ms: Number(byId('browser-mic-pre-roll')?.value || 200),
            maximum_segment_ms: Number(byId('browser-mic-maximum-segment')?.value || 30000),
        };
        const vadBackend = byId('browser-mic-vad')?.value || 'web_rtc';
        const languageMode = byId('recognition-language-mode')?.value || 'fixed';
        const fixedLanguage = byId('recognition-language-fixed')?.value || 'en';
        const languageRouting = {
            mode: languageMode === 'detect'
                ? { mode: 'detect', candidates: [] }
                : { mode: 'fixed', language: fixedLanguage },
            switching: {
                minimum_confidence: Number(byId('recognition-language-confidence')?.value || 0.65),
                minimum_evidence_ms: Number(byId('recognition-language-evidence')?.value || 300),
                switch_margin: Number(byId('recognition-language-margin')?.value || 0.15),
                consecutive_segments: 2,
            },
            unsupported: {
                policy: 'fallback',
                provider_id: 'common-phone',
                language: 'en',
            },
        };
        const cleanup = [];
        if (byId('browser-cleanup-dc')?.checked) cleanup.push({ kind: 'dc_removal', pole: 0.995 });
        if (byId('browser-cleanup-gate')?.checked) cleanup.push({ kind: 'noise_gate', threshold_rms: 0.012, attenuation: 0.15 });
        if (byId('browser-cleanup-gain')?.checked) cleanup.push({ kind: 'gain_control', target_rms: 0.08, maximum_gain: 4, adaptation: 0.05 });
        if (byId('browser-cleanup-low-pass')?.checked) cleanup.push({ kind: 'low_pass', cutoff_hz: 7000 });
        if (byId('browser-cleanup-echo')?.checked) cleanup.push({ kind: 'echo_cancellation', delay_ms: 100, strength: 0.5 });
        if (byId('browser-cleanup-separation')?.checked) cleanup.push({ kind: 'source_separation', floor_rms: 0.01 });
        const events = byId('browser-mic-events');
        if (events) events.replaceChildren();
        if (start) start.disabled = true;
        setBrowserMicStatus('Requesting browser microphone permission…');
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                },
                video: false,
            });
            state.browserMicStream = stream;
            const AudioContext = browser.AudioContext || browser.webkitAudioContext;
            if (!AudioContext) throw new Error('this browser does not expose Web Audio');
            const context = new AudioContext();
            await context.audioWorklet.addModule('/browser-mic-worklet.js');
            await context.resume();
            const socket = new WebSocket(browserMicSocketUrl());
            socket.binaryType = 'arraybuffer';
            state.browserMicContext = context;
            state.browserMicSocket = socket;
            state.browserMicSequence = 0;
            state.browserMicStartFrame = 0;

            const ready = new Promise((resolve, reject) => {
                const timeout = browser.setTimeout(
                    () => reject(new Error('server did not open the microphone stream')),
                    5000,
                );
                socket.addEventListener('open', () => {
                    socket.send(JSON.stringify({
                        type: 'open',
                        schema_version: 1,
                        sample_rate_hz: context.sampleRate,
                        channels: 1,
                        vad_backend: vadBackend,
                        segmentation,
                        cleanup,
                        language_routing: languageRouting,
                    }));
                }, { once: true });
                socket.addEventListener('message', (event) => {
                    let message;
                    try {
                        message = JSON.parse(event.data);
                    } catch {
                        return;
                    }
                    if (message.type === 'ready') {
                        browser.clearTimeout(timeout);
                        resolve(message);
                    } else if (message.type === 'level') {
                        const meter = byId('browser-mic-level');
                        if (meter) meter.value = Math.min(1, Number(message.rms || 0) * 4);
                        setBrowserMicStatus(
                            `Browser audio reached Tongues: RMS ${Number(message.rms || 0).toFixed(3)}, peak ${Number(message.peak || 0).toFixed(3)}, VAD ${Number(message.speech_probability || 0).toFixed(2)} (${message.is_speech ? 'speech' : 'silence'}), frame ${message.chunk_sequence}.`,
                        );
                    } else if (message.type === 'cleanup_compared') {
                        const rawMeter = byId('browser-mic-raw-level');
                        const processedMeter = byId('browser-mic-processed-level');
                        if (rawMeter) rawMeter.value = Math.min(1, Number(message.raw_rms || 0) * 4);
                        if (processedMeter) processedMeter.value = Math.min(1, Number(message.processed_rms || 0) * 4);
                        const cleanupStatus = byId('browser-cleanup-status');
                        if (cleanupStatus) {
                            cleanupStatus.textContent = `Raw RMS ${Number(message.raw_rms || 0).toFixed(3)} / processed RMS ${Number(message.processed_rms || 0).toFixed(3)}; ${message.stages?.length || 0} server stages; ${message.algorithmic_latency_frames || 0} frames declared latency.`;
                        }
                    } else if (message.type === 'segment_opened') {
                        appendBrowserMicEvent(
                            `Segment ${message.segment_id} opened with ${message.pre_roll_frames} pre-roll frames.`,
                        );
                    } else if (message.type === 'speech_ended') {
                        appendBrowserMicEvent(
                            `Acoustic speech ended for ${message.segment_id}; endpoint latency ${message.endpoint_latency_ms} ms, conversational segment still open.`,
                        );
                    } else if (message.type === 'segment_final') {
                        appendBrowserMicEvent(
                            `${message.accepted ? 'Finalized' : 'Dropped short'} segment ${message.segment_id}: ${message.speech_duration_ms} ms speech, ${message.reason}.`,
                        );
                    } else if (message.type === 'language_routed') {
                        const route = message.route || {};
                        const detected = route.detection?.hypotheses?.map((hypothesis) => (
                            `${hypothesis.language} ${Number(hypothesis.confidence || 0).toFixed(2)}`
                        )).join(', ') || 'fixed';
                        const routeStatus = byId('recognition-language-status');
                        if (routeStatus) {
                            routeStatus.dataset.routed = 'true';
                            routeStatus.textContent = `Detected ${detected}; active route ${route.selected_language} → ${route.provider_id}.`;
                        }
                        appendBrowserMicEvent(
                            `Language ${detected}; routed ${route.selected_language} to ${route.provider_id} (${route.model_id})${route.fallback_reason ? `; ${route.fallback_reason}` : ''}.`,
                        );
                    } else if (message.type === 'discontinuity') {
                        appendBrowserMicEvent(
                            `Discontinuity: ${message.reason} (expected ${message.expected_chunk_sequence}, received ${message.received_chunk_sequence}).`,
                        );
                        setBrowserMicStatus(
                            `Browser audio discontinuity: ${message.reason} (expected ${message.expected_chunk_sequence}, received ${message.received_chunk_sequence}).`,
                            true,
                        );
                    } else if (message.type === 'error') {
                        setBrowserMicStatus(`${message.code}: ${message.message}`, true);
                    } else if (message.type === 'ended') {
                        socket.close();
                        stopBrowserMicProbe(false);
                        appendBrowserMicEvent(
                            `Stopped: ${Number(message.metrics?.segments_finalized || 0)} finalized, ${Number(message.metrics?.segments_dropped || 0)} dropped, ${Number(message.metrics?.forced_flushes || 0)} forced flushes.`,
                        );
                        setBrowserMicStatus(
                            `Browser microphone test stopped. Speech ratio ${Number(message.metrics?.speech_frames || 0)}/${Number(message.metrics?.frames_observed || 0)}; forced flushes ${Number(message.metrics?.forced_flushes || 0)}.`,
                        );
                    }
                });
                socket.addEventListener('error', () => {
                    browser.clearTimeout(timeout);
                    reject(new Error('browser microphone WebSocket failed'));
                }, { once: true });
            });
            await ready;

            const source = context.createMediaStreamSource(stream);
            const node = new AudioWorkletNode(context, 'tongues-browser-mic-capture');
            const silentGain = context.createGain();
            silentGain.gain.value = 0;
            node.port.onmessage = (event) => {
                if (socket.readyState !== WebSocket.OPEN) return;
                const pcm = event.data;
                const packet = new ArrayBuffer(16 + pcm.byteLength);
                const header = new DataView(packet, 0, 16);
                header.setBigUint64(0, BigInt(state.browserMicSequence), true);
                header.setBigUint64(8, BigInt(state.browserMicStartFrame), true);
                new Uint8Array(packet, 16).set(new Uint8Array(pcm));
                socket.send(packet);
                state.browserMicSequence += 1;
                state.browserMicStartFrame += pcm.byteLength / Float32Array.BYTES_PER_ELEMENT;
            };
            source.connect(node);
            node.connect(silentGain);
            silentGain.connect(context.destination);
            state.browserMicSource = source;
            state.browserMicNode = node;
            state.browserMicGain = silentGain;
            if (stop) stop.disabled = false;
            setBrowserMicStatus(
                `Streaming this browser microphone at ${context.sampleRate} Hz to Tongues…`,
            );
        } catch (error) {
            stopBrowserMicProbe(false);
            if (start) start.disabled = false;
            throw error;
        }
    }

    function stopBrowserMicProbe(sendEnd = true) {
        const socket = state.browserMicSocket;
        if (sendEnd && socket?.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: 'end' }));
        } else {
            socket?.close();
        }
        state.browserMicNode?.disconnect();
        state.browserMicSource?.disconnect();
        state.browserMicGain?.disconnect();
        for (const track of state.browserMicStream?.getTracks?.() || []) track.stop();
        state.browserMicContext?.close?.().catch(() => {});
        state.browserMicSocket = null;
        state.browserMicContext = null;
        state.browserMicStream = null;
        state.browserMicSource = null;
        state.browserMicNode = null;
        state.browserMicGain = null;
        const start = byId('browser-mic-start');
        const stop = byId('browser-mic-stop');
        if (start) start.disabled = false;
        if (stop) stop.disabled = true;
    }

    function renderRecipeSummary(path) {
        const target = byId('speech-recipe-summary');
        if (!target || !path) return;
        const varieties = varietiesForPath(path);
        const device = state.values.get('device')
            || path.controls?.find((control) => control.field === 'device')?.default
            || 'runtime default';
        target.innerHTML = `
            <div>
                <p class="eyebrow">Selected recipe</p>
                <strong>${escapeHtml(path.display_name)}</strong>
                <p>${escapeHtml(
                    listedValues(path.speakers?.values)[0]?.label
                    || varieties[0]?.label
                    || path.family
                    || 'Speech synthesis'
                )}</p>
            </div>
            <dl>
                <div><dt>Language</dt><dd>${escapeHtml(varieties[0]?.label || 'Model default')}</dd></div>
                <div><dt>Audio</dt><dd>${escapeHtml(
                    path.output?.sample_rate_hz
                        ? `${path.output.sample_rate_hz} Hz WAV`
                        : 'WAV'
                )}</dd></div>
                <div><dt>Request device</dt><dd>${escapeHtml(device)}</dd></div>
                <div><dt>Verification</dt><dd>${escapeHtml(
                    path.verified ? 'Verified' : path.verification_status || 'Unknown'
                )}</dd></div>
            </dl>
        `;
    }

    function renderVarieties(path) {
        const select = byId('variety');
        const fixed = byId('fixed-variety');
        const varieties = varietiesForPath(path);
        select.replaceChildren();
        for (const variety of varieties) {
            select.appendChild(new Option(variety.label, variety.id));
        }
        const previous = state.values.get(`variety:${path.id}`);
        if (previous && varieties.some((item) => item.id === previous)) select.value = previous;
        select.classList.toggle('hidden', varieties.length === 1);
        fixed.classList.toggle('hidden', varieties.length !== 1);
        fixed.textContent = varieties[0]?.label || 'No supported variety declared';
        select.disabled = varieties.length < 2;
        byId('variety-detail').textContent = varieties.length === 1
            ? 'This path is fixed to its declared linguistic variety.'
            : `${varieties.length} model-supported varieties.`;
    }

    function renderSpeakers(path) {
        const control = byId('speaker-control');
        const input = byId('speaker');
        const datalist = byId('speaker-options');
        const speakers = listedValues(path.speakers?.values);
        control.classList.toggle('hidden', speakers.length === 0 && !path.speakers?.required);
        datalist.replaceChildren(...speakers.map((speaker) => {
            const option = document.createElement('option');
            option.value = speaker.id;
            option.label = speaker.numeric_id == null
                ? speaker.label
                : `${speaker.label} · embedding ${speaker.numeric_id}`;
            return option;
        }));
        const previous = state.values.get(`speaker:${path.id}`);
        input.value = previous && speakers.some((item) => item.id === previous)
            ? previous
            : (speakers.find((item) => item.id === 'p225')?.id || '');
        input.required = Boolean(path.speakers?.required);
        input.placeholder = speakers.length > 20 ? `Search ${speakers.length} named speakers` : 'Speaker name';
        byId('speaker-detail').textContent = `${speakers.length} named embeddings${path.speakers?.required ? ' · selection required' : ''}.`;
    }

    function renderModelCard(path) {
        const card = byId('speech-model-card');
        const catalog = path.catalog || [];
        const licenses = [...new Set(catalog.map((entry) => entry.license.expression))];
        const provenance = [...new Set(catalog.map((entry) => (
            `${entry.provenance.format} · ${entry.provenance.source}`
        )))];
        const rate = path.output?.sample_rate_hz;
        const languages = [...new Set(catalog.flatMap((entry) => entry.languages || []))];
        const scripts = [...new Set(catalog.map((entry) => entry.script).filter(Boolean))];
        const preprocessing = [...new Set(catalog.flatMap((entry) => entry.preprocessing || []))];
        const statuses = path.statuses.map((status) => `<span class="status-badge">${escapeHtml(status)}</span>`).join('');
        card.innerHTML = `
            <div class="speech-section-heading">
                <div>
                    <p class="eyebrow">${escapeHtml(path.family || 'speech')}</p>
                    <h2>${escapeHtml(path.display_name)}</h2>
                </div>
                <div class="status-badges">${statuses}</div>
            </div>
            <dl class="metadata-grid">
                <div><dt>Architecture</dt><dd>${escapeHtml(catalog.map((entry) => entry.architecture).join(' → ') || path.model)}</dd></div>
                <div><dt>Audio</dt><dd>${rate ? `${rate} Hz · ${path.output.channels} channel` : 'Not declared'}</dd></div>
                <div><dt>Devices</dt><dd>${escapeHtml((path.controls.find((item) => item.field === 'device')?.options || []).map((item) => item.label).join(', '))}</dd></div>
                <div><dt>Language</dt><dd>${escapeHtml(languages.join(', ') || 'Not declared')}</dd></div>
                <div><dt>Script</dt><dd>${escapeHtml(scripts.join(', ') || 'Source default')}</dd></div>
                <div><dt>Preprocessing</dt><dd>${escapeHtml(preprocessing.join(', ') || 'No additional preprocessing required')}</dd></div>
                <div><dt>Speakers</dt><dd>${listedValues(path.speakers?.values).length || catalog.reduce((sum, entry) => sum + (entry.speakers?.count || 0), 0)}</dd></div>
                <div><dt>License</dt><dd>${escapeHtml(
                    licenses.length
                        ? licenses.map((license) => (
                            license === 'NOASSERTION'
                                ? 'Metadata unavailable; check upstream before redistribution'
                                : license
                        )).join(', ')
                        : 'No catalog license metadata'
                )}</dd></div>
                <div><dt>Readiness</dt><dd>${escapeHtml(path.runnable ? 'Ready' : path.statuses.join(', '))}</dd></div>
                <div><dt>Resident</dt><dd>${escapeHtml(path.load_state)}</dd></div>
                <div><dt>Provenance</dt><dd>${escapeHtml(provenance.join(' | ') || path.provenance.join(', ') || 'Not declared')}</dd></div>
            </dl>
            ${path.unavailable_reason ? `<p class="inline-error">${escapeHtml(path.unavailable_reason)}</p>` : ''}
            <div class="model-actions">
                ${path.install_command ? '<button type="button" class="secondary-button fetch-speech-artifact">Fetch artifact</button>' : ''}
                ${path.install_command ? `<button type="button" class="secondary-button copy-install-command" data-command="${escapeAttribute(path.install_command)}">Copy install command</button>` : ''}
                ${catalog.length && path.verification_status !== 'verified' && path.verification_status !== 'unavailable' ? '<button type="button" class="secondary-button verify-speech-pipeline">Verify pipeline</button>' : ''}
                ${path.load_state === 'loaded' ? '<button type="button" class="secondary-button unload-speech-path">Unload model</button>' : ''}
            </div>`;
        card.querySelector('.fetch-speech-artifact')?.addEventListener('click', async (event) => {
            const button = event.currentTarget;
            button.disabled = true;
            button.textContent = 'Starting fetch…';
            try {
                await startCatalogFetch(path, path.display_name);
            } catch (error) {
                showError(`Fetch failed: ${error.message}`);
                button.disabled = false;
                button.textContent = 'Fetch artifact';
            }
        });
        card.querySelector('.copy-install-command')?.addEventListener('click', async (event) => {
            await navigator.clipboard.writeText(event.currentTarget.dataset.command);
            event.currentTarget.textContent = 'Install command copied';
        });
        card.querySelector('.unload-speech-path')?.addEventListener('click', async (event) => {
            event.currentTarget.disabled = true;
            try {
                const response = await fetch('/api/speech/runtime/unload', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(path.pipeline
                        ? { pipeline: path.pipeline }
                        : { backend: path.backend, model: path.model }),
                });
                if (!response.ok) throw new Error(await response.text());
                renderRuntime(await response.json());
                await refreshDiscovery();
            } catch (error) {
                showError(`Unload failed: ${error.message}`);
            }
        });
        card.querySelector('.verify-speech-pipeline')?.addEventListener('click', async (event) => {
            event.currentTarget.disabled = true;
            try {
                await verifyModelIds(catalog.map((entry) => entry.id));
            } catch (error) {
                showError(`Pipeline verification failed: ${error.message}`);
            } finally {
                event.currentTarget.disabled = false;
            }
        });
    }

    function renderSelectedPath() {
        const path = selectedPath();
        if (!path) {
            showError(state.discovery?.error || 'No synthesis paths were discovered.');
            byId('submit-btn').disabled = true;
            byId('compose-validity').dataset.state = 'failed';
            byId('compose-validity').textContent = 'Incomplete';
            return;
        }
        clearError();
        byId('synthesis-path-detail').textContent = path.runnable
            ? `${path.display_name} is a complete, contract-valid pipeline.`
            : 'Pipeline unavailable. Inspect its model card for installation or verification details.';
        renderComposition(path);
        renderVarieties(path);
        renderSpeakers(path);
        renderModelCard(path);
        renderRecipeSummary(path);
        renderControls(path);
        byId('submit-btn').disabled = !path.complete || !path.runnable;
        const validity = byId('compose-validity');
        validity.dataset.state = path.runnable ? 'ready' : 'failed';
        validity.textContent = path.runnable ? 'Contract valid' : 'Blocked';
        byId('test-pipeline').disabled = !path.complete || !path.runnable;
        byId('open-pipeline-in-speak').disabled = !path.complete || !path.runnable;
        byId('compose-cli').textContent = cliRepresentation(path);
        const name = byId('compose-recipe-name');
        const userRecipe = state.userRecipes.find((candidate) => candidate.id === state.presetId);
        if (name && document.activeElement !== name) {
            const preset = (state.discovery.presets || []).find(
                (candidate) => candidate.composition_id === state.pathKey,
            );
            name.value = userRecipe?.name || preset?.display_name || path.display_name;
        }
        byId('delete-recipe').disabled = !userRecipe;
        if (!path.runnable) {
            showError(
                path.unavailable_reason || 'This pipeline is incomplete or unavailable.',
                byId('compose-error'),
            );
        } else {
            clearError(byId('compose-error'));
        }
        renderCompareCandidates();
        if (!state.restoringSpeechSelection) syncSpeechSelectionUrl();
    }

    function catalogFamily(item) {
        const catalog = item.catalog || [];
        if (item.backend === 'fairseq'
            || catalog.some((entry) => entry.provenance?.format === 'fairseq-mms-vits')) {
            return { id: 'mms', label: 'MMS' };
        }
        if (item.backend === 'styletts2'
            || catalog.some((entry) => /styletts/i.test(entry.architecture || ''))) {
            return { id: 'styletts2', label: 'StyleTTS2' };
        }
        const family = item.family ?? item.capabilities?.family;
        if (typeof family === 'string' && family) {
            return {
                id: 'other',
                label: family.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()),
            };
        }
        return {
            id: 'other',
            label: (item.backend || item.kind || 'Other').replaceAll('_', ' '),
        };
    }

    function catalogText(item) {
        const catalog = item.catalog || [];
        return [
            item.id,
            item.model,
            item.display_name,
            item.backend,
            item.kind,
            item.architecture,
            ...catalog.flatMap((entry) => [
                entry.id,
                entry.display_name,
                entry.architecture,
                entry.script,
                ...(entry.languages || []),
            ]),
        ].filter(Boolean).join(' ').toLocaleLowerCase();
    }

    function catalogLicense(item) {
        return [...new Set((item.catalog || []).map((entry) => entry.license?.expression).filter(Boolean))];
    }

    function matchesCatalogFilters(item) {
        const search = byId('catalog-search')?.value.trim().toLocaleLowerCase() || '';
        const family = byId('catalog-family')?.value || '';
        const license = byId('catalog-license')?.value || '';
        const capability = byId('catalog-capability')?.value || '';
        const verification = byId('catalog-verification')?.value || '';
        const device = byId('catalog-device')?.value || '';
        const capabilityText = [
            item.kind,
            item.stage,
            item.family,
            item.capabilities?.family,
            item.backend,
            ...(item.spans || []),
        ].filter(Boolean).join(' ').toLocaleLowerCase();
        const statuses = (item.statuses || []).join(' ').toLocaleLowerCase();
        const devices = [
            ...(item.devices || []),
            ...(item.capabilities?.devices || []),
            ...((item.controls || []).find(
                (control) => control.field === 'device',
            )?.options || []),
        ].map((entry) => typeof entry === 'string' ? entry : `${entry.value} ${entry.label}`)
            .join(' ').toLocaleLowerCase();
        const capabilityMatches = !capability
            || (capability === 'speech' && (
                item.pipeline || /acoustic|end.to.end|speech|voice/.test(capabilityText)
            ))
            || (capability === 'voice_conversion' && /conversion|clone/.test(capabilityText))
            || (capability === 'vocoder' && /vocoder/.test(capabilityText))
            || (capability === 'developer' && /trainer|developer|import|test/.test(capabilityText));
        const verificationMatches = !verification
            || (verification === 'verified' && /verified/.test(statuses))
            || (verification === 'pending' && /pending|changed/.test(statuses))
            || (verification === 'failed' && /failed|unavailable|missing/.test(statuses));
        return (!search || catalogText(item).includes(search))
            && (!family || catalogFamily(item).id === family)
            && (!license || catalogLicense(item).includes(license))
            && capabilityMatches
            && verificationMatches
            && (!device || devices.includes(device));
    }

    function statusBadges(statuses) {
        const badges = document.createElement('span');
        badges.className = 'status-badges';
        for (const status of statuses || []) {
            const badge = document.createElement('span');
            badge.className = 'status-badge';
            badge.textContent = status;
            badges.appendChild(badge);
        }
        return badges;
    }

    function missingCatalogIds(path) {
        return [...new Set(
            (path?.missing_catalog_ids || path?.catalog?.map((entry) => entry.id) || [])
                .filter(Boolean),
        )];
    }

    async function startCatalogFetch(path, label) {
        const modelIds = missingCatalogIds(path);
        if (!modelIds.length) {
            throw new Error('This catalog entry has no missing installable artifacts.');
        }
        const results = await Promise.all(modelIds.map(async (modelId) => {
            const response = await fetch('/api/jobs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    label: `Fetched ${label} · ${modelId}`,
                    command: 'cargo',
                    args: ['run', '--bin', 'tongues', '--', 'models', 'install', modelId],
                }),
            });
            if (!response.ok) throw new Error(`${modelId}: ${await response.text()}`);
            return response.json();
        }));
        navigateWorkflow('operate');
        await refreshOperateJobs();
        return results;
    }

    function catalogPipelineCard(composition, discovery) {
        const path = (discovery.paths || []).find((candidate) => (
            candidate.backend === composition.backend && candidate.model === composition.model
        )) || {};
        const card = document.createElement('article');
        card.className = 'catalog-pipeline-card';
        const heading = document.createElement('div');
        heading.className = 'catalog-card-heading';
        const title = document.createElement('strong');
        title.textContent = composition.display_name;
        heading.append(title, statusBadges(composition.statuses));
        const facts = document.createElement('p');
        const catalog = path.catalog || [];
        facts.textContent = [
            composition.backend,
            catalog.flatMap((entry) => entry.languages || []).join(', '),
            catalog.map((entry) => entry.license?.expression)
                .filter((license) => license && license !== 'NOASSERTION').join(', '),
        ].filter(Boolean).join(' · ');
        const actions = document.createElement('div');
        actions.className = 'model-actions';
        if (composition.runnable) {
            const selectComposition = () => {
                state.discovery = mergeSelectedResultIntoDiscovery(
                    state.discovery,
                    discovery,
                    composition,
                );
                state.pathKey = composition.id;
                state.presetId = '';
                renderPathSelector();
                renderSelectedPath();
            };
            const use = document.createElement('button');
            use.type = 'button';
            use.textContent = 'Use in Speak';
            use.addEventListener('click', () => {
                selectComposition();
                navigateWorkflow('speak');
            });
            actions.appendChild(use);
            const compose = document.createElement('button');
            compose.type = 'button';
            compose.className = 'secondary-button';
            compose.textContent = 'Open in Compose';
            compose.addEventListener('click', () => {
                selectComposition();
                navigateWorkflow('compose');
            });
            actions.appendChild(compose);
        } else if (path.install_command) {
            const install = document.createElement('button');
            install.type = 'button';
            install.className = 'secondary-button';
            install.textContent = 'Fetch';
            install.addEventListener('click', async () => {
                install.disabled = true;
                try {
                    await startCatalogFetch(path, composition.display_name);
                } catch (error) {
                    byId('catalog-status').textContent = `Fetch failed: ${error.message}`;
                    install.disabled = false;
                }
            });
            actions.appendChild(install);
        }
        if (path.installed && path.verification_status !== 'verified') {
            const verify = document.createElement('button');
            verify.type = 'button';
            verify.className = 'secondary-button';
            verify.textContent = 'Verify';
            verify.addEventListener('click', async () => {
                verify.disabled = true;
                try {
                    await verifyModelIds((path.catalog || []).map((entry) => entry.id));
                } catch (error) {
                    byId('catalog-status').textContent = `Verification failed: ${error.message}`;
                } finally {
                    verify.disabled = false;
                }
            });
            actions.appendChild(verify);
        }
        card.append(heading, facts, actions);
        return card;
    }

    function catalogComponentCard(component) {
        const details = document.createElement('details');
        details.className = 'component-card';
        const summary = document.createElement('summary');
        const name = document.createElement('strong');
        name.textContent = component.display_name;
        summary.append(name, statusBadges(component.statuses));
        const body = document.createElement('div');
        body.className = 'component-detail';
        const explanation = document.createElement('p');
        explanation.textContent = component.explanation;
        const catalog = component.catalog || [];
        const preprocessing = [...new Set(catalog.flatMap((entry) => entry.preprocessing || []))];
        const licenses = [...new Set(catalog.map((entry) => entry.license?.expression).filter(Boolean))];
        const facts = document.createElement('p');
        facts.textContent = [
            `Architecture: ${component.architecture}`,
            `State: ${component.readiness}`,
            `Load: ${component.load_state}`,
            component.control_fields?.length ? `Controls: ${component.control_fields.join(', ')}` : null,
            catalog.length ? `Language: ${[...new Set(catalog.flatMap((entry) => entry.languages || []))].join(', ')}` : null,
            catalog.some((entry) => entry.script)
                ? `Script: ${[...new Set(catalog.map((entry) => entry.script).filter(Boolean))].join(', ')}`
                : null,
            `Preprocessing: ${preprocessing.join(', ') || 'No additional preprocessing required'}`,
            licenses.length
                ? `License: ${licenses.map((license) => (
                    license === 'NOASSERTION' ? 'Metadata unavailable; check upstream before redistribution' : license
                )).join(', ')}`
                : null,
            component.compatible_paths.length ? `Paths: ${component.compatible_paths.join(', ')}` : null,
        ].filter(Boolean).join(' · ');
        body.append(explanation, facts);
        if (component.install_command) {
            const code = document.createElement('code');
            code.textContent = component.install_command;
            body.appendChild(code);
        }
        details.append(summary, body);
        return details;
    }

    function renderInventory() {
        const discovery = state.catalogDiscovery || state.discovery;
        const target = byId('component-inventory');
        const view = state.catalogView;
        const items = view === 'components'
            ? (discovery.components || []).filter(matchesCatalogFilters)
            : (discovery.compositions || []).filter((composition) => {
                const path = (discovery.paths || []).find((candidate) => (
                    candidate.backend === composition.backend
                    && candidate.model === composition.model
                ));
                return composition.backend !== 'mock' && matchesCatalogFilters({
                    ...composition,
                    catalog: path?.catalog || [],
                })
                    && (view === 'ready'
                        ? composition.runnable
                        : !composition.runnable && Boolean(path?.catalog?.length));
            });
        const groups = new Map();
        for (const item of items) {
            const family = view === 'components'
                ? {
                    id: `${catalogFamily(item).id}:${item.kind}`,
                    label: `${catalogFamily(item).label} · ${item.kind.replaceAll('_', ' ')}`,
                }
                : catalogFamily(item);
            const group = groups.get(family.id) || { label: family.label, items: [] };
            group.items.push(item);
            groups.set(family.id, group);
        }
        target.replaceChildren();
        let groupIndex = 0;
        for (const group of groups.values()) {
            const family = document.createElement('details');
            family.className = 'catalog-family';
            family.open = view === 'ready' && groupIndex === 0;
            const summary = document.createElement('summary');
            summary.textContent = `${group.label} (${group.items.length})`;
            const cards = document.createElement('div');
            cards.className = 'catalog-family-items';
            cards.replaceChildren(...group.items.map((item) => (
                view === 'components'
                    ? catalogComponentCard(item)
                    : catalogPipelineCard(item, discovery)
            )));
            family.append(summary, cards);
            target.appendChild(family);
            groupIndex += 1;
        }
        if (!items.length) {
            const empty = document.createElement('p');
            empty.className = 'catalog-empty';
            empty.textContent = view === 'ready'
                ? 'No ready pipelines match these filters.'
                : 'No catalog entries match these filters.';
            target.appendChild(empty);
        }
        const total = discovery.page?.total;
        byId('component-count').textContent = `(${items.length} shown)`;
        byId('catalog-status').textContent = total == null
            ? `${items.length} matching entries`
            : `${items.length} matching entries loaded · ${total.toLocaleString()} catalog models`;
        const verifyAll = byId('verify-all-models');
        const pending = pendingVerificationIds(state.discovery);
        verifyAll.disabled = pending.length === 0;
        verifyAll.textContent = pending.length
            ? `Verify all changed models (${pending.length})`
            : 'All installed models verified';
    }

    function renderRuntime(runtime) {
        const badge = byId('speech-runtime-state');
        const stateName = runtime.state || 'unknown';
        badge.dataset.state = stateName;
        badge.textContent = stateName;
        const values = [
            ['Device', `${runtime.device || 'unknown'}${Number.isInteger(runtime.device_index) ? ` ${runtime.device_index}` : ''}`],
            ['Requests', `${runtime.active ?? 0} active · ${runtime.queued ?? 0} queued`],
            ['Concurrency', `${runtime.capacity ?? 0} maximum · ${runtime.concurrency || 'unknown'}`],
            ['Warm paths', (runtime.loaded || []).join(', ') || 'None; models load on first synthesis'],
        ];
        const grid = byId('speech-runtime-grid');
        grid.replaceChildren(...values.map(([label, value]) => metadataItem(label, value)));
        const failures = Object.entries(runtime.failed || {});
        const error = byId('speech-runtime-errors');
        if (failures.length) {
            showError(failures.map(([engine, message]) => `${engine}: ${message}`).join(' · '), error);
        } else {
            clearError(error);
        }
    }

    async function loadRuntime(signal) {
        const response = await fetch('/api/speech/runtime', { cache: 'no-store', signal });
        if (!response.ok) throw new Error(await response.text());
        const runtime = await response.json();
        renderRuntime(runtime);
        return runtime;
    }

    function stopRuntimePolling() {
        state.runtimePollGeneration += 1;
        if (state.runtimeTimer != null) {
            window.clearTimeout(state.runtimeTimer);
            state.runtimeTimer = null;
        }
        state.runtimePollController?.abort();
        state.runtimePollController = null;
    }

    function startRuntimePolling() {
        stopRuntimePolling();
        const generation = state.runtimePollGeneration;
        const poll = async () => {
            if (generation !== state.runtimePollGeneration) return;
            const controller = new AbortController();
            state.runtimePollController = controller;
            try {
                await loadRuntime(controller.signal);
            } catch (error) {
                if (error.name !== 'AbortError') {
                    showError(
                        `Runtime status unavailable: ${error.message}`,
                        byId('speech-runtime-errors'),
                    );
                }
            } finally {
                if (state.runtimePollController === controller) {
                    state.runtimePollController = null;
                }
                if (generation === state.runtimePollGeneration) {
                    // Delay from completion so slow runtime requests never overlap.
                    state.runtimeTimer = window.setTimeout(poll, 750);
                }
            }
        };
        poll();
    }

    function metadataItem(label, value) {
        const wrapper = document.createElement('div');
        const term = document.createElement('dt');
        term.textContent = label;
        const detail = document.createElement('dd');
        detail.textContent = String(value ?? '—');
        wrapper.append(term, detail);
        return wrapper;
    }

    function renderResult(metadata, url) {
        byId('audio-player').src = url;
        const download = byId('speech-download');
        download.href = metadata.audio_url || url;
        download.download = `tongues-${metadata.backend || 'speech'}.wav`;
        const runLink = byId('speech-run-link');
        runLink.classList.toggle('hidden', !metadata.run_url);
        if (metadata.run_url) runLink.href = metadata.run_url;
        const fields = [
            ['Run', metadata.run_id],
            ['Pipeline', metadata.pipeline_id || metadata.path],
            ['Backend', metadata.backend],
            ['Projector', metadata.projector],
            ['Acoustic model', metadata.acoustic_model],
            ['Conditioners', (metadata.conditioners || []).join(', ')],
            ['Vocoder', metadata.vocoder],
            ['Voice / model', metadata.voice_model],
            ['Speaker / reference', metadata.speaker || metadata.reference_voice],
            ['Variety', metadata.variety],
            ['Device', `${metadata.device || 'unknown'}${metadata.device_index == null ? '' : ` ${metadata.device_index}`}`],
            ['Audio', `${metadata.sample_rate_hz} Hz · ${metadata.channels} channel · ${metadata.sample_count} samples`],
            ['Duration', `${Number(metadata.duration_seconds || 0).toFixed(3)} s`],
            ['Queue time', `${Number(metadata.queue_ms || 0).toFixed(2)} ms`],
            ['Model load', `${Number(metadata.model_load_ms || 0).toFixed(2)} ms`],
            ['Synthesis', `${Number(metadata.synthesis_ms || 0).toFixed(2)} ms`],
            ['Real-time factor', Number(metadata.real_time_factor || 0).toFixed(3)],
            ['Resident model', metadata.resident_model_reused ? 'Reused' : 'Loaded for this request'],
            ['Input retention', metadata.text_retained === false ? 'Input text not retained' : null],
            ['Content diagnostics', metadata.content_detail_retained === false
                ? 'Content-derived details not retained'
                : null],
        ].filter(([, value]) => value != null && value !== '');
        byId('speech-result-metadata').replaceChildren(
            ...fields.map(([label, value]) => metadataItem(label, value)),
        );
        const diagnostics = metadata.diagnostics || {};
        const hasDiagnostics = Object.values(diagnostics).some((value) => (
            Array.isArray(value) ? value.length : value != null
        ));
        byId('speech-result-diagnostics').classList.toggle('hidden', !hasDiagnostics);
        byId('speech-diagnostics-output').textContent = JSON.stringify(diagnostics, null, 2);
        byId('result-container').classList.remove('hidden');
    }

    function duplexLines(source) {
        return String(source || '')
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter(Boolean);
    }

    function buildDuplexRequest({
        text = '',
        mockAcoustics = '',
        variety = null,
        journalPath = '',
        fixture = '',
        evidenceCursor = 0,
        evidenceLimit = 20,
        evidenceTargetId = '',
    } = {}) {
        const replayPath = String(journalPath || '').trim();
        const fixtureId = String(fixture || '').trim();
        let payload;
        if (replayPath) {
            payload = { journal_path: replayPath };
        } else if (fixtureId) {
            payload = { fixture: fixtureId };
        } else {
            const chunks = duplexLines(text);
            const mockChunks = duplexLines(mockAcoustics);
            if (!chunks.length && !mockChunks.length) {
                throw new Error('Enter prompt text, mock acoustic chunks, or a saved journal path.');
            }
            payload = {
                chunks,
                mock_acoustics: mockChunks,
                variety,
            };
        }
        if (Number(evidenceCursor) > 0) payload.evidence_cursor = Number(evidenceCursor);
        if (Number(evidenceLimit) !== 20) payload.evidence_limit = Number(evidenceLimit);
        const targetId = String(evidenceTargetId || '').trim();
        if (targetId) payload.evidence_target_id = targetId;
        return payload;
    }

    function duplexRequestFromLocation(locationLike) {
        const params = new URLSearchParams(locationLike?.search || '');
        const shared = {
            evidenceCursor: Number(params.get('duplex_cursor') || 0),
            evidenceLimit: Number(params.get('duplex_limit') || 20),
            evidenceTargetId: params.get('duplex_target') || '',
        };
        const journalPath = params.get('duplex_journal');
        if (journalPath) return buildDuplexRequest({ journalPath, ...shared });
        const fixture = params.get('duplex_fixture');
        if (fixture) return buildDuplexRequest({ fixture, ...shared });
        return null;
    }

    function duplexDeepLinkWith(link, changes = {}) {
        if (!link) return '';
        const url = new URL(link, 'http://localhost');
        for (const [key, value] of Object.entries(changes)) {
            if (value == null || value === '' || value === 0) url.searchParams.delete(key);
            else url.searchParams.set(key, String(value));
        }
        return `${url.pathname}${url.search}`;
    }

    function duplexTokenMarkup(tokens, kind) {
        if (!tokens?.length) return '<span class="duplex-empty">—</span>';
        return tokens.map((token) => (
            `<span class="duplex-token duplex-token-${kind}">${escapeHtml(token)}</span>`
        )).join('');
    }

    function duplexSurfaceList(items) {
        return (items || []).map((item) => (
            typeof item === 'string' ? item : (item?.surface || item?.key || '')
        )).filter(Boolean);
    }

    function formatDuplexTranscriptEvent(event) {
        if (!event) return 'No transcript delta';
        if (event.type === 'partial_hypothesis') return `Provisional: ${event.data?.text || '—'}`;
        if (event.type === 'hypothesis_cancelled') return `Withdraw: ${event.data?.reason || '—'}`;
        if (event.type === 'committed_segment') return `Commit: ${event.data?.text || '—'}`;
        if (event.type === 'revised_hypothesis') {
            const range = event.data?.replaces;
            return `Replace ${range?.start ?? '?'}..${range?.end ?? '?'}: ${event.data?.text || '—'}`;
        }
        return event.type;
    }

    function humanizeEvidenceValue(value) {
        if (value == null) return 'unknown';
        if (typeof value !== 'object') return String(value);
        if (value.type && Object.hasOwn(value, 'value')) {
            return `${value.type}: ${JSON.stringify(value.value)}`;
        }
        return JSON.stringify(value);
    }

    function interpretationClaimMarkup(claim, label) {
        if (!claim) return `
            <div class="evidence-claim evidence-unknown">
                <strong>${escapeHtml(label)}</strong>
                <span>Unknown — no eligible evidence winner.</span>
            </div>`;
        const calibration = claim.calibration || 'uncalibrated';
        const conflict = claim.conflicts_with_winner
            ? '<span class="evidence-conflict">Conflicts with winner</span>'
            : '';
        return `
            <article class="evidence-claim" data-authority="${escapeAttribute(claim.authority)}">
                <div class="evidence-claim-heading">
                    <strong>${escapeHtml(label)} · ${escapeHtml(claim.claim_id)}</strong>
                    <span>${escapeHtml(claim.authority)}</span>
                </div>
                <p>${escapeHtml(humanizeEvidenceValue(claim.value))}</p>
                <dl class="metadata-grid compact">
                    <div><dt>Source</dt><dd>${escapeHtml(`${claim.provenance?.source || 'unknown'} · ${claim.provenance?.method || 'unknown'}`)}</dd></div>
                    <div><dt>Confidence</dt><dd>${Number(claim.confidence).toFixed(3)} · ${escapeHtml(calibration)}</dd></div>
                    <div><dt>Lifecycle</dt><dd>${escapeHtml(claim.lifecycle)}</dd></div>
                    <div><dt>Reason</dt><dd>${escapeHtml(claim.resolution_explanation || claim.rationale || 'Unknown')}</dd></div>
                </dl>
                ${conflict}
                ${claim.supports?.length ? `<p>Supports: ${escapeHtml(claim.supports.join(', '))}</p>` : ''}
                ${claim.conflicts_with?.length ? `<p>Conflicts: ${escapeHtml(claim.conflicts_with.join(', '))}</p>` : ''}
            </article>`;
    }

    function scoreComponentMarkup(score = {}) {
        const available = new Set(score.available_components || []);
        const components = [
            ['acoustic_likelihood', 'Acoustic'],
            ['provider_prior', 'Provider'],
            ['lexical_evidence', 'Lexical'],
            ['grammar_parse_rank', 'Grammar'],
            ['prosody_compatibility', 'Prosody'],
            ['user_markup', 'User'],
            ['direct_observation', 'Direct'],
        ];
        return components.map(([key, label]) => (
            `<span>${escapeHtml(label)}: ${available.has(key) ? Number(score[key]).toFixed(3) : 'unknown'}</span>`
        )).join('');
    }

    function interpretationTargetMarkup(target, deepLink) {
        const winner = interpretationClaimMarkup(target.winner, 'Won');
        const alternatives = (target.alternatives || []).map((claim) => (
            interpretationClaimMarkup(claim, 'Alternative')
        )).join('');
        const consequences = (target.consequences || []).map((consequence) => `
            <article class="evidence-consequence">
                <strong>${escapeHtml(consequence.hypothesis_id)} · ${consequence.selected ? 'selected' : 'not selected'}</strong>
                <p>Output: ${escapeHtml(consequence.output_text || 'unknown')}</p>
                <div class="duplex-score-components">${scoreComponentMarkup(consequence.score)}</div>
                <p>States: ${escapeHtml((consequence.statuses || []).join(', ') || 'candidate')}</p>
                <p>Commit blocks: ${escapeHtml(JSON.stringify(consequence.block_reasons || []))}</p>
            </article>`).join('');
        const audio = (target.acoustic_links || []).map((link) => `
            <li>${escapeHtml(link.evidence_id)} · frames ${link.span.frame_start}–${link.span.frame_end}
                · ${Number(link.span.time_start).toFixed(3)}–${Number(link.span.time_end).toFixed(3)} s
                · ${escapeHtml(link.alignment)}</li>`).join('');
        const targetLink = duplexDeepLinkWith(deepLink, {
            duplex_target: target.target_id,
            duplex_cursor: null,
        });
        return `
            <details class="evidence-target" data-target-id="${escapeAttribute(target.target_id)}">
                <summary>
                    <strong>${escapeHtml(target.kind)} · ${escapeHtml(target.status)}</strong>
                    <span>${escapeHtml(target.target_id)}</span>
                </summary>
                ${targetLink ? `<a class="evidence-deep-link" href="${escapeAttribute(targetLink)}">Permanent link to this target</a>` : '<p>Permanent link unavailable until this run has a saved journal.</p>'}
                ${winner}
                ${alternatives || '<p>No close alternatives were retained for this target.</p>'}
                <details>
                    <summary>Linked spans and output consequences</summary>
                    <p>Linked claims: ${escapeHtml((target.linked_claim_ids || []).join(', ') || 'none')}</p>
                    ${audio ? `<ul>${audio}</ul>` : '<p>Audio frame evidence: unknown.</p>'}
                    ${consequences || '<p>Downstream output consequence: unknown.</p>'}
                </details>
            </details>`;
    }

    function inspectionDiagnosticsMarkup(inspection) {
        const backends = (inspection.backend_reports || []).map((backend) => `
            <li>
                <strong>${escapeHtml(backend.hypothesis_id)} · ${escapeHtml(backend.status)}</strong>
                <span>selected backend: ${escapeHtml(backend.report?.selected || 'none')}</span>
                <span>${escapeHtml(backend.diagnostic || 'No backend diagnostic')}</span>
                <span>${backend.parse_alternatives?.length || 0} parse alternatives</span>
            </li>`).join('');
        const lifecycle = (inspection.lifecycle || []).map((transition) => `
            <li>${escapeHtml(transition.claim_id)} · ${escapeHtml(transition.from)}
                → ${escapeHtml(transition.to)} · ${escapeHtml(transition.reason)}</li>`).join('');
        const losses = (inspection.projection_losses || []).map((loss) => `
            <li>${escapeHtml(loss.evidence.dimension)} · ${escapeHtml(loss.evidence.intended)}
                → ${escapeHtml(loss.evidence.recovered)} · accepted projection loss</li>`).join('');
        const warnings = (inspection.warnings || []).map((warning) => `
            <li><strong>${escapeHtml(warning.code)}</strong> · ${escapeHtml(warning.message)}</li>`
        ).join('');
        return `
            <details class="evidence-diagnostics">
                <summary>Backend, revision, projection-loss, and migration diagnostics</summary>
                <h4>Backend state</h4>
                ${backends ? `<ul>${backends}</ul>` : '<p>Backend evidence: unknown.</p>'}
                <h4>Revision history</h4>
                ${lifecycle ? `<ol>${lifecycle}</ol>` : '<p>No claim lifecycle changes on this page.</p>'}
                <h4>Projection loss</h4>
                ${losses ? `<ul>${losses}</ul>` : '<p>No accepted projection loss recorded.</p>'}
                <h4>Warnings</h4>
                ${warnings ? `<ul>${warnings}</ul>` : '<p>No migration or truncation warnings.</p>'}
            </details>`;
    }

    function renderInterpretationInspection(projection) {
        const section = byId('duplex-evidence');
        const status = byId('duplex-evidence-status');
        const targets = byId('duplex-evidence-targets');
        const pages = byId('duplex-evidence-pages');
        const inspection = projection?.interpretation;
        if (!inspection) {
            status.textContent = 'Interpretation evidence contract is missing from this response.';
            targets.replaceChildren();
            pages.replaceChildren();
            section.classList.remove('hidden');
            return;
        }
        if (inspection.evidence_status === 'missing') {
            status.textContent = inspection.warnings?.[0]?.message
                || 'No linguistic evidence was attached; alternatives and confidence are unknown.';
            targets.replaceChildren();
        } else {
            status.textContent = `Showing ${inspection.returned} of ${inspection.total} evidence targets for ${inspection.utterance_id}.`;
            const targetMarkup = (inspection.targets || []).map((target) => (
                interpretationTargetMarkup(target, projection.deep_link)
            )).join('') || '<p>No target matched this deep link.</p>';
            targets.innerHTML = targetMarkup + inspectionDiagnosticsMarkup(inspection);
        }
        const links = [];
        if (inspection.cursor > 0 && projection.deep_link) {
            links.push(`<a href="${escapeAttribute(duplexDeepLinkWith(projection.deep_link, {
                duplex_cursor: Math.max(0, inspection.cursor - inspection.limit),
                duplex_target: null,
            }))}">Previous evidence page</a>`);
        }
        if (inspection.next_cursor != null && projection.deep_link) {
            links.push(`<a href="${escapeAttribute(duplexDeepLinkWith(projection.deep_link, {
                duplex_cursor: inspection.next_cursor,
                duplex_target: null,
            }))}">Next evidence page</a>`);
        }
        pages.innerHTML = links.join(' ');
        section.classList.remove('hidden');
    }

    function renderDuplexProjection(projection) {
        if (projection?.schema_version !== 2) {
            throw new Error(`Unsupported duplex inspection schema ${projection?.schema_version ?? 'missing'}.`);
        }
        const timeline = byId('duplex-timeline');
        const summary = byId('duplex-summary');
        const finalCommitted = projection?.final_state?.committed || [];
        summary.innerHTML = `
            <dl class="metadata-grid">
                <div><dt>Run</dt><dd>${escapeHtml(projection.run_id)}</dd></div>
                <div><dt>Replay</dt><dd>${projection.replay_verified ? 'Verified' : 'Mismatch'}</dd></div>
                <div><dt>Events</dt><dd>${projection.timeline?.length || 0}</dd></div>
                <div><dt>Final commit</dt><dd>${escapeHtml(finalCommitted.map((item) => item.surface).join(' ') || '—')}</dd></div>
            </dl>
            ${projection.deep_link ? `<a class="evidence-deep-link" href="${escapeAttribute(projection.deep_link)}">Permanent link to this run and utterance</a>` : '<p>Permanent link unavailable for unsaved ad hoc evidence.</p>'}`;
        summary.classList.remove('hidden');
        timeline.innerHTML = (projection.timeline || []).map((snapshot) => `
            <li class="duplex-step">
                <div class="duplex-step-heading">
                    <strong>#${snapshot.sequence} ${escapeHtml(snapshot.layer)}</strong>
                    <span>${escapeHtml(snapshot.event_type)}</span>
                </div>
                <p class="duplex-step-message">${escapeHtml(snapshot.message)}</p>
                <div class="duplex-track"><span>Observed</span>${duplexTokenMarkup(snapshot.observed, 'observed')}</div>
                <div class="duplex-track"><span>Shared</span>${duplexTokenMarkup(snapshot.shared_prefix, 'inferred')}</div>
                <div class="duplex-track"><span>Committed</span>${duplexTokenMarkup(snapshot.committed, 'committed')}</div>
                <div class="duplex-track"><span>Predicted</span>${duplexTokenMarkup(snapshot.predicted, 'predicted')}</div>
                <div class="duplex-step-meta">
                    <span>Frontier ${snapshot.commit_frontier}</span>
                    <span>${snapshot.first_divergent_morpheme_index == null ? 'No branch divergence' : `First divergent stage ${snapshot.first_divergent_morpheme_index}`}</span>
                    <span>${escapeHtml(formatDuplexTranscriptEvent(snapshot.transcript_event))}</span>
                </div>
                <details class="duplex-branches">
                    <summary>Branches (${snapshot.branches.length})</summary>
                    <ul>
                        ${(snapshot.branches || []).map((branch) => `
                            <li>
                                <strong>${escapeHtml(branch.hypothesis_id)}</strong>
                                <span>${branch.selected ? 'selected' : 'standby'} · p=${Number(branch.probability || 0).toFixed(3)}</span>
                                <span>${escapeHtml(branch.provenance)}</span>
                                <div class="duplex-track">${duplexTokenMarkup(branch.morphemes, branch.selected ? 'predicted' : 'observed')}</div>
                                <details>
                                    <summary>Score and evidence references</summary>
                                    <div class="duplex-score-components">${scoreComponentMarkup(branch.score)}</div>
                                    <p>Claims: ${escapeHtml((branch.claim_ids || []).join(', ') || 'unknown')}</p>
                                    <p>Resolutions: ${escapeHtml((branch.resolution_ids || []).join(', ') || 'unknown')}</p>
                                    <p>Commit blocks: ${escapeHtml(JSON.stringify(branch.block_reasons || []))}</p>
                                </details>
                            </li>
                        `).join('')}
                    </ul>
                </details>
            </li>
        `).join('');
        timeline.classList.remove('hidden');
        renderInterpretationInspection(projection);
    }

    async function loadDuplexProjection(payload) {
        const response = await fetch('/api/duplex/project', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!response.ok) throw new Error(await response.text());
        const projection = await response.json();
        state.duplexRequest = structuredClone(payload);
        return projection;
    }

    function escapeHtml(value) {
        return String(value ?? '').replace(/[&<>"']/g, (character) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
        }[character]));
    }

    function escapeAttribute(value) {
        return escapeHtml(value).replace(/`/g, '&#96;');
    }

    async function loadAuxiliaryDiscovery() {
        const [sampleResponse, emotionResponse, audioInputResponse, languageRoutingResponse] = await Promise.allSettled([
            fetch('/api/styletts2-samples').then((response) => response.json()),
            fetch('/api/emotions').then((response) => response.json()),
            fetch('/api/audio-input/capabilities', { cache: 'no-store' })
                .then((response) => {
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    return response.json();
                }),
            fetch('/api/language-routing/capabilities', { cache: 'no-store' })
                .then((response) => {
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    return response.json();
                }),
        ]);
        state.samples = sampleResponse.status === 'fulfilled'
            ? (sampleResponse.value.samples || [])
            : [];
        state.sampleDefaults = sampleResponse.status === 'fulfilled'
            ? (sampleResponse.value.defaults || {})
            : {};
        state.emotions = emotionResponse.status === 'fulfilled'
            ? (emotionResponse.value.emotions || [])
            : [];
        state.audioInputCapabilities = audioInputResponse.status === 'fulfilled'
            ? audioInputResponse.value
            : null;
        state.languageRoutingCapabilities = languageRoutingResponse.status === 'fulfilled'
            ? languageRoutingResponse.value
            : null;
    }

    function acceptDiscovery(discovery, allowVerificationReset = false, append = false) {
        const candidate = append ? mergeDiscovery(state.discovery, discovery) : discovery;
        if (
            !allowVerificationReset
            && !preservesVerificationProgress(state.discovery, candidate)
        ) return false;
        const previous = state.pathKey;
        state.discovery = candidate;
        const previousMissing = (
            !(state.discovery.compositions || []).some((composition) => composition.id === previous)
            && !(state.discovery.paths || []).some((path) => pathKey(path) === previous)
        );
        const previousMayArrive = Boolean(previous && state.discovery.page?.next_cursor != null);
        if (previousMissing && !previousMayArrive) {
            const initial = selectInitialComposition(state.discovery)
                || selectInitialPath(state.discovery);
            state.pathKey = initial?.pipeline ? initial.id : (initial ? pathKey(initial) : '');
            const preset = (state.discovery.presets || []).find(
                (candidate) => candidate.composition_id === state.pathKey,
            );
            state.presetId = preset?.id || '';
        }
        renderSpeechSamples();
        renderPathSelector();
        renderInventory();
        renderSelectedPath();
        return true;
    }

    async function verifyModelIds(modelIds, generation = null) {
        const ids = [...new Set((modelIds || []).filter(Boolean))];
        const activeGeneration = generation ?? (state.verificationGeneration + 1);
        if (generation == null) state.verificationGeneration = activeGeneration;
        let cursor = 0;
        const failures = [];
        const verifyNext = async () => {
            while (activeGeneration === state.verificationGeneration && cursor < ids.length) {
                const modelId = ids[cursor];
                cursor += 1;
                try {
                    const response = await fetch(
                        `/api/speech/models/verify/${encodeURIComponent(modelId)}`,
                        { method: 'POST', cache: 'no-store' },
                    );
                    if (!response.ok) throw new Error(await response.text());
                } catch (error) {
                    failures.push(`${modelId}: ${error.message}`);
                }
            }
        };
        await Promise.all(Array.from(
            { length: Math.min(VERIFICATION_CONCURRENCY, ids.length) },
            verifyNext,
        ));
        if (activeGeneration === state.verificationGeneration && ids.length) {
            try {
                const updated = await fetchDiscoveryPage();
                if (activeGeneration === state.verificationGeneration) {
                    acceptDiscovery(updated, true);
                    await refreshCatalog({ fresh: true });
                }
            } catch (error) {
                failures.push(`final refresh: ${error.message}`);
            }
        }
        if (activeGeneration === state.verificationGeneration && failures.length) {
            throw new Error(failures.join(' · '));
        }
    }

    async function verifyDiscovery(generation, discovery) {
        try {
            await verifyModelIds(pendingVerificationIds(discovery), generation);
        } catch (error) {
            showError(`Background model verification failed: ${error.message}`);
        }
    }

    function discoveryRequestUrl(cursor = 0, limit = null, filters = {}) {
        const query = new URLSearchParams();
        if (cursor) query.set('cursor', String(cursor));
        if (limit) query.set('limit', String(limit));
        for (const [name, value] of Object.entries(filters)) {
            if (value) query.set(name, value);
        }
        return `/api/speech/models${query.size ? `?${query}` : ''}`;
    }

    async function readCachedCatalogPage(url) {
        const memoryEntry = catalogPageCache.get(url);
        if (memoryEntry && Date.now() - memoryEntry.cachedAt < CATALOG_CACHE_TTL_MS) {
            return memoryEntry.discovery;
        }
        catalogPageCache.delete(url);
        if (!browser?.caches) return null;
        try {
            const cache = await browser.caches.open(CATALOG_CACHE_NAME);
            const response = await cache.match(url);
            if (!response) return null;
            const entry = await response.json();
            if (
                !entry?.discovery
                || !Number.isFinite(entry.cached_at)
                || Date.now() - entry.cached_at >= CATALOG_CACHE_TTL_MS
            ) {
                await cache.delete(url);
                return null;
            }
            catalogPageCache.set(url, {
                cachedAt: entry.cached_at,
                discovery: entry.discovery,
            });
            return entry.discovery;
        } catch (_) {
            return null;
        }
    }

    async function cacheCatalogPage(url, discovery) {
        const cachedAt = Date.now();
        catalogPageCache.set(url, { cachedAt, discovery });
        if (!browser?.caches) return;
        try {
            const cache = await browser.caches.open(CATALOG_CACHE_NAME);
            await cache.put(url, new Response(JSON.stringify({
                cached_at: cachedAt,
                discovery,
            }), {
                headers: { 'Content-Type': 'application/json' },
            }));
        } catch (_) {
            // Catalog browsing must still work when browser cache storage is unavailable.
        }
    }

    async function fetchDiscoveryPage(
        cursor = 0,
        limit = null,
        filters = {},
        { cacheCatalog = false, refreshCache = false } = {},
    ) {
        const url = discoveryRequestUrl(cursor, limit, filters);
        if (cacheCatalog && !refreshCache) {
            const cached = await readCachedCatalogPage(url);
            if (cached) return cached;
        }
        const response = await fetch(
            url,
            { cache: 'no-store' },
        );
        if (!response.ok) throw new Error(await response.text());
        const discovery = await response.json();
        if (discovery.error && !(discovery.paths || []).length) {
            throw new Error(discovery.error);
        }
        if (cacheCatalog) await cacheCatalogPage(url, discovery);
        return discovery;
    }

    function catalogFilters() {
        return {
            search: byId('catalog-search')?.value.trim() || '',
            family: byId('catalog-family')?.value || '',
            license: byId('catalog-license')?.value || '',
            capability: byId('catalog-capability')?.value || '',
            verification: byId('catalog-verification')?.value || '',
            device: byId('catalog-device')?.value || '',
        };
    }

    function savedRecipeModelIds(discovery, recipes) {
        const present = new Set((discovery?.paths || []).map((path) => path.model));
        return [...new Set((recipes || []).flatMap((recipe) => [
            recipe.model,
            recipe.pipeline?.end_to_end,
            recipe.pipeline?.acoustic_model,
        ]).filter((id) => id && !present.has(id)))];
    }

    async function hydrateSavedRecipeDiscovery() {
        const modelIds = savedRecipeModelIds(state.discovery, state.userRecipes);
        if (!modelIds.length) return;
        const saved = await fetchDiscoveryPage(0, Math.max(32, modelIds.length), {
            model_ids: modelIds.join(','),
        });
        state.discovery = mergeInventoryDiscovery(state.discovery, saved);
        renderPathSelector();
        renderSelectedPath();
        renderCompareCandidates();
    }

    async function refreshCatalog({ append = false, fresh = false } = {}) {
        const generation = state.catalogRequestGeneration + 1;
        state.catalogRequestGeneration = generation;
        let cursor = append ? state.catalogDiscovery?.page?.next_cursor : 0;
        if (append && cursor == null) return;
        let merged = append ? state.catalogDiscovery : null;
        const filters = catalogFilters();
        while (cursor != null) {
            byId('catalog-status').textContent = cursor
                ? 'Loading the remaining catalog models automatically…'
                : 'Searching the model catalog…';
            const page = await fetchDiscoveryPage(cursor, CATALOG_PAGE_SIZE, filters, {
                cacheCatalog: true,
                refreshCache: fresh,
            });
            if (generation !== state.catalogRequestGeneration) return;
            merged = mergeDiscovery(merged, page);
            state.catalogDiscovery = merged;
            renderInventory();
            cursor = page.page?.next_cursor;
            if (cursor != null) {
                const loaded = Math.min(
                    page.page.total,
                    page.page.cursor + page.page.returned,
                );
                byId('catalog-status').textContent = (
                    `Loading catalog models automatically… ${loaded.toLocaleString()}`
                    + ` of ${page.page.total.toLocaleString()} cached`
                );
            }
        }
        if (generation === state.catalogRequestGeneration && merged) {
            state.discovery = mergeInventoryDiscovery(state.discovery, merged);
            renderSpeechSamples();
            renderPathSelector();
            renderSelectedPath();
        }
    }

    async function refreshDiscovery(verifyChanged = false) {
        const generation = state.verificationGeneration + 1;
        state.verificationGeneration = generation;
        const firstPage = await fetchDiscoveryPage();
        acceptDiscovery(firstPage, true);
        state.catalogDiscovery = firstPage;
        renderInventory();
        refreshCatalog({ append: true }).catch((error) => {
            byId('catalog-status').textContent = `Catalog loading failed: ${error.message}`;
        });
        await hydrateSavedRecipeDiscovery();
        if (verifyChanged) await verifyDiscovery(generation, firstPage);
    }

    function hideDuplexResult() {
        byId('duplex-summary').classList.add('hidden');
        byId('duplex-evidence').classList.add('hidden');
        byId('duplex-timeline').classList.add('hidden');
    }

    function pathForComposition(composition, discovery = state.discovery) {
        if (!composition) return null;
        const legacy = (discovery?.paths || []).find((path) => (
            path.backend === composition.backend && path.model === composition.model
        )) || {};
        return {
            ...legacy,
            ...(composition.capabilities || {}),
            ...composition,
            id: composition.model,
            complete: true,
            acoustic_model: composition.pipeline.acoustic_model,
            vocoder: composition.pipeline.vocoder,
            voice_model: composition.pipeline.end_to_end,
        };
    }

    async function synthesisPayload(path, values, context) {
        const payload = buildPayload(path, values, context);
        payload.recipe_id = state.presetId || null;
        payload.composition_id = state.pathKey || null;
        const tokenFields = ['pitch', 'energy', 'durations']
            .filter((field) => Array.isArray(payload[field]));
        let projection = null;
        if (tokenFields.length) {
            const projectionResponse = await fetch('/api/speech/project', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: payload.text,
                    variety: payload.variety,
                    ...(payload.pipeline
                        ? { pipeline: payload.pipeline }
                        : { backend: payload.backend }),
                }),
            });
            if (!projectionResponse.ok) throw new Error(await projectionResponse.text());
            projection = await projectionResponse.json();
            for (const field of tokenFields) {
                if (payload[field].length !== projection.projected_token_count) {
                    throw new Error(
                        `${field} has ${payload[field].length} values, but the selected model projects ${projection.projected_token_count} tokens.`,
                    );
                }
            }
        }
        return { payload, projection };
    }

    async function restoreSpeechSelectionFromLocation() {
        const requested = speechSelectionFromLocation();
        if (!requested.runId && !requested.compositionId && !requested.recipeId) return false;
        let record = null;
        if (requested.runId) {
            const response = await fetch(`/api/speech/runs/${encodeURIComponent(requested.runId)}`, {
                cache: 'no-store',
            });
            if (!response.ok) throw new Error(await response.text());
            record = await response.json();
        }
        const selection = record?.selection || {};
        const modelId = requested.modelId
            || selection.model
            || selection.pipeline?.end_to_end
            || selection.pipeline?.acoustic_model
            || '';
        const compositionId = record?.composition_id || requested.compositionId;
        if (
            modelId
            && !(state.discovery?.compositions || []).some(
                (composition) => composition.id === compositionId,
            )
        ) {
            const targeted = await fetchDiscoveryPage(0, 32, { model_ids: modelId });
            state.discovery = mergeInventoryDiscovery(state.discovery, targeted);
        }
        const composition = (state.discovery?.compositions || []).find(
            (candidate) => candidate.id === compositionId,
        ) || (state.discovery?.compositions || []).find(
            (candidate) => candidate.backend === selection.backend && candidate.model === selection.model,
        );
        if (!composition) {
            throw new Error(`Saved speech composition ${compositionId || modelId} is unavailable.`);
        }
        state.pathKey = composition.id;
        state.presetId = record?.recipe_id || requested.recipeId;
        const path = pathForComposition(composition);
        const controls = Object.fromEntries(
            (path.controls || [])
                .filter((control) => selection[control.field] != null)
                .map((control) => [control.field, selection[control.field]]),
        );
        restoreRecipeValues(state.values, path, {
            controls,
            variety: selection.variety,
            speaker: selection.speaker,
        });
        renderSpeechSamples();
        renderPathSelector();
        renderSelectedPath();
        if (record) {
            state.speechRun = {
                id: record.run_id,
                compositionId: composition.id,
                recipeId: record.recipe_id || requested.recipeId,
            };
            const metadata = {
                ...record.metadata,
                run_id: record.run_id,
                run_url: record.speech_url,
                audio_url: record.audio_url,
                recipe_id: record.recipe_id,
                composition_id: record.composition_id,
                text_retained: record.text_retained,
                content_detail_retained: record.content_detail_retained,
            };
            renderResult(metadata, record.audio_url);
            byId('speech-result-state').dataset.state = 'ready';
            byId('speech-result-state').textContent = 'Completed';
            byId('speech-submit-status').textContent = (
                `Restored speech run ${record.run_id}; input text was not retained.`
            );
        }
        state.restoringSpeechSelection = false;
        syncSpeechSelectionUrl({ runId: record?.run_id || '' });
        return true;
    }

    async function requestSynthesis(path, values, context, options = {}) {
        const { payload, projection } = await synthesisPayload(path, values, context);
        let response;
        do {
            response = await fetch('/api/speak', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: options.signal,
            });
            if (response.status !== 429 || !options.waitForCapacity) break;
            const retryAfter = Number(response.headers.get('Retry-After'));
            await response.text();
            await new Promise((resolve) => window.setTimeout(
                resolve,
                Number.isFinite(retryAfter) ? Math.max(100, retryAfter * 1000) : 1000,
            ));
            if (options.signal?.aborted) throw new DOMException('Aborted', 'AbortError');
        } while (true);
        if (!response.ok) throw new Error(await response.text());
        const metadata = JSON.parse(response.headers.get('X-Tongues-Speech-Metadata') || '{}');
        if (projection) {
            metadata.diagnostics = { ...(metadata.diagnostics || {}), projection };
        }
        const blob = await response.blob();
        return {
            metadata,
            blob,
            url: URL.createObjectURL(blob),
            payload,
        };
    }

    function currentSynthesisContext(text = byId('text')?.value, path = selectedPath()) {
        const emotionName = state.values.get('emotion');
        return {
            text,
            variety: selectedVariety(path),
            speaker: selectedSpeaker(path),
            emotionVector: state.emotions.find((item) => item.name === emotionName)?.vector,
        };
    }

    function speechInstructionForPath(path, variety = '') {
        const catalog = path?.catalog || [];
        const languages = [...new Set(catalog.flatMap((entry) => entry.languages || []))];
        const scripts = [...new Set(catalog.map((entry) => entry.script).filter(Boolean))];
        const preprocessing = [...new Set(
            catalog.map((entry) => entry.preprocessing).filter(Boolean),
        )];
        const varietyOption = varietiesForPath(path).find((item) => item.id === variety);
        return {
            language: languages.join(', ') || varietyOption?.label || variety || null,
            variety: varietyOption?.label || variety || null,
            script: scripts.join(', ') || null,
            normalization: preprocessing.join(', ') || null,
        };
    }

    async function loadLiveProviders() {
        const response = await fetch('/api/live/providers', { cache: 'no-store' });
        if (!response.ok) throw new Error(await response.text());
        state.liveProviders = (await response.json()).providers || [];
        const select = byId('live-provider');
        select.replaceChildren();
        for (const provider of state.liveProviders) {
            const option = new Option(provider.label, provider.id);
            option.disabled = !provider.available;
            select.appendChild(option);
        }
        const ollama = state.liveProviders.find((provider) => provider.id === 'ollama');
        const preferred = ollama?.available && ollama.models.length
            ? ollama
            : state.liveProviders.find((provider) => provider.available);
        if (preferred) select.value = preferred.id;
        renderLiveProvider();
    }

    function renderLiveProvider() {
        const provider = state.liveProviders.find(
            (candidate) => candidate.id === byId('live-provider')?.value,
        );
        const models = byId('live-model');
        models.replaceChildren();
        for (const model of provider?.models || []) models.appendChild(new Option(model, model));
        byId('live-provider-detail').textContent = provider?.detail || 'Provider unavailable.';
        byId('live-send').disabled = !provider?.available || !(provider.models || []).length
            || !selectedPath()?.runnable;
    }

    function appendLiveJournal(event) {
        state.liveJournal.push(structuredClone(event));
        const journal = byId('live-journal');
        const printable = structuredClone(event);
        const audio = printable?.event?.data?.audio_base64;
        if (audio) {
            printable.event.data.audio_base64 =
                `[${audio.length} base64 characters; included in turn stream]`;
        }
        const line = JSON.stringify(printable);
        journal.textContent = journal.textContent === 'No turn events yet.'
            ? line
            : `${journal.textContent}\n${line}`;
        journal.scrollTop = journal.scrollHeight;
    }

    function renderLiveConversation() {
        const conversation = byId('live-conversation');
        const message = (record) => {
            const article = document.createElement('article');
            article.className = `live-message ${record.role}`;
            article.dataset.messageId = record.id;
            article.dataset.turnId = record.turn_id;
            article.dataset.state = record.status;
            const label = document.createElement('strong');
            label.textContent = record.role === 'user' ? 'You' : 'Tongues';
            const body = document.createElement('p');
            body.className = record.role === 'assistant' ? 'live-assistant-text' : '';
            body.textContent = record.content;
            if (
                record.role === 'assistant'
                && !record.content
                && LIVE_TERMINAL_MESSAGE_STATES.has(record.status)
            ) {
                body.textContent = record.status === 'stopped'
                    ? 'Stopped before a response was produced.'
                    : 'This turn failed before a response was produced.';
            }
            article.append(label, body);
            if (record.role === 'assistant' && record.status !== 'completed') {
                const status = document.createElement('small');
                status.className = 'live-message-status';
                status.textContent = record.status;
                article.append(status);
            }
            return article;
        };
        conversation.replaceChildren(...state.liveConversation.messages.map(message));
        if (!state.liveConversation.messages.length) {
            const empty = document.createElement('p');
            empty.className = 'live-empty';
            empty.textContent =
                'Start a turn to watch generation, planning, and playback move independently.';
            conversation.append(empty);
        }
        conversation.scrollTop = conversation.scrollHeight;
    }

    function resetLiveTurn(turnId, userText) {
        state.liveGenerated = '';
        state.liveCommitted = '';
        state.liveSpoken = '';
        state.liveSegments = new Map();
        state.liveAudioBuffers = [];
        state.liveNextAudioTime = 0;
        state.liveFinalTokenAt = 0;
        state.liveFirstAudioAt = 0;
        state.liveContract = createStreamContractState();
        state.liveJournal = [];
        byId('live-journal').textContent = 'No turn events yet.';
        for (const id of ['live-generated-count', 'live-planned-count', 'live-spoken-count']) {
            byId(id).textContent = '0';
        }
        byId('live-replay').disabled = true;
        byId('live-download').classList.add('hidden');
        byId('live-session-link').classList.add('hidden');
        beginLiveConversationTurn(state.liveConversation, turnId, userText);
        renderLiveConversation();
    }

    function renderLiveAssistant(turnId = state.liveTurn?.id) {
        if (!turnId || !updateLiveAssistant(state.liveConversation, turnId, state.liveGenerated)) {
            return;
        }
        const messageId = `${turnId}:assistant`;
        const article = [...byId('live-conversation').querySelectorAll('[data-message-id]')]
            .find((candidate) => candidate.dataset.messageId === messageId);
        const body = article?.querySelector('.live-assistant-text');
        if (!body) return;
        article.dataset.state = 'streaming';
        const status = article.querySelector('.live-message-status');
        if (status) status.textContent = 'streaming';
        body.replaceChildren();
        let plannedChars = 0;
        for (const segment of state.liveSegments.values()) {
            const span = document.createElement('span');
            span.dataset.segmentId = String(segment.segment_id);
            span.className = `live-segment ${segment.playback || 'planned'}`;
            span.textContent = segment.text;
            body.appendChild(span);
            plannedChars += [...segment.text].length;
        }
        const generatedChars = [...state.liveGenerated];
        const pending = generatedChars.slice(plannedChars).join('');
        if (pending) {
            const span = document.createElement('span');
            span.className = 'live-generating';
            span.textContent = pending;
            body.appendChild(span);
        }
        byId('live-generated-count').textContent = String(generatedChars.length);
        byId('live-planned-count').textContent = String([...state.liveCommitted].length);
        byId('live-spoken-count').textContent = String([...state.liveSpoken].length);
        byId('live-conversation').scrollTop = byId('live-conversation').scrollHeight;
    }

    async function* ndjsonEvents(response) {
        if (!response.body) throw new Error('The browser did not expose the live response stream.');
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
            const { value, done } = await reader.read();
            buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
            let newline;
            while ((newline = buffer.indexOf('\n')) >= 0) {
                const line = buffer.slice(0, newline).trim();
                buffer = buffer.slice(newline + 1);
                if (line) yield JSON.parse(line);
            }
            if (done) break;
        }
        if (buffer.trim()) yield JSON.parse(buffer);
    }

    function createStreamContractState() {
        return {
            streamId: null,
            nextSequence: null,
            segments: new Map(),
            committed: new Set(),
            terminal: false,
        };
    }

    function applyStreamEnvelope(contract, envelope) {
        if (envelope?.schema_version !== 1) {
            throw new Error(`Unsupported stream schema ${envelope?.schema_version}.`);
        }
        if (contract.terminal) throw new Error('Stream event arrived after a terminal event.');
        if (contract.streamId == null) {
            contract.streamId = envelope.stream_id;
            contract.nextSequence = envelope.sequence;
        }
        if (contract.streamId !== envelope.stream_id) {
            throw new Error('Stream identity changed inside one response.');
        }
        if (envelope.sequence !== contract.nextSequence) {
            throw new Error(
                `Out-of-order stream event: expected ${contract.nextSequence}, received ${envelope.sequence}.`,
            );
        }
        contract.nextSequence += 1;
        const payload = {
            type: envelope.event?.type,
            ...(envelope.event?.data || {}),
            event_id: envelope.event_id,
            sequence: envelope.sequence,
        };
        if (payload.type === 'partial_hypothesis') {
            if (contract.committed.has(payload.segment_id)) {
                throw new Error(`Committed segment ${payload.segment_id} was revised.`);
            }
            contract.segments.set(payload.segment_id, payload.text);
        } else if (payload.type === 'revised_hypothesis') {
            if (contract.committed.has(payload.segment_id)) {
                throw new Error(`Committed segment ${payload.segment_id} was revised.`);
            }
            const current = [...(contract.segments.get(payload.segment_id) || '')];
            const { start, end } = payload.replaces;
            if (start > end || end > current.length) {
                throw new Error(`Invalid replacement range ${start}..${end}.`);
            }
            current.splice(start, end - start, ...payload.text);
            contract.segments.set(payload.segment_id, current.join(''));
        } else if (payload.type === 'committed_segment') {
            contract.segments.set(payload.segment_id, payload.text);
            contract.committed.add(payload.segment_id);
        }
        if (
            payload.type === 'completed'
            || payload.type === 'cancelled'
            || (payload.type === 'error' && !payload.recoverable)
        ) contract.terminal = true;
        return payload;
    }

    function liveAudioContext() {
        if (!state.liveAudioContext) {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (!AudioContextClass) throw new Error('This browser does not support Web Audio.');
            state.liveAudioContext = new AudioContextClass();
        }
        return state.liveAudioContext;
    }

    function markLiveSegment(segmentId, playback) {
        const segment = state.liveSegments.get(segmentId);
        if (!segment) return;
        segment.playback = playback;
        renderLiveAssistant(state.liveTurn?.id);
    }

    function settleLivePlaybackSource(source) {
        const pending = state.livePlaybackPending.get(source);
        if (!pending) return;
        state.livePlaybackPending.delete(source);
        pending.resolve();
    }

    async function waitForLivePlayback(generation) {
        while (generation === state.liveGeneration && state.livePlaybackPending.size) {
            await Promise.all(
                [...state.livePlaybackPending.values()].map((pending) => pending.promise),
            );
        }
    }

    async function scheduleLiveAudioSegment(event, generation, signal) {
        if (generation !== state.liveGeneration || signal.aborted) return;
        const binary = window.atob(event.audio_base64);
        const encoded = new Uint8Array(binary.length);
        for (let index = 0; index < binary.length; index += 1) {
            encoded[index] = binary.charCodeAt(index);
        }
        const context = liveAudioContext();
        await context.resume();
        const audioBuffer = await context.decodeAudioData(encoded.buffer);
        if (generation !== state.liveGeneration || signal.aborted) return;
        state.liveAudioBuffers.push(audioBuffer);
        const source = context.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(context.destination);
        const startsAt = Math.max(context.currentTime + 0.035, state.liveNextAudioTime);
        state.liveNextAudioTime = startsAt + audioBuffer.duration;
        const delayMs = Math.max(0, (startsAt - context.currentTime) * 1000);
        const startTimer = window.setTimeout(() => {
            if (generation !== state.liveGeneration) return;
            if (!state.liveFirstAudioAt) state.liveFirstAudioAt = performance.now();
            if (state.liveTurn && !state.liveTurn.times.firstAudio) {
                state.liveTurn.times.firstAudio = performance.now();
                renderConversationLatencies(state.liveTurn.times);
            }
            markLiveSegment(event.segment_id, 'speaking');
            appendLiveJournal({
                type: 'playback_acknowledged',
                turn_id: state.liveTurn?.id,
                segment_id: event.segment_id,
                state: 'started',
                at_ms: Date.now(),
            });
        }, delayMs);
        let resolvePlayback;
        const playback = new Promise((resolve) => {
            resolvePlayback = resolve;
        });
        state.livePlaybackPending.set(source, {
            promise: playback,
            resolve: resolvePlayback,
        });
        source.onended = () => {
            window.clearTimeout(startTimer);
            state.liveSources.delete(source);
            settleLivePlaybackSource(source);
            if (generation !== state.liveGeneration) return;
            state.liveSpoken += event.text;
            if (state.liveTurn) {
                state.liveTurn.times.playbackCompleted = performance.now();
                renderConversationLatencies(state.liveTurn.times);
            }
            markLiveSegment(event.segment_id, 'spoken');
            appendLiveJournal({
                type: 'playback_acknowledged',
                turn_id: state.liveTurn?.id,
                segment_id: event.segment_id,
                state: 'completed',
                at_ms: Date.now(),
            });
        };
        state.liveSources.add(source);
        try {
            source.start(startsAt);
        } catch (error) {
            window.clearTimeout(startTimer);
            state.liveSources.delete(source);
            settleLivePlaybackSource(source);
            throw error;
        }
    }

    function enqueueLiveSegment(event, generation, signal) {
        state.liveSegments.set(event.segment_id, { ...event, playback: 'planned' });
        if (state.liveTurn && !state.liveTurn.times.firstPlanned) {
            state.liveTurn.times.firstPlanned = performance.now();
            renderConversationLatencies(state.liveTurn.times);
        }
        state.liveCommitted += event.text;
        renderLiveAssistant(state.liveTurn?.id);
    }

    function enqueueLiveAudio(event, generation, signal) {
        state.liveSynthesisTail = state.liveSynthesisTail
            .then(() => scheduleLiveAudioSegment(event, generation, signal))
            .catch((error) => {
                if (error.name !== 'AbortError' && generation === state.liveGeneration) {
                    appendLiveJournal({
                        type: 'audio_decode_failed',
                        segment_id: event.segment_id,
                        message: error.message,
                        at_ms: Date.now(),
                    });
                    showError(`Live audio decoding failed: ${error.message}`, byId('live-error'));
                    markLiveSegment(event.segment_id, 'failed');
                }
            });
    }

    function wavBlobFromBuffers(buffers) {
        if (!buffers.length) return null;
        const sampleRate = buffers[0].sampleRate;
        if (buffers.some((buffer) => buffer.sampleRate !== sampleRate)) {
            throw new Error('Live segment sample rates changed during the turn.');
        }
        const samples = buffers.reduce((total, buffer) => total + buffer.length, 0);
        const array = new ArrayBuffer(44 + samples * 2);
        const view = new DataView(array);
        const writeText = (offset, text) => {
            for (let index = 0; index < text.length; index += 1) {
                view.setUint8(offset + index, text.charCodeAt(index));
            }
        };
        writeText(0, 'RIFF');
        view.setUint32(4, 36 + samples * 2, true);
        writeText(8, 'WAVEfmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeText(36, 'data');
        view.setUint32(40, samples * 2, true);
        let offset = 44;
        for (const buffer of buffers) {
            for (const sample of buffer.getChannelData(0)) {
                const clamped = Math.max(-1, Math.min(1, sample));
                view.setInt16(offset, clamped < 0 ? clamped * 32768 : clamped * 32767, true);
                offset += 2;
            }
        }
        return new Blob([array], { type: 'audio/wav' });
    }

    async function replayLiveAudio() {
        const context = liveAudioContext();
        await context.resume();
        let startsAt = context.currentTime + 0.035;
        for (const buffer of state.liveAudioBuffers) {
            const source = context.createBufferSource();
            source.buffer = buffer;
            source.connect(context.destination);
            source.start(startsAt);
            startsAt += buffer.duration;
        }
    }

    async function stopLiveTurn() {
        const turn = state.liveTurn;
        if (!turn) return;
        state.liveGeneration += 1;
        turn.controller.abort();
        for (const source of state.liveSources) {
            try { source.stop(); } catch (_) { /* already stopped */ }
        }
        for (const source of [...state.livePlaybackPending.keys()]) {
            settleLivePlaybackSource(source);
        }
        state.liveSources.clear();
        state.liveSynthesisTail = Promise.resolve();
        fetch(`/api/live/turn/${encodeURIComponent(turn.id)}/cancel`, {
            method: 'POST',
        }).catch(() => {});
        state.liveTurn = null;
        finishLiveAssistant(state.liveConversation, turn.id, 'stopped');
        renderLiveConversation();
        byId('live-state').dataset.state = 'failed';
        byId('live-state').textContent = 'Stopped';
        byId('live-stop').disabled = true;
        byId('live-send').disabled = false;
        appendLiveJournal({ type: 'turn_cancelled', turn_id: turn.id, at_ms: Date.now() });
        try {
            await saveLiveConversationSession();
        } catch (error) {
            showError(`Stopped turn could not be saved: ${error.message}`, byId('live-error'));
        }
    }

    async function startLiveTurn(userText) {
        const path = selectedPath();
        if (!path?.runnable) throw new Error('Choose a ready speech recipe first.');
        await liveAudioContext().resume();
        const provider = byId('live-provider').value;
        const model = byId('live-model').value;
        if (!provider || !model) throw new Error('Choose an available text provider and model.');
        const generation = state.liveGeneration + 1;
        state.liveGeneration = generation;
        const turnId = `turn-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        const controller = new AbortController();
        state.liveTurn = {
            id: turnId,
            controller,
            times: {
                captureStarted: state.conversationMicStartedAt,
                speechStarted: state.conversationMicStartedAt,
                firstPartial: state.conversationFirstPartialAt,
                committed: state.conversationCommittedAt || performance.now(),
            },
        };
        state.liveSynthesisTail = Promise.resolve();
        resetLiveTurn(turnId, userText);
        clearError(byId('live-error'));
        byId('live-state').dataset.state = 'busy';
        byId('live-state').textContent = 'Generating';
        byId('live-stop').disabled = false;
        byId('live-send').disabled = true;
        state.liveMessages = liveProviderMessages(state.liveConversation);
        const synthesis = buildPayload(
            path,
            state.values,
            currentSynthesisContext(userText, path),
        );
        for (const field of ['pitch', 'energy', 'durations']) delete synthesis[field];
        const response = await fetch('/api/live/turn', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: controller.signal,
            body: JSON.stringify({
                turn_id: turnId,
                provider,
                model,
                messages: state.liveMessages,
                response_instructions: byId('live-instructions').value.trim(),
                speech: speechInstructionForPath(path, selectedVariety(path)),
                synthesis,
            }),
        });
        if (!response.ok) throw new Error(await response.text());
        let completedEvent = null;
        for await (const envelope of ndjsonEvents(response)) {
            if (generation !== state.liveGeneration) return;
            appendLiveJournal(envelope);
            const event = applyStreamEnvelope(state.liveContract, envelope);
            if (event.type === 'partial_hypothesis' && event.role === 'generation') {
                if (state.liveTurn && !state.liveTurn.times.firstGenerated) {
                    state.liveTurn.times.firstGenerated = performance.now();
                    renderConversationLatencies(state.liveTurn.times);
                }
                state.liveGenerated = event.text;
                renderLiveAssistant(turnId);
            } else if (event.type === 'committed_segment' && event.role === 'generation') {
                enqueueLiveSegment(event, generation, controller.signal);
            } else if (event.type === 'audio_chunk' && event.direction === 'output') {
                event.text = event.metadata?.text || '';
                enqueueLiveAudio(event, generation, controller.signal);
            } else if (event.type === 'text_completed' && event.role === 'generation') {
                state.liveFinalTokenAt = performance.now();
            } else if (event.type === 'completed') {
                completedEvent = event;
            } else if (event.type === 'error' && !event.recoverable) {
                throw new Error(event.message);
            }
        }
        await state.liveSynthesisTail;
        await waitForLivePlayback(generation);
        if (generation !== state.liveGeneration) return;
        if (!completedEvent || state.liveGenerated !== state.liveCommitted) {
            throw new Error('Committed speech transcript does not exactly match generated text.');
        }
        finishLiveAssistant(state.liveConversation, turnId, 'completed');
        state.liveMessages = liveProviderMessages(state.liveConversation);
        renderLiveConversation();
        const overlap = state.liveFirstAudioAt > 0
            && state.liveFirstAudioAt < state.liveFinalTokenAt;
        appendLiveJournal({
            type: 'turn_acceptance',
            turn_id: turnId,
            transcript_exact: state.liveGenerated === state.liveCommitted,
            first_audio_before_final_token: overlap,
            generated_chars: [...state.liveGenerated].length,
            committed_chars: [...state.liveCommitted].length,
            at_ms: Date.now(),
        });
        const durableEvents = state.liveJournal.filter((item) => (
            item?.event?.type === 'committed_segment'
            || item?.event?.data?.type === 'committed_segment'
        ));
        state.liveDurableEvents.push(...durableEvents);
        await saveLiveConversationSession();
        const wav = wavBlobFromBuffers(state.liveAudioBuffers);
        if (wav) {
            const download = byId('live-download');
            if (download.href) URL.revokeObjectURL(download.href);
            download.href = URL.createObjectURL(wav);
            download.classList.remove('hidden');
            byId('live-replay').disabled = false;
        }
        state.liveTurn = null;
        byId('live-state').dataset.state = overlap ? 'ready' : 'loading';
        byId('live-state').textContent = overlap ? 'Streamed' : 'Completed';
        byId('live-stop').disabled = true;
        byId('live-send').disabled = false;
    }

    async function saveLiveConversationSession() {
        if (!state.liveSessionId) {
            state.liveSessionId =
                `conversation:${Date.now()}-${Math.random().toString(16).slice(2)}`;
        }
        const { sessionFromEvents } = await import('/wavedeck-model.mjs');
        const session = sessionFromEvents(state.liveSessionId, state.liveDurableEvents);
        const response = await fetch(
            `/api/timeline/sessions/${encodeURIComponent(state.liveSessionId)}`,
            {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    schema_version: 1,
                    session,
                    context: {
                        graph_id: 'starter:live_conversation',
                        source: 'live conversation',
                        conversation: structuredClone(state.liveConversation),
                    },
                }),
            },
        );
        if (!response.ok) {
            throw new Error(`Conversation session could not be saved: ${await response.text()}`);
        }
        const sessionLink = byId('live-session-link');
        sessionLink.href = `/sessions/${encodeURIComponent(state.liveSessionId)}/correct`;
        sessionLink.classList.remove('hidden');
        const liveUrl = new URL(browser.location.href);
        liveUrl.searchParams.set('session', state.liveSessionId);
        browser.history.replaceState({ workflow: 'live' }, '', liveUrl);
    }

    async function restoreLiveConversationSession() {
        const sessionId = new URL(browser.location.href).searchParams.get('session');
        if (!sessionId) return;
        const response = await fetch(
            `/api/timeline/sessions/${encodeURIComponent(sessionId)}`,
            { cache: 'no-store' },
        );
        if (!response.ok) throw new Error(await response.text());
        const record = await response.json();
        state.liveConversation = createLiveConversation(record.context?.conversation);
        state.liveSessionId = sessionId;
        state.liveDurableEvents = structuredClone(record.session?.source_events || []);
        state.liveMessages = liveProviderMessages(state.liveConversation);
        renderLiveConversation();
        const sessionLink = byId('live-session-link');
        sessionLink.href = `/sessions/${encodeURIComponent(sessionId)}/correct`;
        sessionLink.classList.remove('hidden');
    }

    function renderConversationLatencies(times = {}) {
        const delta = (end, start) => end && start ? Math.max(0, end - start) : null;
        const rows = [
            ['Auditory detection', delta(times.speechStarted, times.captureStarted)],
            ['Segmentation / final ASR', delta(times.committed, times.speechStarted)],
            ['ASR first partial', delta(times.firstPartial, times.speechStarted)],
            ['LLM first token', delta(times.firstGenerated, times.committed)],
            ['Speech planning', delta(times.firstPlanned, times.firstGenerated)],
            ['TTS first audio', delta(times.firstAudio, times.firstPlanned)],
            ['Playback', delta(times.playbackCompleted, times.firstAudio)],
        ];
        const target = byId('live-latencies');
        if (!target) return;
        target.replaceChildren(...rows.flatMap(([label, value]) => {
            const term = document.createElement('dt');
            term.textContent = label;
            const detail = document.createElement('dd');
            detail.textContent = value == null ? 'pending' : `${value.toFixed(1)} ms`;
            return [term, detail];
        }));
    }

    async function handleConversationAsrEvent(event) {
        if (event.type === 'partial_hypothesis' || event.type === 'revised_hypothesis') {
            if (!state.conversationFirstPartialAt) state.conversationFirstPartialAt = performance.now();
            byId('live-asr-text').className = 'live-generating';
            byId('live-asr-text').textContent = `Provisional: ${event.data?.text || '…'}`;
            renderConversationLatencies({
                captureStarted: state.conversationMicStartedAt,
                speechStarted: state.conversationMicStartedAt,
                firstPartial: state.conversationFirstPartialAt,
            });
            return;
        }
        if (event.type !== 'committed_segment' || event.data?.role !== 'recognition') return;
        state.conversationCommittedAt = performance.now();
        byId('live-asr-text').className = 'committed';
        byId('live-asr-text').textContent = `Committed: ${event.data.text}`;
        const { committedTurnAction } = await import('/live-conversation-model.mjs');
        const decision = committedTurnAction({
            event,
            playbackActive: state.liveSources.size > 0 || Boolean(state.liveTurn),
            bargeIn: byId('live-barge-in').checked,
            spokenText: `${state.liveSpoken} ${state.liveCommitted}`,
        });
        appendLiveJournal({
            type: 'recognition_commit_decision',
            decision: decision.action,
            reason: decision.reason,
            text: event.data.text,
            at_ms: Date.now(),
        });
        if (decision.action === 'ignore_echo' || decision.action === 'wait') return;
        if (decision.action === 'cancel_and_restart') await stopLiveTurn();
        await startLiveTurn(event.data.text);
    }

    async function startConversationMicrophone() {
        if (state.conversationMicSocket) return;
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        const context = new AudioContextClass();
        const source = context.createMediaStreamSource(stream);
        const node = context.createScriptProcessor(4096, 1, 1);
        const socket = new WebSocket(
            `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/api/asr/stream`,
        );
        socket.binaryType = 'arraybuffer';
        state.conversationMicStream = stream;
        state.conversationMicContext = context;
        state.conversationMicSource = source;
        state.conversationMicNode = node;
        state.conversationMicSocket = socket;
        state.conversationMicStartedAt = performance.now();
        state.conversationFirstPartialAt = 0;
        state.conversationCommittedAt = 0;
        socket.addEventListener('open', () => socket.send(JSON.stringify({
            type: 'open',
            schema_version: 1,
            provider: 'fixture',
            sample_rate_hz: context.sampleRate,
            channels: 1,
            language: selectedVariety(selectedPath())?.split('-')[0] || 'en',
        })));
        socket.addEventListener('message', (message) => {
            const payload = JSON.parse(message.data);
            if (payload.type === 'recognition') {
                handleConversationAsrEvent(payload.event).catch((error) => {
                    showError(`Conversation recognition failed: ${error.message}`, byId('live-error'));
                });
            } else if (payload.type === 'error') {
                showError(`${payload.code}: ${payload.message}`, byId('live-error'));
            }
        });
        socket.addEventListener('close', stopConversationMicrophone);
        node.onaudioprocess = (audioEvent) => {
            if (socket.readyState === WebSocket.OPEN) {
                socket.send(audioEvent.inputBuffer.getChannelData(0).slice().buffer);
            }
        };
        source.connect(node);
        node.connect(context.destination);
        byId('live-mic-start').disabled = true;
        byId('live-mic-stop').disabled = false;
        byId('live-asr-text').textContent = 'Listening; only committed recognition starts a turn.';
    }

    function stopConversationMicrophone() {
        const socket = state.conversationMicSocket;
        state.conversationMicSocket = null;
        if (socket?.readyState === WebSocket.OPEN) socket.send('{"type":"end"}');
        state.conversationMicNode?.disconnect();
        state.conversationMicSource?.disconnect();
        state.conversationMicStream?.getTracks().forEach((track) => track.stop());
        state.conversationMicContext?.close();
        state.conversationMicNode = null;
        state.conversationMicSource = null;
        state.conversationMicStream = null;
        state.conversationMicContext = null;
        byId('live-mic-start').disabled = false;
        byId('live-mic-stop').disabled = true;
    }

    function comparisonRecipes() {
        const builtIns = availableCompositions(state.discovery)
            .filter((composition) => composition.runnable)
            .map((composition) => ({
                id: composition.id,
                name: composition.display_name,
                compositionId: composition.id,
                builtIn: true,
                controls: defaultControlValues(composition, state.sampleDefaults),
            }));
        return [
            ...builtIns,
            ...state.userRecipes.filter((recipe) => (
                builtIns.some((candidate) => candidate.compositionId === recipe.compositionId)
            )),
        ];
    }

    function renderCompareCandidates() {
        const target = byId('compare-candidates');
        if (!target || !state.discovery) return;
        const checked = new Set(
            [...target.querySelectorAll('input:checked')].map((input) => input.value),
        );
        const recipes = comparisonRecipes();
        target.replaceChildren();
        recipes.forEach((recipe, index) => {
            const label = document.createElement('label');
            label.className = 'compare-candidate';
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.name = 'compare-recipe';
            input.value = recipe.id;
            input.checked = checked.size ? checked.has(recipe.id) : index < 2;
            const text = document.createElement('span');
            text.innerHTML = `<strong>${escapeHtml(recipe.name)}</strong><small>${
                recipe.builtIn ? 'Built-in verified recipe' : 'Saved user recipe'
            }</small>`;
            label.append(input, text);
            target.appendChild(label);
        });
    }

    function selectedComparisonRecipes() {
        const ids = new Set(
            [...byId('compare-candidates').querySelectorAll('input:checked')]
                .map((input) => input.value),
        );
        return comparisonRecipes().filter((recipe) => ids.has(recipe.id));
    }

    function comparisonLane(recipe, index, blind) {
        const lane = document.createElement('article');
        lane.className = 'compare-lane';
        lane.dataset.recipeId = recipe.id;
        lane.dataset.state = 'queued';
        lane.innerHTML = `
            <div class="speech-section-heading">
                <div>
                    <span class="runtime-badge" data-compare-state data-state="loading">Queued</span>
                    <h3 data-compare-name>${escapeHtml(blind ? `Candidate ${String.fromCharCode(65 + index)}` : recipe.name)}</h3>
                </div>
                <label class="preferred-choice hidden">
                    <input type="radio" name="preferred-recipe" value="${escapeAttribute(recipe.id)}">
                    <span>Preferred</span>
                </label>
            </div>
            <p data-compare-detail>${escapeHtml(blind ? 'Identity hidden' : (recipe.builtIn ? 'Built-in recipe' : 'Saved recipe'))}</p>
            <audio class="hidden" controls></audio>
            <details class="advanced-section hidden">
                <summary>Execution details</summary>
                <pre class="source-preview"></pre>
            </details>
        `;
        lane.querySelector('input[name="preferred-recipe"]').addEventListener('change', (event) => {
            state.comparePreferred = event.target.value;
            byId('save-preferred').disabled = false;
        });
        return lane;
    }

    function revealComparison() {
        const recipes = new Map(comparisonRecipes().map((recipe) => [recipe.id, recipe]));
        byId('compare-results').querySelectorAll('.compare-lane').forEach((lane) => {
            const recipe = recipes.get(lane.dataset.recipeId);
            if (recipe) lane.querySelector('[data-compare-name]').textContent = recipe.name;
            lane.querySelector('[data-compare-detail]').textContent = recipe?.builtIn
                ? 'Built-in recipe'
                : 'Saved user recipe';
        });
        byId('compare-blind').checked = false;
    }

    async function generateComparison() {
        clearError(byId('compare-error'));
        const recipes = selectedComparisonRecipes();
        const text = byId('compare-text').value.trim();
        if (!text) {
            showError('Enter one shared prompt for the comparison.', byId('compare-error'));
            return;
        }
        if (recipes.length < 2) {
            showError('Select at least two complete recipes.', byId('compare-error'));
            return;
        }
        for (const result of state.compareResults.values()) {
            if (result.url) URL.revokeObjectURL(result.url);
        }
        state.compareResults.clear();
        state.comparePreferred = '';
        byId('save-preferred').disabled = true;
        const blind = byId('compare-blind').checked;
        const target = byId('compare-results');
        target.replaceChildren(...recipes.map((recipe, index) => comparisonLane(recipe, index, blind)));
        byId('generate-all').disabled = true;
        byId('compare-status').textContent = `${recipes.length} candidates queued.`;
        startRuntimePolling();
        const runtime = await loadRuntime().catch(() => null);
        const concurrency = Math.max(
            1,
            Number(runtime?.capacity) || DEFAULT_COMPARISON_CONCURRENCY,
        );
        await mapWithConcurrency(recipes, concurrency, async (recipe) => {
            const lane = target.querySelector(`[data-recipe-id="${CSS.escape(recipe.id)}"]`);
            const badge = lane.querySelector('[data-compare-state]');
            badge.textContent = 'Running';
            badge.dataset.state = 'busy';
            lane.dataset.state = 'running';
            const composition = state.discovery.compositions.find(
                (candidate) => candidate.id === recipe.compositionId,
            );
            const path = pathForComposition(composition);
            const values = new Map(state.values);
            for (const [field, value] of Object.entries(recipe.controls || {})) values.set(field, value);
            const context = {
                text,
                variety: recipe.variety || varietiesForPath(path)[0]?.id,
                speaker: comparisonSpeaker(path, recipe),
            };
            try {
                const result = await requestSynthesis(
                    path,
                    values,
                    context,
                    { waitForCapacity: true },
                );
                result.recipe = {
                    ...recipe,
                    variety: result.payload.variety || recipe.variety || null,
                    speaker: result.payload.speaker || recipe.speaker || null,
                    controls: Object.fromEntries(values),
                };
                state.compareResults.set(recipe.id, result);
                lane.dataset.state = 'completed';
                badge.textContent = 'Completed';
                badge.dataset.state = 'ready';
                const audio = lane.querySelector('audio');
                audio.src = result.url;
                audio.classList.remove('hidden');
                lane.querySelector('.preferred-choice').classList.remove('hidden');
                const details = lane.querySelector('details');
                details.classList.remove('hidden');
                details.querySelector('pre').textContent = JSON.stringify({
                    recipe: result.recipe,
                    request: result.payload,
                    synthesis: result.metadata,
                }, null, 2);
            } catch (error) {
                state.compareResults.set(recipe.id, { recipe, error: error.message });
                lane.dataset.state = 'failed';
                badge.textContent = 'Failed';
                badge.dataset.state = 'failed';
                lane.querySelector('[data-compare-detail]').textContent = error.message;
            }
        });
        stopRuntimePolling();
        const successes = [...state.compareResults.values()].filter((result) => result.url).length;
        byId('compare-status').textContent = `${successes} of ${recipes.length} candidates completed. Results remain playable without regeneration.`;
        byId('generate-all').disabled = false;
    }

    function humanJobLabel(job) {
        if (job.label && !job.label.startsWith('cargo ')) return job.label;
        const args = job.args || [];
        const command = args.slice(args.indexOf('--') + 1);
        if (command[0] === 'models' && command[1] === 'install') {
            return `Fetched ${command[2] || 'speech model'}`;
        }
        if (command[0] === 'speak') return 'Generated speech utterance';
        return command.join(' ') || job.label || 'Tongues job';
    }

    function renderOperateJobOutput(card) {
        const raw = card.querySelector('.advanced-section');
        if (!raw?.dataset.loaded) return;
        raw.querySelector('pre').textContent = (card.jobOutput || [])
            .map((line) => `[${line.stream}] ${line.line}`).join('\n') || 'No log output.';
    }

    async function loadOperateJobDetail(card, { force = false } = {}) {
        const raw = card.querySelector('.advanced-section');
        if (!raw || raw.dataset.loading || (raw.dataset.loaded && !force)) return;
        raw.dataset.loading = 'true';
        try {
            const response = await fetch(`/api/jobs/${encodeURIComponent(card.dataset.jobId)}`);
            if (!response.ok) {
                raw.querySelector('pre').textContent = await response.text();
                return;
            }
            const detail = await response.json();
            card.jobOutput = detail.output || [];
            raw.dataset.loaded = 'true';
            renderOperateJobOutput(card);
            const artifacts = raw.querySelector('.job-artifacts');
            artifacts.replaceChildren(...(detail.artifacts || []).map((artifact) => {
                const item = document.createElement(artifact.download_url ? 'a' : 'span');
                item.textContent = artifact.label || artifact.path;
                if (artifact.download_url) item.href = artifact.download_url;
                return item;
            }));
        } finally {
            delete raw.dataset.loading;
        }
    }

    function updateOperateJobCard(card, update) {
        const job = { ...(card.job || {}), ...update };
        if (update.progress) job.progress = update.progress;
        card.job = job;
        card.querySelector('[data-job-label]').textContent = humanJobLabel(job);
        card.querySelector('[data-job-phase]').textContent = job.progress?.phase || job.status;
        const badge = card.querySelector('.runtime-badge');
        badge.dataset.state = job.status;
        badge.textContent = job.status;
        const progress = card.querySelector('[data-job-progress]');
        progress.textContent = job.progress?.total
            ? `${job.progress.current || 0} / ${job.progress.total}`
            : job.progress?.phase || job.status;
        const percent = job.progress?.total
            ? Math.min(100, Math.round((job.progress.current || 0) / job.progress.total * 100))
            : (job.status === 'running' ? 35 : 100);
        card.querySelector('.progress-bar').style.width = `${percent}%`;
        const cancel = card.querySelector('[data-cancel-job]');
        if (job.status !== 'running') cancel?.remove();
    }

    function operateJobCard(job) {
        const details = document.createElement('details');
        details.className = 'operate-job';
        details.dataset.jobId = job.id;
        details.job = job;
        details.jobOutput = [];
        const summary = document.createElement('summary');
        summary.innerHTML = `
            <span><strong data-job-label>${escapeHtml(humanJobLabel(job))}</strong>
            <small data-job-phase>${escapeHtml(job.progress?.phase || job.status)}</small></span>
            <span class="runtime-badge" data-state="${escapeAttribute(job.status)}">${escapeHtml(job.status)}</span>
        `;
        const body = document.createElement('div');
        body.className = 'operate-job-detail';
        body.innerHTML = `
            <div class="progress-shell"><div class="progress-bar"></div></div>
            <p data-job-progress>${escapeHtml(job.progress?.total
                ? `${job.progress.current || 0} / ${job.progress.total}`
                : job.progress?.phase || job.status)}</p>
            ${job.status === 'running'
                ? `<button type="button" class="danger-button" data-cancel-job="${escapeAttribute(job.id)}">Cancel</button>`
                : ''}
            <details class="advanced-section">
                <summary>Command, logs, and artifacts</summary>
                <code>${escapeHtml(`${job.command} ${(job.args || []).join(' ')}`)}</code>
                <pre class="source-preview">Loading details…</pre>
                <div class="job-artifacts"></div>
            </details>
        `;
        body.querySelector('[data-cancel-job]')?.addEventListener('click', async (event) => {
            event.currentTarget.disabled = true;
            const response = await fetch(
                `/api/jobs/${encodeURIComponent(event.currentTarget.dataset.cancelJob)}/cancel`,
                { method: 'POST' },
            );
            if (!response.ok) {
                event.currentTarget.disabled = false;
                body.querySelector('[data-job-progress]').textContent = await response.text();
            } else {
                body.querySelector('[data-job-progress]').textContent = 'Cancel requested';
            }
        });
        const raw = body.querySelector('.advanced-section');
        raw.addEventListener('toggle', async () => {
            if (raw.open) {
                await loadOperateJobDetail(details);
            }
        });
        details.append(summary, body);
        updateOperateJobCard(details, job);
        return details;
    }

    function closeOperateJobStream(jobId) {
        state.jobStreams.get(jobId)?.close();
        state.jobStreams.delete(jobId);
    }

    function stopOperateJobStreams() {
        for (const stream of state.jobStreams.values()) stream.close();
        state.jobStreams.clear();
    }

    function streamOperateJob(job) {
        if (
            state.workflow !== 'operate'
            || job.status !== 'running'
            || state.jobStreams.has(job.id)
            || !browser?.EventSource
        ) return;
        const stream = new browser.EventSource(
            `/api/jobs/${encodeURIComponent(job.id)}/events`,
        );
        state.jobStreams.set(job.id, stream);
        stream.addEventListener('message', (event) => {
            let message;
            try {
                message = JSON.parse(event.data);
            } catch (_) {
                return;
            }
            const card = [...byId('operate-jobs')?.querySelectorAll('[data-job-id]') || []]
                .find((candidate) => candidate.dataset.jobId === job.id);
            if (!card) return;
            if (message.type === 'snapshot') {
                card.jobOutput = message.output || [];
                updateOperateJobCard(card, message.summary);
                renderOperateJobOutput(card);
                if (message.summary.status !== 'running') closeOperateJobStream(job.id);
            } else if (message.type === 'output') {
                card.jobOutput.push(message);
                if (card.jobOutput.length > JOB_OUTPUT_LIMIT) card.jobOutput.shift();
                renderOperateJobOutput(card);
            } else if (message.type === 'progress') {
                updateOperateJobCard(card, { progress: message.progress });
            } else if (message.type === 'status') {
                updateOperateJobCard(card, message.summary);
                closeOperateJobStream(job.id);
                if (card.querySelector('.advanced-section')?.open) {
                    loadOperateJobDetail(card, { force: true }).catch(() => {});
                }
            }
        });
    }

    async function refreshOperateJobs() {
        if (!byId('operate-jobs')) return;
        const response = await fetch('/api/jobs', { cache: 'no-store' });
        if (!response.ok) throw new Error(await response.text());
        const jobs = await response.json();
        const target = byId('operate-jobs');
        const existing = new Map(
            [...target.querySelectorAll('[data-job-id]')]
                .map((card) => [card.dataset.jobId, card]),
        );
        const cards = jobs.map((job) => {
            const card = existing.get(job.id) || operateJobCard(job);
            updateOperateJobCard(card, job);
            streamOperateJob(job);
            return card;
        });
        for (const jobId of state.jobStreams.keys()) {
            if (!jobs.some((job) => job.id === jobId && job.status === 'running')) {
                closeOperateJobStream(jobId);
            }
        }
        target.replaceChildren(...cards);
        if (!jobs.length) {
            const empty = document.createElement('p');
            empty.className = 'catalog-empty';
            empty.textContent = 'No background activity yet.';
            target.appendChild(empty);
        }
    }

    async function init() {
        const page = byId('speech-page');
        if (!page) return;
        page.innerHTML = studioShell();
        loadUserRecipes();
        const requestedSpeechSelection = speechSelectionFromLocation();
        state.restoringSpeechSelection = Boolean(
            requestedSpeechSelection.runId
            || requestedSpeechSelection.recipeId
            || requestedSpeechSelection.compositionId
        );
        document.querySelectorAll('[data-studio-route]').forEach((link) => {
            link.addEventListener('click', (event) => {
                event.preventDefault();
                navigateWorkflow(workflowForPath(link.dataset.studioRoute));
            });
        });
        setWorkflow(browser?.location?.pathname || '/speech');
        try {
            await restoreLiveConversationSession();
        } catch (error) {
            showError(`Conversation history could not be restored: ${error.message}`, byId('live-error'));
        }
        const submit = byId('submit-btn');
        try {
            await loadAuxiliaryDiscovery();
            await refreshDiscovery(false);
            await loadLiveProviders();
        } catch (error) {
            showError(`Speech discovery failed: ${error.message}`);
            submit.disabled = true;
            byId('speech-runtime-state').dataset.state = 'failed';
            byId('speech-runtime-state').textContent = 'failed';
        }
        if (state.restoringSpeechSelection) {
            try {
                await restoreSpeechSelectionFromLocation();
            } catch (error) {
                state.restoringSpeechSelection = false;
                showError(`Saved speech selection could not be restored: ${error.message}`);
            }
        }
        const linkedDuplexRequest = duplexRequestFromLocation(browser?.location);
        if (linkedDuplexRequest) {
            if (linkedDuplexRequest.journal_path) {
                byId('duplex-journal-path').value = linkedDuplexRequest.journal_path;
            }
            try {
                renderDuplexProjection(await loadDuplexProjection(linkedDuplexRequest));
                byId('duplex-evidence-heading').focus();
            } catch (error) {
                showError(`Linked interpretation evidence failed: ${error.message}`, byId('duplex-error'));
            }
        }

        let catalogSearchTimer = null;
        document.querySelectorAll('[data-catalog-view]').forEach((button) => {
            button.addEventListener('click', () => {
                state.catalogView = button.dataset.catalogView;
                document.querySelectorAll('[data-catalog-view]').forEach((candidate) => {
                    candidate.setAttribute(
                        'aria-selected',
                        String(candidate === button),
                    );
                });
                renderInventory();
            });
        });
        byId('catalog-search').addEventListener('input', () => {
            window.clearTimeout(catalogSearchTimer);
            catalogSearchTimer = window.setTimeout(() => {
                refreshCatalog().catch((error) => {
                    byId('catalog-status').textContent = `Catalog search failed: ${error.message}`;
                });
            }, 250);
        });
        for (const id of [
            'catalog-family',
            'catalog-license',
            'catalog-capability',
            'catalog-verification',
            'catalog-device',
        ]) {
            byId(id).addEventListener('change', () => {
                refreshCatalog().catch((error) => {
                    byId('catalog-status').textContent = `Catalog filter failed: ${error.message}`;
                });
            });
        }
        byId('speech-preset').addEventListener('change', (event) => {
            const userRecipe = state.userRecipes.find(
                (candidate) => candidate.id === event.target.value,
            );
            if (userRecipe) {
                state.presetId = userRecipe.id;
                applyRecipe(userRecipe);
                return;
            }
            const preset = state.discovery.presets.find(
                (candidate) => candidate.id === event.target.value,
            );
            if (!preset) {
                state.presetId = '';
                renderSelectedPath();
                return;
            }
            state.pathKey = preset.composition_id;
            state.presetId = preset.id;
            renderSelectedPath();
        });
        byId('speech-language').addEventListener('change', (event) => {
            markSpeechSelectionChanged();
            const language = event.target.value;
            const current = selectedComposition();
            const candidates = availableCompositions(state.discovery).filter((composition) => (
                pathSupportsSampleLanguage(
                    pathForComposition(composition, state.discovery),
                    language,
                )
            ));
            const next = current && candidates.some((candidate) => candidate.id === current.id)
                ? current
                : (candidates.find((candidate) => candidate.runnable) || candidates[0]);
            if (next) {
                state.pathKey = next.id;
                state.presetId = '';
            }
            renderSpeechSamples({ replaceText: true });
            renderPathSelector();
            renderSelectedPath();
        });
        byId('speech-sample').addEventListener('change', (event) => {
            const language = SPEECH_SAMPLES.find(
                (candidate) => candidate.code === byId('speech-language').value,
            );
            const text = language?.texts?.[Number(event.target.value)];
            if (text) {
                byId('text').value = text;
                markSpeechSelectionChanged();
            }
        });
        byId('text').addEventListener('input', markSpeechSelectionChanged);
        byId('live-recipe').addEventListener('change', (event) => {
            byId('speech-preset').value = event.target.value;
            byId('speech-preset').dispatchEvent(new Event('change'));
        });
        byId('live-provider').addEventListener('change', renderLiveProvider);
        byId('live-form').addEventListener('submit', (event) => {
            event.preventDefault();
            const input = byId('live-message');
            const text = input.value.trim();
            if (!text) return;
            input.value = '';
            startLiveTurn(text).catch((error) => {
                if (error.name === 'AbortError') return;
                const failedTurn = state.liveTurn;
                if (failedTurn) {
                    finishLiveAssistant(state.liveConversation, failedTurn.id, 'failed');
                    renderLiveConversation();
                    saveLiveConversationSession().catch(() => {});
                }
                showError(`Live turn failed: ${error.message}`, byId('live-error'));
                byId('live-state').dataset.state = 'failed';
                byId('live-state').textContent = 'Failed';
                byId('live-stop').disabled = true;
                byId('live-send').disabled = false;
                state.liveTurn = null;
            });
        });
        byId('live-stop').addEventListener('click', () => {
            stopLiveTurn().catch(() => {});
        });
        byId('live-mic-start').addEventListener('click', () => {
            startConversationMicrophone().catch((error) => {
                showError(`Microphone conversation failed: ${error.message}`, byId('live-error'));
            });
        });
        byId('live-mic-stop').addEventListener('click', stopConversationMicrophone);
        byId('live-replay').addEventListener('click', () => {
            replayLiveAudio().catch((error) => {
                showError(`Replay failed: ${error.message}`, byId('live-error'));
            });
        });
        byId('speech-voice').addEventListener('change', (event) => {
            const option = [...byId('speech-voice-options').options].find(
                (candidate) => candidate.value === event.target.value,
            );
            const next = state.discovery.compositions.find(
                (composition) => composition.id === option?.dataset.compositionId,
            );
            if (!next) {
                showError('Choose a voice or language from the discovered complete recipes.');
                return;
            }
            state.pathKey = next.id;
            state.presetId = '';
            renderPathSelector();
            renderSelectedPath();
        });
        const selectCompositionForStage = (stage, componentId) => {
            const current = selectedComposition();
            const candidates = (state.discovery.compositions || []).filter((composition) => {
                if (stage === 'generator') return compositionGenerator(composition) === componentId;
                if (stage === 'projector') return composition.pipeline.projector === componentId;
                if (stage === 'vocoder') return composition.pipeline.vocoder === componentId;
                return false;
            });
            const next = candidates.find((composition) => (
                composition.runnable
                && (!current || stage === 'generator'
                    || compositionGenerator(composition) === compositionGenerator(current))
            )) || candidates.find((composition) => composition.runnable) || candidates[0];
            if (!next) return;
            state.pathKey = next.id;
            state.presetId = '';
            renderPathSelector();
            renderSelectedPath();
        };
        byId('pipeline-generator').addEventListener('change', (event) => {
            selectCompositionForStage('generator', event.target.value);
        });
        byId('pipeline-projector').addEventListener('change', (event) => {
            selectCompositionForStage('projector', event.target.value);
        });
        byId('pipeline-vocoder').addEventListener('change', (event) => {
            selectCompositionForStage('vocoder', event.target.value);
        });
        document.querySelectorAll('.pipeline-stage[data-stage]').forEach((stage) => {
            stage.addEventListener('click', (event) => {
                if (event.target.closest('select')) return;
                renderStageInspector(selectedPath(), stage.dataset.stage);
            });
            stage.addEventListener('focusin', () => {
                renderStageInspector(selectedPath(), stage.dataset.stage);
            });
        });
        byId('variety').addEventListener('change', (event) => {
            const path = selectedPath();
            if (path) state.values.set(`variety:${path.id}`, event.target.value);
            markSpeechSelectionChanged();
        });
        byId('speaker').addEventListener('input', (event) => {
            const path = selectedPath();
            if (path) state.values.set(`speaker:${path.id}`, event.target.value);
            markSpeechSelectionChanged();
        });
        byId('select-mock-path').addEventListener('click', () => {
            const mock = state.discovery.compositions.find((path) => path.backend === 'mock');
            if (!mock) return;
            state.pathKey = mock.id;
            state.presetId = '';
            renderPathSelector();
            renderSelectedPath();
        });
        byId('verify-all-models').addEventListener('click', async (event) => {
            const button = event.currentTarget;
            button.disabled = true;
            byId('verification-status').textContent = 'Verifying changed installed models…';
            try {
                await verifyModelIds(pendingVerificationIds(state.discovery));
                byId('verification-status').textContent = 'Changed models verified.';
            } catch (error) {
                byId('verification-status').textContent = `Verification failed: ${error.message}`;
            } finally {
                renderInventory();
            }
        });
        byId('reload-speech-runtime').addEventListener('click', async (event) => {
            const button = event.currentTarget;
            button.disabled = true;
            clearError(byId('speech-runtime-errors'));
            try {
                const response = await fetch('/api/speech/runtime/reload', { method: 'POST' });
                if (!response.ok) throw new Error(await response.text());
                renderRuntime(await response.json());
                await refreshDiscovery();
            } catch (error) {
                showError(`Reload failed: ${error.message}`, byId('speech-runtime-errors'));
                byId('speech-runtime-state').dataset.state = 'failed';
                byId('speech-runtime-state').textContent = 'failed';
            } finally {
                button.disabled = false;
            }
        });
        byId('run-duplex-preview').addEventListener('click', async (event) => {
            const button = event.currentTarget;
            clearError(byId('duplex-error'));
            byId('speech-submit-status').textContent = '';
            button.disabled = true;
            try {
                const path = selectedPath();
                const variety = varietiesForPath(path).length === 1
                    ? varietiesForPath(path)[0].id
                    : byId('variety').value;
                const projection = await loadDuplexProjection(buildDuplexRequest({
                    text: byId('text').value,
                    mockAcoustics: byId('duplex-mock-acoustics').value,
                    variety,
                }));
                renderDuplexProjection(projection);
            } catch (error) {
                showError(`Duplex preview failed: ${error.message}`, byId('duplex-error'));
            } finally {
                button.disabled = false;
            }
        });
        byId('replay-duplex-journal').addEventListener('click', async (event) => {
            const button = event.currentTarget;
            clearError(byId('duplex-error'));
            byId('speech-submit-status').textContent = '';
            button.disabled = true;
            try {
                const projection = await loadDuplexProjection(buildDuplexRequest({
                    journalPath: byId('duplex-journal-path').value,
                }));
                renderDuplexProjection(projection);
            } catch (error) {
                showError(`Duplex replay failed: ${error.message}`, byId('duplex-error'));
            } finally {
                button.disabled = false;
            }
        });

        byId('show-pipeline').addEventListener('click', () => navigateWorkflow('compose'));
        byId('open-pipeline-in-speak').addEventListener('click', () => navigateWorkflow('speak'));
        for (const id of ['add-current-to-compare', 'add-pipeline-to-compare']) {
            byId(id).addEventListener('click', () => {
                navigateWorkflow('compare');
                const checkbox = [...byId('compare-candidates').querySelectorAll('input')]
                    .find((input) => input.value === state.pathKey);
                if (checkbox) checkbox.checked = true;
            });
        }
        byId('copy-compose-cli').addEventListener('click', async (event) => {
            await navigator.clipboard.writeText(byId('compose-cli').textContent);
            event.currentTarget.textContent = 'Copied';
        });
        byId('duplicate-recipe').addEventListener('click', () => {
            byId('compose-recipe-name').value = `${byId('compose-recipe-name').value || selectedPath()?.display_name} copy`;
            state.presetId = '';
            renderPathSelector();
            byId('delete-recipe').disabled = true;
        });
        byId('save-recipe').addEventListener('click', () => {
            const recipe = recipeSnapshot(byId('compose-recipe-name').value);
            if (!recipe) {
                showError('Select a complete pipeline before saving.', byId('compose-error'));
                return;
            }
            const existingIndex = state.userRecipes.findIndex(
                (candidate) => candidate.id === state.presetId,
            );
            if (existingIndex >= 0) {
                recipe.id = state.userRecipes[existingIndex].id;
                state.userRecipes.splice(existingIndex, 1, recipe);
            } else {
                state.userRecipes.push(recipe);
            }
            state.presetId = recipe.id;
            persistUserRecipes();
            renderPathSelector();
            renderCompareCandidates();
            byId('delete-recipe').disabled = false;
            byId('compose-test-status').textContent = `Saved ${recipe.name}.`;
        });
        byId('delete-recipe').addEventListener('click', () => {
            const result = deleteUserRecipe(state.userRecipes, state.presetId);
            if (!result.deleted) return;
            state.userRecipes = result.recipes;
            const preset = (state.discovery.presets || []).find(
                (candidate) => candidate.composition_id === state.pathKey,
            );
            state.presetId = preset?.id || '';
            persistUserRecipes();
            renderPathSelector();
            renderSelectedPath();
            byId('compose-test-status').textContent = (
                `Deleted saved copy ${result.deleted.name}. The built-in pipeline remains available.`
            );
        });
        byId('restore-recipe').addEventListener('click', () => {
            const recipe = state.userRecipes.find((candidate) => candidate.id === state.presetId);
            if (recipe) applyRecipe(recipe);
            else {
                const preset = state.discovery.presets.find(
                    (candidate) => candidate.composition_id === state.pathKey,
                );
                if (preset) {
                    state.presetId = preset.id;
                    renderSelectedPath();
                }
            }
            byId('compose-test-status').textContent = 'Recipe restored.';
        });
        byId('test-pipeline').addEventListener('click', async (event) => {
            const button = event.currentTarget;
            const path = selectedPath();
            if (!path?.runnable) {
                showError(
                    path?.unavailable_reason || 'The pipeline is not executable.',
                    byId('compose-error'),
                );
                return;
            }
            button.disabled = true;
            byId('compose-test-status').textContent = 'Pipeline test running.';
            try {
                const result = await requestSynthesis(
                    path,
                    state.values,
                    currentSynthesisContext(byId('text').value, path),
                );
                const audio = byId('compose-audio');
                if (audio.src) URL.revokeObjectURL(audio.src);
                audio.src = result.url;
                audio.classList.remove('hidden');
                byId('compose-test-status').textContent = 'Pipeline test completed.';
            } catch (error) {
                showError(`Pipeline test failed: ${error.message}`, byId('compose-error'));
                byId('compose-test-status').textContent = 'Pipeline test failed.';
            } finally {
                button.disabled = !selectedPath()?.runnable;
            }
        });
        byId('generate-all').addEventListener('click', () => {
            generateComparison().catch((error) => {
                showError(`Comparison failed: ${error.message}`, byId('compare-error'));
                byId('generate-all').disabled = false;
            });
        });
        byId('reveal-comparison').addEventListener('click', revealComparison);
        byId('save-preferred').addEventListener('click', () => {
            const result = state.compareResults.get(state.comparePreferred);
            if (!result?.recipe) return;
            const composition = state.discovery.compositions.find(
                (candidate) => candidate.id === result.recipe.compositionId,
            );
            if (!composition) return;
            state.pathKey = composition.id;
            const path = pathForComposition(composition);
            const recipe = recipeSnapshot(`${result.recipe.name} preferred`, path);
            recipe.controls = result.recipe.controls || {};
            recipe.variety = result.payload?.variety || recipe.variety;
            recipe.speaker = result.payload?.speaker || recipe.speaker;
            state.userRecipes.push(recipe);
            state.presetId = recipe.id;
            persistUserRecipes();
            renderPathSelector();
            renderCompareCandidates();
            byId('compare-status').textContent = `Saved ${recipe.name}.`;
        });
        byId('refresh-operate-jobs').addEventListener('click', () => {
            refreshOperateJobs().catch((error) => {
                byId('operate-jobs').textContent = `Activity unavailable: ${error.message}`;
            });
        });

        byId('synth-form').addEventListener('submit', async (event) => {
            event.preventDefault();
            clearError();
            const path = selectedPath();
            submit.disabled = true;
            submit.classList.add('loading');
            byId('speech-submit-status').textContent = 'Speech queued.';
            byId('speech-result-state').dataset.state = 'loading';
            byId('speech-result-state').textContent = 'Queued';
            byId('result-container').classList.add('hidden');
            hideDuplexResult();
            startRuntimePolling();
            try {
                byId('speech-submit-status').textContent = 'Speech synthesis running.';
                byId('speech-result-state').dataset.state = 'busy';
                byId('speech-result-state').textContent = 'Running';
                const result = await requestSynthesis(
                    path,
                    state.values,
                    currentSynthesisContext(byId('text').value, path),
                );
                if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
                state.audioUrl = result.url;
                renderResult(result.metadata, state.audioUrl);
                syncSpeechSelectionUrl({
                    runId: result.metadata.run_id,
                });
                byId('speech-result-state').dataset.state = 'ready';
                byId('speech-result-state').textContent = 'Completed';
                byId('speech-submit-status').textContent = 'Speech synthesis complete.';
                byId('audio-player').play().catch(() => {});
            } catch (error) {
                showError(`Synthesis failed: ${error.message}`);
                byId('speech-result-state').dataset.state = 'failed';
                byId('speech-result-state').textContent = 'Failed';
                byId('speech-submit-status').textContent = 'Speech synthesis failed.';
            } finally {
                stopRuntimePolling();
                submit.classList.remove('loading');
                submit.disabled = !selectedPath()?.runnable;
                loadRuntime().catch((error) => {
                    showError(`Runtime status unavailable: ${error.message}`, byId('speech-runtime-errors'));
                });
            }
        });

        loadRuntime().catch((error) => {
            byId('speech-runtime-state').dataset.state = 'failed';
            byId('speech-runtime-state').textContent = 'failed';
            showError(`Runtime status unavailable: ${error.message}`, byId('speech-runtime-errors'));
        });
    }

    return {
        availablePaths,
        applyStreamEnvelope,
        availableCompositions,
        beginLiveConversationTurn,
        buildPayload,
        buildDuplexRequest,
        compatibilityFor,
        compositionGenerator,
        controlDefault,
        controlsForPath,
        createStreamContractState,
        createLiveConversation,
        defaultControlValues,
        deleteUserRecipe,
        cliRepresentation,
        duplexDeepLinkWith,
        duplexLines,
        duplexRequestFromLocation,
        finishLiveAssistant,
        humanizeEvidenceValue,
        init,
        parseNumberArray,
        pathKey,
        pathLanguageCodes,
        pathSupportsSampleLanguage,
        pathForComposition,
        mergeDiscovery,
        mergeInventoryDiscovery,
        mergeSelectedResultIntoDiscovery,
        missingCatalogIds,
        comparisonSpeaker,
        mapWithConcurrency,
        pendingVerificationIds,
        preservesVerificationProgress,
        recipeSnapshot,
        renderDuplexProjection,
        restoreRecipeValues,
        savedRecipeModelIds,
        selectInitialPath,
        selectInitialComposition,
        sampleLanguagesForDiscovery,
        SPEECH_SAMPLES,
        setWorkflow,
        speechInstructionForPath,
        speechSelectionFromLocation,
        studioShell,
        liveProviderMessages,
        updateLiveAssistant,
        varietiesForPath,
        workflowForPath,
    };
}));
