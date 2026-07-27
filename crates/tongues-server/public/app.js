const byId = (id) => document.getElementById(id);

const customPages = [
    {
        title: 'Command Workbench',
        path: '/commands',
        group: 'Commands',
        summary: 'Discover, configure, run, cancel, and inspect browser-safe CLI commands.',
        capabilityIds: ['server-delivered Clap schema'],
        page: 'command-workbench',
    },
    {
        title: 'Speech Studio',
        path: '/speech',
        aliases: ['/styletts2'],
        group: 'Speech',
        summary: 'Generate speech with registered backends and verified model recipes.',
        capabilityIds: ['speak'],
        page: 'speech',
    },
    {
        title: 'Pronunciation Demo',
        path: '/pronunciation-demo',
        group: 'Speech',
        summary: 'Try spelling-to-pronunciation and Wiktionary pronunciation tasks.',
        capabilityIds: ['g2p2g/infer', 'wiktionary/infer'],
        page: 'pronunciation-demo',
    },
];

const V1_WORKFLOWS = [
    {
        title: 'Speak',
        path: '/speech',
        summary: 'Choose a ready recipe, enter text, generate speech, and inspect the resulting run.',
        clientRoute: true,
    },
    {
        title: 'Compose',
        path: '/speech/compose',
        summary: 'Inspect a typed recipe or continue into Graph Studio for safe graph editing.',
        clientRoute: true,
    },
    {
        title: 'Compare',
        path: '/speech/compare',
        summary: 'Run the same prompt through compatible recipes and compare their evidence.',
        clientRoute: true,
    },
    {
        title: 'Catalog',
        path: '/speech/catalog',
        summary: 'Find ready capabilities and recipes without loading the full component catalog.',
        clientRoute: true,
    },
    {
        title: 'Operate',
        path: '/speech/operate',
        summary: 'Follow readiness, verification, jobs, artifacts, failures, and runtime identity.',
        clientRoute: true,
    },
    {
        title: 'Advanced / Commands',
        path: '/commands',
        summary: 'Replay supported workflows through durable, schema-owned command pages.',
        clientRoute: true,
    },
    {
        title: 'Tracks / WaveDeck',
        path: '/runs',
        summary: 'Inspect immutable execution evidence, then open a provenance-preserving correction.',
        clientRoute: false,
    },
    {
        title: 'Live',
        path: '/speech/live',
        summary: 'Run an interruption-safe streamed conversation and retain its evidence.',
        clientRoute: true,
    },
];

let commandPages = [];
let cliCommands = [];
let activePage = null;
let activeJobId = null;
let activeJobSource = null;
let jobOutputLines = [];
let jobArtifacts = [];
let commandLevel = 'workflow';
const RECENT_COMMANDS_KEY = 'tongues.command-workbench.recent.v1';

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('/api/cli/schema');
        if (!response.ok) throw new Error(await response.text());
        const schema = await response.json();
        cliCommands = flattenCommands(schema.commands);
        commandPages = [...customPages, ...cliCommands];
    } catch (error) {
        console.error('Unable to load the Clap Web CLI schema', error);
        commandPages = [...customPages];
    }
    renderNavigation();
    renderRoute(false);
    window.addEventListener('popstate', () => renderRoute(true));
    initJobs();
    await initPronunciationDemo();
    await window.SpeechStudio.init();
});

function flattenCommands(commands, parentHelp = '') {
    return commands.flatMap((command) => {
        const help = command.help || parentHelp;
        const own = command.exposed ? [{
            ...command,
            title: titleCase(command.command.join(' ')),
            path: command.route,
            group: groupName(command.command[0]),
            summary: help || `Run tongues ${command.command.join(' ')}.`,
            capabilityIds: [command.id],
            page: 'command',
        }] : [];
        return [...own, ...flattenCommands(command.subcommands || [], help)];
    });
}

function groupName(name) {
    const names = {
        'g2p2g': 'G2P2G',
        'sentence-boundary': 'Sentence Parser',
        'head2phones': 'Head2Phones',
        'common-phone': 'Common Phone',
        'styletts2': 'StyleTTS2',
    };
    return names[name] || titleCase(name);
}

function titleCase(value) {
    return String(value)
        .split(/[- ]/)
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(' ');
}

function renderNavigation() {
    const nav = byId('primary-nav');
    nav.innerHTML = `
        <div class="nav-group">
            <div class="nav-heading">V1 workflows</div>
            ${V1_WORKFLOWS.map(workflowLink).join('')}
        </div>
        <div class="nav-group">
            <div class="nav-heading">Specialized workspaces</div>
            <a href="/studio/graphs/new">Graph Studio</a>
            <a href="/sessions/new/correct">WaveDeck</a>
            <a href="/pronunciation-demo" data-route="/pronunciation-demo">Pronunciation Demo</a>
        </div>
        <div class="nav-group">
            <div class="nav-heading">Runtime</div>
            <a href="/jobs" data-route="/jobs">Background Jobs</a>
        </div>`;
    nav.addEventListener('click', activateRouteLink);
}

function workflowLink(workflow) {
    const route = workflow.clientRoute ? ` data-route="${workflow.path}"` : '';
    return `<a href="${workflow.path}"${route}>${escapeHtml(workflow.title)}</a>`;
}

function activateRouteLink(event) {
    const link = event.target.closest('a[data-route]');
    if (!link || !shouldHandleClientNavigation(event, link)) return;
    event.preventDefault();
    navigateTo(link.getAttribute('href'));
}

function shouldHandleClientNavigation(event, link) {
    if (event.button !== 0) return false;
    return !event.defaultPrevented
        && !event.metaKey
        && !event.ctrlKey
        && !event.shiftKey
        && !event.altKey
        && !link.hasAttribute('download');
}

function navigateTo(path) {
    if (!path || `${window.location.pathname}${window.location.search}` === path) return;
    history.pushState({}, '', path);
    renderRoute(true);
}

function renderRoute(focus = false) {
    const path = normalizePath(window.location.pathname);
    const jobsRoute = path === '/jobs';
    const workbenchRoute = path === '/commands' || path.startsWith('/commands/');
    const commandId = workbenchRoute ? decodeURIComponent(path.slice('/commands/'.length)) : '';
    const selectedCommand = commandId
        ? cliCommands.find((candidate) => candidate.id === commandId)
        : null;
    const page = selectedCommand
        || commandPages.find((candidate) => candidate.path === path)
        || commandPages.find((candidate) => (candidate.aliases || []).includes(path))
        || commandPages.find((candidate) => candidate.page === 'speech' && path.startsWith('/speech/'));
    const speechRoute = page?.page === 'speech';
    const pronunciationRoute = page?.page === 'pronunciation-demo';
    const commandRoute = page?.page === 'command' || workbenchRoute;

    byId('speech-page').classList.toggle('hidden', !speechRoute);
    byId('pronunciation-demo-page').classList.toggle('hidden', !pronunciationRoute);
    byId('dashboard-page').classList.toggle('hidden', jobsRoute || Boolean(page) || workbenchRoute);
    byId('command-workbench-page').classList.toggle('hidden', !commandRoute);
    byId('jobs-page').classList.toggle('hidden', !jobsRoute);
    document.querySelectorAll('[data-route]').forEach((link) => {
        link.classList.toggle('active',
            link.dataset.route === page?.path
            || (workbenchRoute && link.dataset.route === '/commands')
            || (jobsRoute && link.dataset.route === '/jobs'));
    });

    if (jobsRoute) {
        activePage = null;
        setHeader('Runtime', 'Background Jobs', 'Check running commands, output, artifacts, or cancellation.', 'jobs');
        loadJobs();
        completeRouteTransition(focus);
        return;
    }
    if (!page) {
        if (workbenchRoute) {
            activePage = defaultWorkbenchCommand();
            setHeader('Command Workbench', 'Commands', 'Browser-safe commands from the current Clap schema.', 'tongues');
            renderWorkbench(activePage);
            completeRouteTransition(focus);
            return;
        }
        activePage = null;
        if (path !== '/') {
            setHeader('Route unavailable', 'Workspace not found', `No Tongues workspace is registered at ${path}. Choose a recovery path below.`, 'not found');
        } else {
            setHeader('Command surface', 'Tongues Web', 'Pick a workflow backed by the current CLI schema.', 'tongues');
        }
        renderDashboard();
        completeRouteTransition(focus);
        return;
    }
    activePage = page;
    if (speechRoute) {
        const speechTitles = {
            '/speech': 'Speak',
            '/speech/live': 'Live conversation',
            '/speech/compose': 'Compose',
            '/speech/compare': 'Compare',
            '/speech/catalog': 'Model catalog',
            '/speech/operate': 'Operate',
        };
        const title = speechTitles[path] || 'Speak';
        const identity = path === '/speech/live'
            ? 'Executing a live conversation and recording evidence.'
            : 'Configuring a speech workflow.';
        setHeader('Speech Studio', title, `${identity} ${page.summary}`, 'tongues speak');
        window.SpeechStudio?.setWorkflow(path, { focus });
    } else if (commandRoute) {
        setHeader('Command Workbench', page.title || 'Commands', page.summary || 'Browser-safe commands from Clap.', `tongues ${(page.command || []).join(' ')}`);
    } else {
        setHeader(page.group, page.title, page.summary, `tongues ${(page.command || []).join(' ')}`);
    }
    if (commandRoute) renderWorkbench(page.page === 'command' ? page : defaultWorkbenchCommand());
    completeRouteTransition(focus && !speechRoute);
}

function completeRouteTransition(focus) {
    document.title = `${byId('page-title').textContent} · Tongues`;
    byId('route-status').textContent = `Opened ${byId('page-title').textContent}. ${byId('page-summary').textContent}`;
    if (focus) {
        byId('page-title').setAttribute('tabindex', '-1');
        byId('page-title').focus();
    }
}

function setHeader(kicker, title, summary, command) {
    byId('page-kicker').textContent = kicker;
    byId('page-title').textContent = title;
    byId('page-summary').textContent = summary;
    byId('page-command').textContent = command;
}

function renderDashboard() {
    byId('dashboard-grid').innerHTML = `
        <section class="command-card" aria-labelledby="first-run-heading">
            <span>First run</span>
            <strong id="first-run-heading">Start with Speak</strong>
            <p>Choose a ready recipe, generate speech, then continue through Compose or Compare → Operate → Tracks / WaveDeck.</p>
            <a href="/speech" data-route="/speech">Begin the supported starter journey</a>
        </section>
        ${V1_WORKFLOWS.map((workflow) => {
        const route = workflow.clientRoute ? ` data-route="${workflow.path}"` : '';
        return `
            <a class="command-card" href="${workflow.path}"${route}>
                <span>V1 workflow</span>
                <strong>${escapeHtml(workflow.title)}</strong>
                <small>${escapeHtml(workflow.path)}</small>
                <p>${escapeHtml(workflow.summary)}</p>
            </a>`;
    }).join('')}`;
    byId('dashboard-grid').onclick = activateRouteLink;
}

function renderWorkbench(page) {
    const capabilityQuery = new URLSearchParams(window.location.search).get('capability');
    if (capabilityQuery && !byId('command-search').value) {
        byId('command-search').value = capabilityQuery;
        commandLevel = 'all';
    }
    if (page && commandLevel !== 'all' && page.presentation !== commandLevel) {
        commandLevel = page.presentation;
    }
    document.querySelectorAll('[data-command-level]').forEach((button) => {
        button.classList.toggle('active', button.dataset.commandLevel === commandLevel);
    });
    renderCommandResults();
    renderRecentCommands();
    if (page) renderCommandPage(page);
}

function defaultWorkbenchCommand() {
    return cliCommands.find((page) => page.presentation === 'workflow') || cliCommands[0] || null;
}

function renderCommandResults() {
    const query = byId('command-search').value.trim().toLowerCase();
    const matches = cliCommands.filter((page) => {
        if (commandLevel !== 'all' && page.presentation !== commandLevel) return false;
        return !query || [
            page.id, page.title, page.summary, ...(page.aliases || []),
            ...(page.arguments || []).flatMap((argument) => [argument.name, argument.help, ...argument.aliases]),
        ].join(' ').toLowerCase().includes(query);
    });
    byId('command-results').innerHTML = matches.length ? matches.map((page) => `
        <a href="${page.capability_href}" data-route="${page.capability_href}"
            class="command-result${page.id === activePage?.id ? ' active' : ''}">
            <strong>${escapeHtml(page.title)}</strong>
            <small>${escapeHtml(page.id)} · ${escapeHtml(page.presentation)}</small>
            <span>${escapeHtml(page.summary)}</span>
        </a>`).join('') : '<p class="empty-controls">No exposed command matches this search.</p>';
    byId('command-results').onclick = activateRouteLink;
}

function readRecentCommands() {
    try {
        return JSON.parse(localStorage.getItem(RECENT_COMMANDS_KEY) || '[]');
    } catch {
        return [];
    }
}

function rememberCommand(page) {
    const recent = [{ id: page.id, command: commandFromControls(page).join(' ') }]
        .concat(readRecentCommands().filter((item) => item.id !== page.id))
        .slice(0, 8);
    localStorage.setItem(RECENT_COMMANDS_KEY, JSON.stringify(recent));
    renderRecentCommands();
}

function renderRecentCommands() {
    const recent = readRecentCommands().filter((item) => cliCommands.some((page) => page.id === item.id));
    byId('recent-commands').innerHTML = recent.length ? recent.map((item) => `
        <a href="/commands/${escapeHtml(item.id)}" data-route="/commands/${escapeHtml(item.id)}">
            <strong>${escapeHtml(item.id)}</strong><code>${escapeHtml(item.command)}</code>
        </a>`).join('') : '<small>No commands run in this browser yet.</small>';
    byId('recent-commands').onclick = activateRouteLink;
}

function renderCommandPage(page) {
    activePage = page;
    const arguments_ = page.arguments || [];
    const primary = arguments_.filter((argument) => !argument.global);
    const advanced = arguments_.filter((argument) => argument.global);
    byId('command-preview').value = commandExample(page);
    byId('skeleton-doc').innerHTML = `
        <p>${escapeHtml(page.summary)}</p>
        <p class="cli-equivalent">
            <strong>${page.exposed ? 'Browser-safe exposure' : 'Documentation only'}</strong>
            · hierarchy <code>${escapeHtml(page.command.join(' → '))}</code>
            ${page.aliases?.length ? ` · aliases <code>${escapeHtml(page.aliases.join(', '))}</code>` : ''}
        </p>`;
    byId('skeleton-fields').innerHTML = primary.length
        ? primary.map(renderControl).join('')
        : '<p class="empty-controls">This command has no command-specific controls.</p>';
    byId('skeleton-advanced-fields').innerHTML = advanced.map(renderControl).join('');
    byId('skeleton-advanced').classList.toggle('hidden', advanced.length === 0);
    byId('command-capability-link').href = page.capability_href;
    byId('command-model-link').classList.toggle('hidden', !page.model_href);
    if (page.model_href) byId('command-model-link').href = page.model_href;
    byId('command-studio-link').classList.toggle('hidden', !page.studio_template);
    if (page.studio_template) {
        byId('command-studio-link').href = `/studio/graphs/new?starter=${encodeURIComponent(page.studio_template)}`;
    }
    document.querySelectorAll('#command-workbench-page [data-control]').forEach((node) => {
        node.addEventListener('input', () => {
            syncConflicts(page);
            byId('command-preview').value = commandFromControls(page).join(' ');
            saveFormState(page);
        });
    });
    restoreFormState(page);
    syncConflicts(page);
    byId('command-preview').value = commandFromControls(page).join(' ');
    renderCommandResults();
}

function renderControl(argument) {
    const metadata = [
        argument.aliases?.length ? `aliases ${argument.aliases.join(', ')}` : '',
        argument.defaults?.length ? `default ${argument.defaults.join(', ')}` : '',
        argument.conflicts?.length ? `conflicts with ${argument.conflicts.join(', ')}` : '',
        argument.required ? 'required' : 'optional',
        argument.cardinality?.repeatable ? 'repeatable' : '',
    ].filter(Boolean).join(' · ');
    const description = `
        <small>${escapeHtml(argument.help || 'CLI argument.')}</small>
        <small class="control-contract">${escapeHtml(`${argument.value_type} · ${metadata}`)}</small>`;
    const required = argument.required ? ' required' : '';
    if (argument.kind === 'flag') {
        return `
            <label class="checkbox-row control-checkbox">
                <input type="checkbox" data-control="${escapeHtml(argument.name)}" data-arg-id="${escapeHtml(argument.id)}">
                <span>${escapeHtml(argument.name)}${argument.required ? ' (required)' : ''}</span>
                ${description}
            </label>`;
    }
    if (argument.value_enum.length) {
        const empty = argument.required ? '' : '<option value="">Use CLI default</option>';
        return `
            <div class="form-group">
                <label>${escapeHtml(argument.name)}</label>
                <select data-control="${escapeHtml(argument.name)}" data-arg-id="${escapeHtml(argument.id)}"${required}>
                    ${empty}
                    ${argument.value_enum.map((value) => `
                        <option value="${escapeHtml(value)}"${argument.defaults.includes(value) ? ' selected' : ''}>${escapeHtml(titleCase(value))}</option>
                    `).join('')}
                </select>
                ${description}
            </div>`;
    }
    const value = argument.defaults[0] || '';
    const repeatable = argument.cardinality.repeatable;
    const input = repeatable
        ? `<textarea data-control="${escapeHtml(argument.name)}" data-arg-id="${escapeHtml(argument.id)}" placeholder="One value per line"${required}>${escapeHtml(argument.defaults.join('\n'))}</textarea>`
        : `<input type="text" data-control="${escapeHtml(argument.name)}" data-arg-id="${escapeHtml(argument.id)}" value="${escapeHtml(value)}" placeholder="${escapeHtml(argument.name)}"${required}>`;
    return `
        <div class="form-group">
            <label>${escapeHtml(argument.name)}${repeatable ? ' (repeatable)' : ''}</label>
            ${input}
            ${description}
        </div>`;
}

function syncConflicts(page) {
    const activeIds = new Set(
        [...document.querySelectorAll('#command-workbench-page [data-arg-id]')]
            .filter(hasControlValue)
            .map((node) => node.dataset.argId),
    );
    for (const argument of page.arguments || []) {
        const node = document.querySelector(`#command-workbench-page [data-arg-id="${cssEscape(argument.id)}"]`);
        if (!node || hasControlValue(node)) continue;
        node.disabled = argument.conflicts.some((id) => activeIds.has(id));
    }
}

function hasControlValue(node) {
    return node.type === 'checkbox' ? node.checked : Boolean(node.value.trim());
}

function commandExample(page) {
    const parts = ['tongues', ...page.command];
    for (const argument of page.arguments || []) {
        if (argument.kind === 'flag' || !argument.defaults.length) continue;
        const value = argument.defaults[0];
        if (argument.global) parts.splice(1, 0, argument.name, quoteArg(value));
        else if (argument.kind === 'positional') parts.push(quoteArg(value));
        else parts.push(argument.name, quoteArg(value));
    }
    return parts.join(' ');
}

function commandFromControls(page) {
    const global = [];
    const options = [];
    const positional = [];
    for (const argument of page.arguments || []) {
        const node = document.querySelector(`#command-workbench-page [data-arg-id="${cssEscape(argument.id)}"]`);
        if (!node || node.disabled) continue;
        const values = node.type === 'checkbox'
            ? (node.checked ? [''] : [])
            : node.value.split('\n').map((value) => value.trim()).filter(Boolean);
        for (const value of values) {
            const target = argument.global ? global : (argument.kind === 'positional' ? positional : options);
            if (argument.kind === 'positional') target.push(quoteArg(value));
            else if (argument.kind === 'flag') target.push(argument.name);
            else target.push(argument.name, quoteArg(value));
        }
    }
    return ['tongues', ...global, ...page.command, ...options, ...positional];
}

function buildJobRequest(page) {
    const rendered = commandFromControls(page);
    return {
        command: 'cargo',
        args: ['run', '--bin', 'tongues', '--', ...rendered.slice(1).map(unquoteArg)],
    };
}

function formState(page) {
    return Object.fromEntries((page.arguments || []).flatMap((argument) => {
        const node = document.querySelector(`#command-workbench-page [data-arg-id="${cssEscape(argument.id)}"]`);
        if (!node) return [];
        const value = node.type === 'checkbox' ? node.checked : node.value;
        return value === '' || value === false ? [] : [[argument.id, value]];
    }));
}

function saveFormState(page) {
    const url = new URL(window.location.href);
    url.search = '';
    for (const [id, value] of Object.entries(formState(page))) {
        url.searchParams.set(`arg.${id}`, String(value));
    }
    history.replaceState({}, '', `${url.pathname}${url.search}`);
}

function restoreFormState(page) {
    const query = new URLSearchParams(window.location.search);
    for (const argument of page.arguments || []) {
        const value = query.get(`arg.${argument.id}`);
        if (value === null) continue;
        const node = document.querySelector(`#command-workbench-page [data-arg-id="${cssEscape(argument.id)}"]`);
        if (!node) continue;
        if (node.type === 'checkbox') node.checked = value === 'true';
        else node.value = value;
    }
}

function initJobs() {
    byId('run-command').addEventListener('click', startCurrentPageJob);
    byId('cancel-command').addEventListener('click', cancelActiveJob);
    byId('refresh-jobs').addEventListener('click', loadJobs);
    byId('cancel-job').addEventListener('click', cancelActiveJob);
    byId('copy-command').addEventListener('click', async (event) => {
        await navigator.clipboard.writeText(byId('command-preview').value);
        event.currentTarget.textContent = 'Copied';
    });
    byId('command-search').addEventListener('input', renderCommandResults);
    document.querySelectorAll('[data-command-level]').forEach((button) => {
        button.addEventListener('click', () => {
            commandLevel = button.dataset.commandLevel;
            document.querySelectorAll('[data-command-level]').forEach((candidate) => {
                candidate.classList.toggle('active', candidate === button);
            });
            renderCommandResults();
        });
    });
    byId('output-mode').addEventListener('change', renderWorkbenchOutput);
    loadJobs();
}

async function startCurrentPageJob() {
    if (!activePage || activePage.page !== 'command') return;
    const request = buildJobRequest(activePage);
    const response = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: activePage.title, ...request }),
    });
    if (!response.ok) return alert(await response.text());
    const data = await response.json();
    rememberCommand(activePage);
    byId('workbench-job-status').textContent = `Running ${data.job_id}`;
    byId('cancel-command').classList.remove('hidden');
    jobOutputLines = [];
    jobArtifacts = [];
    await loadJobs();
    selectJob(data.job_id);
}

async function loadJobs() {
    const response = await fetch('/api/jobs');
    if (!response.ok) return;
    const jobs = await response.json();
    const list = byId('job-list');
    list.innerHTML = jobs.length ? jobs.map((job) => `
        <button type="button" class="job-item ${job.id === activeJobId ? 'active' : ''}" data-job-id="${job.id}">
            <span>${escapeHtml(job.label)}</span>
            <small>${escapeHtml(job.status)} · ${escapeHtml(job.progress.phase)}</small>
        </button>`).join('') : '<div class="empty-controls">No background jobs yet.</div>';
    list.querySelectorAll('[data-job-id]').forEach((button) => {
        button.addEventListener('click', () => selectJob(button.dataset.jobId));
    });
}

async function selectJob(jobId) {
    activeJobId = jobId;
    activeJobSource?.close();
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
    if (response.ok) {
        const detail = await response.json();
        renderJobDetail(detail.summary, detail.output || [], detail.artifacts || []);
    }
    activeJobSource = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
    activeJobSource.onmessage = (message) => applyJobEvent(JSON.parse(message.data));
}

function applyJobEvent(event) {
    if (event.type === 'snapshot') {
        renderJobDetail(event.summary, event.output || [], jobArtifacts);
        renderWorkbenchStatus(event.summary);
    }
    if (event.type === 'output') {
        jobOutputLines.push(event);
        renderJobOutput();
        renderWorkbenchOutput();
    }
    if (event.type === 'progress') renderProgress(event.progress);
    if (event.type === 'status') {
        renderJobSummary(event.summary);
        renderWorkbenchStatus(event.summary);
        loadJobs();
    }
}

function renderJobDetail(summary, output, artifacts) {
    jobOutputLines = output;
    jobArtifacts = artifacts;
    renderJobSummary(summary);
    renderJobOutput();
    renderWorkbenchStatus(summary);
    renderWorkbenchOutput();
    byId('job-artifacts').innerHTML = artifacts.length
        ? artifacts.map((artifact) => `<div class="artifact-row">${escapeHtml(artifact.path)}</div>`).join('')
        : '<div class="artifact-empty">Output files will appear here.</div>';
}

function renderJobSummary(summary) {
    byId('job-title').textContent = summary.label;
    byId('job-command').textContent = `${summary.command} ${summary.args.join(' ')}`;
    renderProgress(summary.progress, summary.status);
    byId('cancel-job').classList.toggle('hidden', summary.status !== 'running');
}

function renderProgress(progress, status = 'running') {
    const complete = ['succeeded', 'failed', 'canceled'].includes(status);
    const percent = progress.total
        ? Math.min(100, Math.round((progress.current || 0) / progress.total * 100))
        : (complete ? 100 : 35);
    byId('job-progress-bar').style.width = `${percent}%`;
    byId('job-progress-label').textContent = progress.total
        ? `${progress.phase}: ${progress.current || 0} / ${progress.total}`
        : progress.phase;
}

function renderJobOutput() {
    byId('job-output').textContent = jobOutputLines
        .slice(-500)
        .map((line) => `[${line.stream}] ${line.line}`)
        .join('\n');
}

function renderWorkbenchStatus(summary) {
    byId('workbench-job-status').textContent = `${summary.label}: ${summary.status} · ${summary.progress.phase}`;
    byId('cancel-command').classList.toggle('hidden', summary.status !== 'running');
}

function renderWorkbenchOutput() {
    const mode = byId('output-mode').value;
    const raw = jobOutputLines.map((line) => line.line).join('\n');
    if (mode === 'raw') {
        byId('workbench-output').textContent = raw;
        return;
    }
    if (mode === 'jsonl') {
        byId('workbench-output').textContent = jobOutputLines.map((line) => JSON.stringify(line)).join('\n');
        return;
    }
    if (mode === 'json') {
        const parsed = jobOutputLines.map((line) => {
            try {
                return JSON.parse(line.line);
            } catch {
                return { stream: line.stream, text: line.line };
            }
        });
        byId('workbench-output').textContent = JSON.stringify(parsed, null, 2);
        return;
    }
    byId('workbench-output').textContent = jobOutputLines
        .map((line) => JSON.stringify({ type: 'output', stream: line.stream, line: line.line, at_ms: line.at_ms }))
        .join('\n');
}

async function cancelActiveJob() {
    if (!activeJobId) return;
    const response = await fetch(`/api/jobs/${encodeURIComponent(activeJobId)}/cancel`, { method: 'POST' });
    if (!response.ok) alert(await response.text());
}

async function initPronunciationDemo() {
    const form = byId('pronunciation-form');
    const family = byId('pronunciation-family');
    const model = byId('pronunciation-model');
    const task = byId('pronunciation-task');
    const lang = byId('pronunciation-lang');
    const variety = byId('pronunciation-variety');
    const notation = byId('pronunciation-notation');
    let metadata;
    try {
        const response = await fetch('/api/pronunciation-demo/models');
        metadata = await response.json();
    } catch (error) {
        model.innerHTML = `<option>${escapeHtml(error.message)}</option>`;
        return;
    }
    const fill = (select, options, selected = '') => {
        select.innerHTML = options.map((option) => {
            const value = option.value || option.path;
            return `<option value="${escapeHtml(value)}"${value === selected ? ' selected' : ''}>${escapeHtml(option.label)}</option>`;
        }).join('');
    };
    const sync = () => {
        const wiktionary = family.value === 'wiktionary';
        document.querySelectorAll('.wiktionary-only').forEach((node) => node.classList.toggle('hidden', !wiktionary));
        fill(model, metadata.models.filter((item) => item.family === family.value));
        fill(task, wiktionary ? metadata.wiktionary_tasks : metadata.g2p2g_tasks);
        fill(lang, metadata.languages);
        fill(variety, metadata.varieties);
        fill(notation, metadata.notations, 'phones');
    };
    family.addEventListener('change', sync);
    sync();
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const response = await fetch('/api/pronunciation-demo/infer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                family: family.value,
                model: model.value,
                task: task.value,
                lang: lang.value,
                variety: variety.value,
                notation: notation.value,
                input: byId('pronunciation-input').value,
                raw: byId('pronunciation-raw').checked,
                cpu: byId('pronunciation-cpu').checked,
            }),
        });
        const result = response.ok ? await response.json() : { output: await response.text(), command: [] };
        byId('pronunciation-output').textContent = result.output || '(empty output)';
        byId('pronunciation-command').textContent = (result.command || []).join(' ');
        byId('pronunciation-source').textContent = result.source || '';
        byId('pronunciation-source-block').classList.toggle('hidden', !result.source);
        byId('pronunciation-result').classList.remove('hidden');
    });
}

function quoteArg(value) {
    const text = String(value);
    return /\s/.test(text) ? `"${text.replaceAll('"', '\\"')}"` : text;
}

function unquoteArg(value) {
    return value.startsWith('"') && value.endsWith('"')
        ? value.slice(1, -1).replaceAll('\\"', '"')
        : value;
}

function escapeHtml(value) {
    return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function cssEscape(value) {
    return window.CSS?.escape ? window.CSS.escape(value) : String(value).replaceAll('"', '\\"');
}

function normalizePath(path) {
    return path.length > 1 && path.endsWith('/') ? path.slice(0, -1) : path;
}
