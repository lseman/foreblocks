const API_BASE = '/api';
const STORAGE_THEME_KEY = 'mltracker_theme';
const STORAGE_VIEW_PREFS_PREFIX = 'mltracker_view_prefs_';
const STORAGE_SAVED_VIEWS_PREFIX = 'mltracker_saved_views_';
const PARALLEL_PANEL_HEIGHT_PX = 360;

const STATUS_ORDER = ['RUNNING', 'FINISHED', 'FAILED', 'CANCELED'];
const PALETTE = ['#f6a623', '#4f8ef7', '#34d399', '#fb7185', '#a78bfa', '#22d3ee', '#f59e0b', '#60a5fa'];

let metricsChart = null;
let sweepChart = null;
let compareChart = null;
let importanceChart = null;
let pinnedChartInstances = [];

const appState = {
    experiments: [],
    currentExperiment: null,
    currentRuns: [],
    experimentFilter: '',
    selectedRunIds: new Set(),
    pinnedCharts: [],
    sweepAxis: { x: null, y: null },
    runListRepaint: null,
    runDetailRepaint: null,
    currentViewSnapshot: null,
    paletteActions: [],
    paletteIndex: 0,
};

const els = {
    experimentList: document.getElementById('experiment-list'),
    contentShell: document.getElementById('content-shell'),
    runSidebar: document.getElementById('run-sidebar'),
    runSidebarMeta: document.getElementById('run-sidebar-meta'),
    runSidebarList: document.getElementById('run-sidebar-list'),
    runSidebarClear: document.getElementById('run-sidebar-clear'),
    viewContainer: document.getElementById('view-container'),
    breadcrumbs: document.getElementById('breadcrumbs'),
    refreshBtn: document.getElementById('refresh-btn'),
    themeToggle: document.getElementById('theme-toggle'),
    experimentSearch: document.getElementById('experiment-search'),
    topbarMeta: document.getElementById('topbar-meta'),
    savedViewsBtn: document.getElementById('saved-views-btn'),
    commandPaletteBtn: document.getElementById('command-palette-btn'),
    connectionPill: document.getElementById('connection-pill'),
    drawerOverlay: document.getElementById('drawer-overlay'),
    drawerContent: document.getElementById('drawer-content'),
    drawerClose: document.getElementById('drawer-close'),
    commandPaletteOverlay: document.getElementById('command-palette-overlay'),
    commandPaletteInput: document.getElementById('command-palette-input'),
    commandPaletteList: document.getElementById('command-palette-list'),
    commandPaletteClose: document.getElementById('command-palette-close'),
};

function setTopbarMeta(text) {
    if (els.topbarMeta) {
        els.topbarMeta.textContent = text;
    }
}

function setRunSidebarVisible(visible) {
    if (els.contentShell) {
        els.contentShell.classList.toggle('with-run-sidebar', Boolean(visible));
    }
    if (els.runSidebar) {
        els.runSidebar.classList.toggle('hidden', !visible);
    }
}

function renderRunSidebar(runs, compareBtn) {
    if (!els.runSidebarList) {
        return;
    }

    const selected = runs.filter((run) => appState.selectedRunIds.has(run.run_id)).length;
    if (els.runSidebarMeta) {
        els.runSidebarMeta.textContent = `${runs.length} runs · ${selected} selected`;
    }

    if (!runs.length) {
        els.runSidebarList.innerHTML = '<div class="run-rail-empty">No runs match current filters.</div>';
        return;
    }

    els.runSidebarList.innerHTML = runs.map((run) => {
        const status = run.status || 'UNKNOWN';
        const selectedClass = appState.selectedRunIds.has(run.run_id) ? 'is-selected' : '';
        return `
            <article class="run-rail-item ${selectedClass}" data-run-id="${escapeHtml(run.run_id)}">
                <input class="run-rail-check" type="checkbox" data-run-check="${escapeHtml(run.run_id)}" ${appState.selectedRunIds.has(run.run_id) ? 'checked' : ''} />
                <div class="run-rail-main">
                    <div class="run-rail-head">
                        <span class="run-rail-name" title="${escapeHtml(run.name || run.run_id)}">${escapeHtml(run.name || run.run_id)}</span>
                        <span class="run-rail-status">${escapeHtml(status)}</span>
                    </div>
                    <div class="run-rail-meta">${escapeHtml(run.run_id)} · ${escapeHtml(formatDuration(run.duration))}</div>
                </div>
            </article>
        `;
    }).join('');

    els.runSidebarList.querySelectorAll('.run-rail-item').forEach((node) => {
        node.addEventListener('click', () => {
            const runId = node.dataset.runId;
            if (runId) {
                loadRun(runId);
            }
        });
    });

    els.runSidebarList.querySelectorAll('.run-rail-check').forEach((checkbox) => {
        checkbox.addEventListener('click', (event) => event.stopPropagation());
        checkbox.addEventListener('change', () => {
            const runId = checkbox.dataset.runCheck;
            if (!runId) {
                return;
            }
            if (checkbox.checked) {
                appState.selectedRunIds.add(runId);
            } else {
                appState.selectedRunIds.delete(runId);
            }
            if (compareBtn) {
                updateCompareButton(compareBtn);
            }
            renderRunSidebar(runs, compareBtn);
        });
    });
}

async function fetchJSON(url, options = undefined) {
    const res = await fetch(url, options);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
        const detail = data && data.detail ? data.detail : `${res.status} ${res.statusText}`;
        throw new Error(detail);
    }
    return data;
}

async function fetchExperiments() {
    return fetchJSON(`${API_BASE}/experiments`);
}

async function fetchRuns(experimentName, q = null) {
    let url = `${API_BASE}/runs?experiment_name=${encodeURIComponent(experimentName)}&limit=300&include=full`;
    if (q) {
        url += `&q=${encodeURIComponent(q)}`;
    }
    const data = await fetchJSON(url);
    return data.runs || [];
}

async function fetchRunDetail(runId) {
    return fetchJSON(`${API_BASE}/runs/${runId}`);
}

async function fetchMetricHistory(runId, metricKey = null) {
    let url = `${API_BASE}/runs/${runId}/metrics/history`;
    if (metricKey) {
        url += `?metric_key=${encodeURIComponent(metricKey)}`;
    }
    return fetchJSON(url);
}

async function fetchArtifacts(runId) {
    const data = await fetchJSON(`${API_BASE}/runs/${runId}/artifacts`);
    return data.artifacts || [];
}

async function fetchArtifactJson(runId, artifactPath) {
    const res = await fetch(`${API_BASE}/runs/${runId}/artifacts/${encodeURIComponent(artifactPath).replace(/%2F/g, '/')}`);
    if (!res.ok) {
        throw new Error(`Artifact fetch failed: ${res.status}`);
    }
    return res.json();
}

async function fetchRunCompare(runIds) {
    return fetchJSON(`${API_BASE}/runs/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_ids: runIds }),
    });
}

async function checkHealth() {
    if (!els.connectionPill) {
        return;
    }
    els.connectionPill.className = 'connection-pill pending';
    els.connectionPill.textContent = 'Checking API';

    try {
        const health = await fetchJSON(`${API_BASE}/health`);
        const ok = health.status === 'healthy' || health.status === 'degraded';
        els.connectionPill.className = `connection-pill ${ok ? 'online' : 'offline'}`;
        els.connectionPill.textContent = ok ? 'API Online' : 'API Degraded';
    } catch {
        els.connectionPill.className = 'connection-pill offline';
        els.connectionPill.textContent = 'API Offline';
    }
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function parseTime(iso) {
    if (!iso) {
        return 0;
    }
    const t = new Date(iso).getTime();
    return Number.isFinite(t) ? t : 0;
}

function formatDate(iso) {
    if (!iso) {
        return '-';
    }
    try {
        const d = new Date(iso);
        return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    } catch {
        return iso;
    }
}

function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds <= 0) {
        return '-';
    }
    if (seconds < 60) {
        return `${seconds}s`;
    }
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins < 60) {
        return `${mins}m ${secs}s`;
    }
    const hrs = Math.floor(mins / 60);
    return `${hrs}h ${mins % 60}m`;
}

function formatMetricValue(value) {
    if (typeof value === 'number' && Number.isFinite(value)) {
        if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) {
            return value.toExponential(2);
        }
        return value.toFixed(4);
    }
    return String(value);
}

function debounce(fn, wait) {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn(...args), wait);
    };
}

function viewPrefsKey(experimentName) {
    return `${STORAGE_VIEW_PREFS_PREFIX}${experimentName}`;
}

function loadViewPrefs(experimentName) {
    try {
        const raw = localStorage.getItem(viewPrefsKey(experimentName));
        if (!raw) {
            return {};
        }
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === 'object' ? parsed : {};
    } catch {
        return {};
    }
}

function saveViewPrefs(experimentName, prefs) {
    try {
        localStorage.setItem(viewPrefsKey(experimentName), JSON.stringify(prefs));
    } catch {
        // no-op
    }
}

function savedViewsKey(experimentName) {
    return `${STORAGE_SAVED_VIEWS_PREFIX}${experimentName}`;
}

function loadSavedViews(experimentName) {
    try {
        const raw = localStorage.getItem(savedViewsKey(experimentName));
        if (!raw) {
            return [];
        }
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) {
            return [];
        }
        return parsed
            .filter((entry) => entry && typeof entry === 'object' && typeof entry.name === 'string' && entry.state)
            .slice(0, 25);
    } catch {
        return [];
    }
}

function saveSavedViews(experimentName, views) {
    try {
        localStorage.setItem(savedViewsKey(experimentName), JSON.stringify(views.slice(0, 25)));
    } catch {
        // no-op
    }
}

function parseNumericString(value) {
    if (value === null || value === undefined || value === '') {
        return '';
    }
    const n = Number(value);
    return Number.isFinite(n) ? n : '';
}

function parseBooleanString(value) {
    if (value === null || value === undefined || value === '') {
        return false;
    }
    return String(value).toLowerCase() === '1' || String(value).toLowerCase() === 'true';
}

function loadViewStateFromUrl(experimentName) {
    const params = new URLSearchParams(window.location.search);
    const exp = params.get('exp');
    if (exp && exp !== experimentName) {
        return {};
    }

    const selectedRaw = params.get('selected') || '';
    const selectedRunIds = selectedRaw
        ? selectedRaw.split(',').map((x) => x.trim()).filter(Boolean).slice(0, 80)
        : [];

    return {
        runQuery: params.get('q') || '',
        sortMode: params.get('sort') || '',
        groupMode: params.get('group') || '',
        activeStatus: params.get('status') || '',
        tagFilter: params.get('tag') || '',
        metricFilterKey: params.get('mkey') || '',
        metricMin: parseNumericString(params.get('mmin')),
        metricMax: parseNumericString(params.get('mmax')),
        onlySelected: parseBooleanString(params.get('only_selected')),
        leaderboardMetric: params.get('lb_metric') || '',
        leaderboardObjective: params.get('lb_obj') || '',
        parallelDims: (params.get('parallel') || '').split(',').map((x) => x.trim()).filter(Boolean).slice(0, 8),
        importanceTarget: params.get('importance') || '',
        selectedRunIds,
    };
}

function writeViewStateToUrl(experimentName, state) {
    const url = new URL(window.location.href);
    const params = url.searchParams;

    const setParam = (key, value) => {
        if (value === null || value === undefined || value === '' || value === false) {
            params.delete(key);
        } else {
            params.set(key, String(value));
        }
    };

    setParam('exp', experimentName);
    setParam('q', state.runQuery || '');
    setParam('sort', state.sortMode || '');
    setParam('group', state.groupMode || '');
    setParam('status', state.activeStatus && state.activeStatus !== 'ALL' ? state.activeStatus : '');
    setParam('tag', state.tagFilter || '');
    setParam('mkey', state.metricFilterKey || '');
    setParam('mmin', state.metricMin === '' ? '' : state.metricMin);
    setParam('mmax', state.metricMax === '' ? '' : state.metricMax);
    setParam('only_selected', state.onlySelected ? '1' : '');
    setParam('lb_metric', state.leaderboardMetric || '');
    setParam('lb_obj', state.leaderboardObjective && state.leaderboardObjective !== 'auto' ? state.leaderboardObjective : '');
    setParam('parallel', Array.isArray(state.parallelDims) ? state.parallelDims.join(',') : '');
    setParam('importance', state.importanceTarget || '');
    setParam('selected', Array.isArray(state.selectedRunIds) && state.selectedRunIds.length ? state.selectedRunIds.slice(0, 80).join(',') : '');

    window.history.replaceState({}, '', `${url.pathname}${params.toString() ? `?${params.toString()}` : ''}${url.hash}`);
}

function buildShareableViewUrl(experimentName, state) {
    const url = new URL(window.location.href);
    url.hash = `#exp/${encodeURIComponent(experimentName)}`;
    const params = url.searchParams;

    const setParam = (key, value) => {
        if (value === null || value === undefined || value === '' || value === false) {
            params.delete(key);
        } else {
            params.set(key, String(value));
        }
    };

    setParam('exp', experimentName);
    setParam('q', state.runQuery || '');
    setParam('sort', state.sortMode || '');
    setParam('group', state.groupMode || '');
    setParam('status', state.activeStatus && state.activeStatus !== 'ALL' ? state.activeStatus : '');
    setParam('tag', state.tagFilter || '');
    setParam('mkey', state.metricFilterKey || '');
    setParam('mmin', state.metricMin === '' ? '' : state.metricMin);
    setParam('mmax', state.metricMax === '' ? '' : state.metricMax);
    setParam('only_selected', state.onlySelected ? '1' : '');
    setParam('lb_metric', state.leaderboardMetric || '');
    setParam('lb_obj', state.leaderboardObjective && state.leaderboardObjective !== 'auto' ? state.leaderboardObjective : '');
    setParam('parallel', Array.isArray(state.parallelDims) ? state.parallelDims.join(',') : '');
    setParam('importance', state.importanceTarget || '');
    setParam('selected', Array.isArray(state.selectedRunIds) && state.selectedRunIds.length ? state.selectedRunIds.slice(0, 80).join(',') : '');

    return url.toString();
}

function sortRuns(runs, sortMode) {
    const data = [...runs];
    const duration = (r) => (Number.isFinite(r.duration) ? r.duration : -1);

    switch (sortMode) {
        case 'start_time':
            return data.sort((a, b) => parseTime(a.start_time) - parseTime(b.start_time));
        case '-duration':
            return data.sort((a, b) => duration(b) - duration(a));
        case 'duration':
            return data.sort((a, b) => duration(a) - duration(b));
        case '-start_time':
        default:
            return data.sort((a, b) => parseTime(b.start_time) - parseTime(a.start_time));
    }
}

function groupRuns(runs, groupMode) {
    if (groupMode === 'none') {
        return [{ label: null, runs }];
    }

    if (groupMode === 'status') {
        const buckets = new Map();
        runs.forEach((run) => {
            const key = run.status || 'UNKNOWN';
            if (!buckets.has(key)) {
                buckets.set(key, []);
            }
            buckets.get(key).push(run);
        });

        const known = STATUS_ORDER.filter((s) => buckets.has(s));
        const extra = [...buckets.keys()].filter((s) => !STATUS_ORDER.includes(s)).sort();
        return [...known, ...extra].map((label) => ({ label, runs: buckets.get(label) }));
    }

    if (groupMode === 'day') {
        const buckets = new Map();
        runs.forEach((run) => {
            const key = run.start_time ? String(run.start_time).slice(0, 10) : 'Unknown day';
            if (!buckets.has(key)) {
                buckets.set(key, []);
            }
            buckets.get(key).push(run);
        });

        return [...buckets.keys()]
            .sort((a, b) => b.localeCompare(a))
            .map((label) => ({ label, runs: buckets.get(label) }));
    }

    return [{ label: null, runs }];
}

function setTheme(theme) {
    const safeTheme = theme === 'light' ? 'light' : 'dark';
    document.body.dataset.theme = safeTheme;
    localStorage.setItem(STORAGE_THEME_KEY, safeTheme);

    if (els.themeToggle) {
        els.themeToggle.textContent = safeTheme === 'dark' ? '☀' : '☾';
    }

    if (metricsChart) {
        metricsChart.destroy();
        metricsChart = null;
    }
    if (sweepChart) {
        sweepChart.destroy();
        sweepChart = null;
    }
    if (compareChart) {
        compareChart.destroy();
        compareChart = null;
    }
    if (importanceChart) {
        importanceChart.destroy();
        importanceChart = null;
    }
    destroyPinnedCharts();

    const refreshTasks = [];
    if (typeof appState.runListRepaint === 'function') {
        refreshTasks.push(Promise.resolve(appState.runListRepaint()));
    }
    if (typeof appState.runDetailRepaint === 'function') {
        refreshTasks.push(Promise.resolve(appState.runDetailRepaint()));
    }
    if (refreshTasks.length) {
        Promise.allSettled(refreshTasks).catch(() => {
            // no-op
        });
    }
}

async function copyTextToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch {
        return false;
    }
}

function closeCommandPalette() {
    if (!els.commandPaletteOverlay) {
        return;
    }
    els.commandPaletteOverlay.classList.add('hidden');
    if (els.commandPaletteInput) {
        els.commandPaletteInput.value = '';
    }
    appState.paletteActions = [];
    appState.paletteIndex = 0;
}

function renderCommandPaletteActions() {
    if (!els.commandPaletteList) {
        return;
    }
    const actions = appState.paletteActions;
    if (!actions.length) {
        els.commandPaletteList.innerHTML = '<div class="palette-empty">No commands match your search.</div>';
        return;
    }

    els.commandPaletteList.innerHTML = actions.map((action, idx) => {
        const active = idx === appState.paletteIndex ? 'is-active' : '';
        return `
            <button type="button" class="palette-item ${active}" data-action-index="${idx}">
                <div class="palette-item-title">${escapeHtml(action.title)}</div>
                <div class="palette-item-meta">${escapeHtml(action.meta || '')}</div>
            </button>
        `;
    }).join('');

    els.commandPaletteList.querySelectorAll('.palette-item').forEach((node) => {
        node.addEventListener('click', async () => {
            const idx = Number(node.dataset.actionIndex);
            if (!Number.isFinite(idx) || !appState.paletteActions[idx]) {
                return;
            }
            await appState.paletteActions[idx].run();
        });
    });
}

function buildCommandPaletteActions(query = '') {
    const q = String(query || '').trim().toLowerCase();
    const actions = [];

    actions.push({
        title: 'Refresh Dashboard',
        meta: 'Reload current experiment or run view',
        run: async () => {
            closeCommandPalette();
            if (appState.currentExperiment) {
                await loadExperiment(appState.currentExperiment);
            } else {
                await loadApp();
            }
        },
    });

    actions.push({
        title: 'Toggle Theme',
        meta: 'Switch between dark and light mode',
        run: async () => {
            const next = document.body.dataset.theme === 'dark' ? 'light' : 'dark';
            setTheme(next);
            closeCommandPalette();
        },
    });

    if (appState.currentExperiment && appState.currentViewSnapshot) {
        actions.push({
            title: 'Copy Shareable View URL',
            meta: 'Copies URL with filters, sorting, and selected runs',
            run: async () => {
                const url = buildShareableViewUrl(appState.currentExperiment, appState.currentViewSnapshot);
                const ok = await copyTextToClipboard(url);
                closeCommandPalette();
                if (!ok) {
                    window.prompt('Copy this URL:', url);
                }
            },
        });

        actions.push({
            title: 'Save Current View',
            meta: 'Store this view preset locally for this experiment',
            run: async () => {
                const now = new Date();
                const defaultName = `view-${now.toISOString().slice(0, 16).replace('T', ' ')}`;
                const name = window.prompt('Saved view name:', defaultName);
                if (!name || !name.trim()) {
                    return;
                }
                const existing = loadSavedViews(appState.currentExperiment);
                const next = [{
                    name: name.trim(),
                    createdAt: now.toISOString(),
                    state: appState.currentViewSnapshot,
                }, ...existing.filter((x) => x.name !== name.trim())].slice(0, 25);
                saveSavedViews(appState.currentExperiment, next);
                closeCommandPalette();
            },
        });

        const saved = loadSavedViews(appState.currentExperiment);
        saved.forEach((entry) => {
            actions.push({
                title: `Load View: ${entry.name}`,
                meta: entry.createdAt ? `Saved ${formatDate(entry.createdAt)}` : 'Saved view preset',
                run: async () => {
                    if (!appState.currentExperiment || !entry.state) {
                        return;
                    }
                    writeViewStateToUrl(appState.currentExperiment, entry.state);
                    closeCommandPalette();
                    await loadExperiment(appState.currentExperiment);
                },
            });
        });
    }

    if (!q) {
        return actions;
    }

    return actions.filter((item) => {
        const haystack = `${item.title} ${item.meta || ''}`.toLowerCase();
        return haystack.includes(q);
    });
}

function openCommandPalette(initialQuery = '') {
    if (!els.commandPaletteOverlay || !els.commandPaletteInput) {
        return;
    }
    els.commandPaletteOverlay.classList.remove('hidden');
    els.commandPaletteInput.value = initialQuery;
    appState.paletteActions = buildCommandPaletteActions(initialQuery);
    appState.paletteIndex = 0;
    renderCommandPaletteActions();
    els.commandPaletteInput.focus();
    els.commandPaletteInput.select();
}

function initTheme() {
    const stored = localStorage.getItem(STORAGE_THEME_KEY);
    if (stored === 'light' || stored === 'dark') {
        setTheme(stored);
        return;
    }
    const prefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;
    setTheme(prefersLight ? 'light' : 'dark');
}

function statusColor(status) {
    switch (status) {
        case 'RUNNING':
            return '#4f8ef7';
        case 'FINISHED':
            return '#34d399';
        case 'FAILED':
            return '#f87171';
        case 'CANCELED':
            return '#fbbf24';
        default:
            return 'rgba(240,240,242,0.38)';
    }
}

function hashColor(seed) {
    let hash = 0;
    for (let i = 0; i < seed.length; i += 1) {
        hash = ((hash << 5) - hash) + seed.charCodeAt(i);
        hash |= 0;
    }
    return PALETTE[Math.abs(hash) % PALETTE.length];
}

function destroyPinnedCharts() {
    pinnedChartInstances.forEach((chart) => {
        try {
            chart.destroy();
        } catch {
            // no-op
        }
    });
    pinnedChartInstances = [];
}

function pinKey(experimentName, runId, metricKey) {
    return `${experimentName}::${runId}::${metricKey}`;
}

function renderExperiments(experiments) {
    els.experimentList.innerHTML = '';
    const query = (appState.experimentFilter || '').trim().toLowerCase();
    const visible = query
        ? experiments.filter((exp) => String(exp.name || '').toLowerCase().includes(query))
        : experiments;

    // Update sidebar workspace counter
    const metaEl = document.getElementById('workspace-meta');
    if (metaEl) {
        const total = experiments.length;
        const shown = visible.length;
        metaEl.textContent = query
            ? `${shown}/${total} shown`
            : `${total} experiment${total !== 1 ? 's' : ''}`;
    }

    if (!visible.length) {
        const li = document.createElement('li');
        li.className = 'loading';
        li.textContent = query ? 'No experiments match your search.' : 'No experiments found.';
        els.experimentList.appendChild(li);
        return;
    }

    visible.forEach((exp) => {
        const li = document.createElement('li');
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'nav-item';
        const isActive = appState.currentExperiment === exp.name;
        if (isActive) btn.classList.add('active');

        const runCount = exp.run_count ?? exp.runs ?? null;
        const countBadge = runCount != null
            ? `<span class="exp-run-count">${runCount}</span>`
            : '';
        btn.innerHTML = `<span class="exp-name-text">${exp.name}</span>${countBadge}`;
        btn.setAttribute('title', exp.name);
        btn.onclick = () => loadExperiment(exp.name);
        li.appendChild(btn);
        els.experimentList.appendChild(li);
    });
}

function renderRunSummary(runs, mount) {
    const total = runs.length;
    const running = runs.filter((r) => r.status === 'RUNNING').length;
    const failed = runs.filter((r) => r.status === 'FAILED').length;
    const finished = runs.filter((r) => r.status === 'FINISHED').length;
    const durations = runs
        .map((r) => (Number.isFinite(r.duration) ? r.duration : 0))
        .filter((v) => v > 0);
    const avgDuration = durations.length
        ? Math.round(durations.reduce((a, b) => a + b, 0) / durations.length)
        : 0;

    mount.innerHTML = `
        <article class="summary-card">
            <div class="summary-label">Total Runs</div>
            <div class="summary-value">${total}</div>
        </article>
        <article class="summary-card">
            <div class="summary-label">Running</div>
            <div class="summary-value">${running}</div>
        </article>
        <article class="summary-card">
            <div class="summary-label">Failed</div>
            <div class="summary-value">${failed}</div>
        </article>
        <article class="summary-card">
            <div class="summary-label">Avg Duration</div>
            <div class="summary-value">${formatDuration(avgDuration)}</div>
        </article>
    `;

    if (failed === 0) {
        mount.children[2].querySelector('.summary-label').textContent = 'Finished';
        mount.children[2].querySelector('.summary-value').textContent = String(finished);
    }
}

function getRunStatus(run) {
    return run.status || 'UNKNOWN';
}

function renderStatusFilters(runs, mount, activeStatus, onChange) {
    if (!mount) {
        return activeStatus;
    }

    const counts = new Map();
    runs.forEach((run) => {
        const status = getRunStatus(run);
        counts.set(status, (counts.get(status) || 0) + 1);
    });

    const known = STATUS_ORDER.filter((status) => counts.has(status));
    const extra = [...counts.keys()].filter((status) => !STATUS_ORDER.includes(status)).sort();
    const statusKeys = [...known, ...extra];
    const options = [{ key: 'ALL', count: runs.length }, ...statusKeys.map((key) => ({ key, count: counts.get(key) || 0 }))];

    let safeActive = activeStatus || 'ALL';
    if (safeActive !== 'ALL' && !counts.has(safeActive)) {
        safeActive = 'ALL';
    }

    mount.innerHTML = options.map((item) => `
        <button type="button" class="status-filter-btn ${item.key === safeActive ? 'active' : ''}" data-status="${escapeHtml(item.key)}">
            <span>${escapeHtml(item.key)}</span>
            <b>${item.count}</b>
        </button>
    `).join('');

    mount.querySelectorAll('.status-filter-btn').forEach((btn) => {
        btn.addEventListener('click', () => onChange(btn.dataset.status || 'ALL'));
    });

    return safeActive;
}

function pickPrimaryMetricKey(runs) {
    const counts = new Map();
    runs.forEach((run) => {
        Object.entries(run.metrics || {}).forEach(([key, value]) => {
            if (typeof value === 'number' && Number.isFinite(value)) {
                counts.set(key, (counts.get(key) || 0) + 1);
            }
        });
    });

    if (!counts.size) {
        return null;
    }

    const preference = ['val_loss', 'valid_loss', 'loss', 'val_rmse', 'rmse', 'val_mae', 'mae', 'mse', 'accuracy', 'acc', 'f1', 'auc'];
    const rank = (k) => {
        const idx = preference.indexOf(String(k).toLowerCase());
        return idx === -1 ? 999 : idx;
    };

    const ranked = [...counts.entries()].sort((a, b) => {
        if (b[1] !== a[1]) {
            return b[1] - a[1];
        }
        return rank(a[0]) - rank(b[0]);
    });

    return ranked[0][0];
}

function metricLowerIsBetter(metricKey) {
    const lowerHints = /(loss|error|rmse|mae|mse|mape|nll|perplexity|wer|cer)/i;
    return lowerHints.test(metricKey);
}

function formatCompactDate(iso) {
    if (!iso) {
        return '-';
    }
    try {
        return new Date(iso).toLocaleString([], {
            month: 'short',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
        });
    } catch {
        return iso;
    }
}

function renderInsightsStrip(runs, mount) {
    if (!mount) {
        return;
    }

    if (!runs.length) {
        mount.innerHTML = '';
        return;
    }

    const total = runs.length;
    const finished = runs.filter((r) => getRunStatus(r) === 'FINISHED').length;
    const successRate = total ? (finished / total) * 100 : 0;

    const latest = [...runs].sort((a, b) => parseTime(b.start_time) - parseTime(a.start_time))[0];
    const last24h = runs.filter((r) => {
        const t = parseTime(r.start_time);
        return t > 0 && t >= (Date.now() - 24 * 60 * 60 * 1000);
    }).length;

    const primaryMetric = pickPrimaryMetricKey(runs);
    let metricText = 'No numeric metric';
    let metricSub = 'Log metrics to unlock ranking';
    if (primaryMetric) {
        const values = runs
            .map((run) => ({
                run,
                value: run.metrics ? run.metrics[primaryMetric] : null,
            }))
            .filter((entry) => typeof entry.value === 'number' && Number.isFinite(entry.value));
        if (values.length) {
            const lowerBetter = metricLowerIsBetter(primaryMetric);
            values.sort((a, b) => lowerBetter ? a.value - b.value : b.value - a.value);
            const best = values[0];
            metricText = `${primaryMetric}: ${formatMetricValue(best.value)}`;
            metricSub = `Best run: ${best.run.name || best.run.run_id}`;
        }
    }

    mount.innerHTML = `
        <article class="insight-card">
            <div class="insight-label">Primary Signal</div>
            <div class="insight-value">${escapeHtml(metricText)}</div>
            <div class="insight-note">${escapeHtml(metricSub)}</div>
        </article>
        <article class="insight-card">
            <div class="insight-label">Execution Quality</div>
            <div class="insight-value">${successRate.toFixed(1)}%</div>
            <div class="insight-note">${finished}/${total} runs finished</div>
        </article>
        <article class="insight-card">
            <div class="insight-label">Recent Activity</div>
            <div class="insight-value">${last24h} runs / 24h</div>
            <div class="insight-note">Latest: ${escapeHtml(latest.name || latest.run_id)} · ${escapeHtml(formatCompactDate(latest.start_time))}</div>
        </article>
    `;
}

function normalizeText(value) {
    return String(value || '').trim().toLowerCase();
}

function runMatchesTagFilter(run, tagFilter) {
    const query = normalizeText(tagFilter);
    if (!query) {
        return true;
    }

    const tags = run.tags || {};
    const entries = Object.entries(tags);

    if (query.includes(':')) {
        const [rawKey, rawVal = ''] = query.split(':', 2);
        const keyQuery = normalizeText(rawKey);
        const valQuery = normalizeText(rawVal);

        if (!keyQuery) {
            return true;
        }

        for (const [k, v] of entries) {
            const keyNorm = normalizeText(k);
            if (!keyNorm.includes(keyQuery)) {
                continue;
            }
            if (!valQuery || normalizeText(v).includes(valQuery)) {
                return true;
            }
        }
        return false;
    }

    for (const [k, v] of entries) {
        if (normalizeText(k).includes(query) || normalizeText(v).includes(query)) {
            return true;
        }
    }
    return false;
}

function metricValueForRun(run, metricKey) {
    if (!metricKey || !run.metrics) {
        return null;
    }
    const value = run.metrics[metricKey];
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function applyAdvancedFilters(
    runs,
    {
        tagFilter,
        metricFilterKey,
        metricMin,
        metricMax,
        onlySelected,
        selectedRunIds,
    },
) {
    const hasMin = metricMin !== '' && metricMin !== null && Number.isFinite(Number(metricMin));
    const hasMax = metricMax !== '' && metricMax !== null && Number.isFinite(Number(metricMax));
    const minVal = hasMin ? Number(metricMin) : null;
    const maxVal = hasMax ? Number(metricMax) : null;
    const useMetricFilter = Boolean(metricFilterKey) && (hasMin || hasMax);

    return runs.filter((run) => {
        if (onlySelected && !selectedRunIds.has(run.run_id)) {
            return false;
        }
        if (!runMatchesTagFilter(run, tagFilter)) {
            return false;
        }
        if (!useMetricFilter) {
            return true;
        }
        const metricValue = metricValueForRun(run, metricFilterKey);
        if (metricValue === null) {
            return false;
        }
        if (minVal !== null && metricValue < minVal) {
            return false;
        }
        if (maxVal !== null && metricValue > maxVal) {
            return false;
        }
        return true;
    });
}

function setMetricSelectOptions(select, metricKeys, selectedValue, anyLabel) {
    if (!select) {
        return selectedValue || '';
    }

    const options = [`<option value="">${escapeHtml(anyLabel)}</option>`]
        .concat(metricKeys.map((key) => `<option value="${escapeHtml(key)}">${escapeHtml(key)}</option>`));
    select.innerHTML = options.join('');

    const safeValue = selectedValue && metricKeys.includes(selectedValue) ? selectedValue : '';
    select.value = safeValue;
    return safeValue;
}

function renderLeaderboardSection(
    runs,
    section,
    metricSelect,
    objectiveSelect,
    listMount,
    metricKey,
    objectiveMode,
) {
    if (!section || !metricSelect || !objectiveSelect || !listMount) {
        return { metricKey, objectiveMode };
    }

    const metricKeys = extractNumericMetricKeys(runs);
    if (!metricKeys.length) {
        section.classList.add('module-muted');
        metricSelect.innerHTML = '<option value="">No metric available</option>';
        metricSelect.disabled = true;
        objectiveSelect.disabled = true;
        listMount.innerHTML = '<li class="module-empty">Need numeric metrics to build a leaderboard.</li>';
        return { metricKey: '', objectiveMode };
    }

    section.classList.remove('module-muted');
    metricSelect.disabled = false;
    objectiveSelect.disabled = false;

    const preferred = pickPrimaryMetricKey(runs);
    const safeMetric = metricKeys.includes(metricKey) ? metricKey : (preferred || metricKeys[0]);
    metricSelect.innerHTML = metricKeys.map((k) => `<option value="${escapeHtml(k)}">${escapeHtml(k)}</option>`).join('');
    metricSelect.value = safeMetric;

    const safeObjective = ['auto', 'min', 'max'].includes(objectiveMode) ? objectiveMode : 'auto';
    objectiveSelect.value = safeObjective;
    const resolvedObjective = safeObjective === 'auto'
        ? (metricLowerIsBetter(safeMetric) ? 'min' : 'max')
        : safeObjective;

    const scored = runs
        .map((run) => ({
            run,
            value: metricValueForRun(run, safeMetric),
        }))
        .filter((entry) => entry.value !== null);

    if (!scored.length) {
        listMount.innerHTML = '<li class="module-empty">No runs have values for this metric.</li>';
        return { metricKey: safeMetric, objectiveMode: safeObjective };
    }

    scored.sort((a, b) => (
        resolvedObjective === 'min'
            ? a.value - b.value
            : b.value - a.value
    ));
    const top = scored.slice(0, 8);
    const values = top.map((e) => e.value);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const spread = Math.max(1e-12, maxVal - minVal);

    listMount.innerHTML = top.map((entry, idx) => {
        const ratio = resolvedObjective === 'min'
            ? (maxVal - entry.value) / spread
            : (entry.value - minVal) / spread;
        const width = maxVal === minVal ? 92 : Math.max(18, Math.round(20 + ratio * 72));

        return `
            <li class="leaderboard-item">
                <div class="leaderboard-rank">#${idx + 1}</div>
                <button type="button" class="leaderboard-run-btn" data-run-id="${escapeHtml(entry.run.run_id)}">
                    <span>${escapeHtml(entry.run.name || entry.run.run_id)}</span>
                    <small>${escapeHtml(entry.run.run_id)}</small>
                </button>
                <div class="leaderboard-bar-wrap">
                    <div class="leaderboard-bar" style="width:${width}%"></div>
                </div>
                <div class="leaderboard-value">${escapeHtml(formatMetricValue(entry.value))}</div>
            </li>
        `;
    }).join('');

    listMount.querySelectorAll('.leaderboard-run-btn').forEach((btn) => {
        btn.addEventListener('click', () => loadRun(btn.dataset.runId));
    });

    return { metricKey: safeMetric, objectiveMode: safeObjective };
}

function dimensionLabel(key) {
    return key === '__duration__' ? 'duration' : key;
}

function collectParallelDimensions(runs) {
    const dims = extractNumericMetricKeys(runs);
    const hasDuration = runs.some((run) => Number.isFinite(run.duration));
    return hasDuration ? ['__duration__', ...dims] : dims;
}

function parallelDimensionValue(run, key) {
    if (key === '__duration__') {
        return Number.isFinite(run.duration) ? run.duration : null;
    }
    return metricValueForRun(run, key);
}

function truncateLabel(value, maxLen = 16) {
    const text = String(value);
    if (text.length <= maxLen) {
        return text;
    }
    return `${text.slice(0, maxLen - 1)}…`;
}

function defaultParallelDimensions(availableDims, primaryMetric) {
    const ordered = [];
    if (availableDims.includes('__duration__')) {
        ordered.push('__duration__');
    }
    if (primaryMetric && availableDims.includes(primaryMetric) && !ordered.includes(primaryMetric)) {
        ordered.push(primaryMetric);
    }
    availableDims.forEach((dim) => {
        if (!ordered.includes(dim)) {
            ordered.push(dim);
        }
    });
    return ordered.slice(0, Math.min(5, ordered.length));
}

function drawParallelCoordinatesCanvas(runs, dims, canvas) {
    const ctx = canvas ? canvas.getContext('2d') : null;
    if (!ctx) {
        return;
    }

    const host = canvas.parentElement;
    const hostWidth = host ? host.clientWidth : canvas.clientWidth;
    const targetHeight = PARALLEL_PANEL_HEIGHT_PX;

    if (host) {
        host.style.minHeight = `${targetHeight}px`;
        host.style.height = `${targetHeight}px`;
        host.style.maxHeight = `${targetHeight}px`;
    }

    const cssWidth = Math.max(320, hostWidth || 680);
    const cssHeight = targetHeight;
    const dpr = window.devicePixelRatio || 1;
    const widthPx = Math.round(cssWidth * dpr);
    const heightPx = Math.round(cssHeight * dpr);
    if (canvas.width !== widthPx || canvas.height !== heightPx) {
        canvas.width = widthPx;
        canvas.height = heightPx;
    }

    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.display = 'block';

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    if (!dims.length || dims.length < 2 || !runs.length) {
        return;
    }

    const textPrimary = getComputedStyle(document.body).getPropertyValue('--text-primary').trim() || '#e2e8f0';
    const textMuted = getComputedStyle(document.body).getPropertyValue('--text-muted').trim() || '#94a3b8';
    const axisColor = getComputedStyle(document.body).getPropertyValue('--border-color').trim() || 'rgba(127,152,176,0.3)';

    const margin = { left: 34, right: 24, top: 24, bottom: 34 };
    const plotW = cssWidth - margin.left - margin.right;
    const plotH = cssHeight - margin.top - margin.bottom;
    const axisStep = dims.length > 1 ? plotW / (dims.length - 1) : plotW;
    const runCount = Math.max(1, runs.length);
    const baseLineWidth = 0.4;
    const selectedLineWidth = 1.05;
    const baseAlpha = Math.max(0.025, Math.min(0.12, 1.6 / Math.sqrt(runCount)));
    const selectedAlpha = 0.72;

    const ranges = dims.map((dim) => {
        const values = runs
            .map((run) => parallelDimensionValue(run, dim))
            .filter((v) => typeof v === 'number' && Number.isFinite(v));
        if (!values.length) {
            return { dim, min: 0, max: 1 };
        }
        const min = Math.min(...values);
        const max = Math.max(...values);
        return { dim, min, max };
    });

    ctx.font = '11px IBM Plex Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    ranges.forEach((range, idx) => {
        const x = margin.left + idx * axisStep;
        ctx.strokeStyle = axisColor;
        ctx.lineWidth = 0.65;
        ctx.beginPath();
        ctx.moveTo(x, margin.top);
        ctx.lineTo(x, margin.top + plotH);
        ctx.stroke();

        ctx.fillStyle = textPrimary;
        ctx.fillText(truncateLabel(dimensionLabel(range.dim), 18), x, cssHeight - 12);

        ctx.fillStyle = textMuted;
        ctx.font = '10px IBM Plex Mono, monospace';
        ctx.fillText(formatMetricValue(range.max), x, margin.top - 10);
        ctx.fillText(formatMetricValue(range.min), x, margin.top + plotH + 12);
        ctx.font = '11px IBM Plex Mono, monospace';
    });

    let plotted = 0;
    runs.forEach((run) => {
        const points = [];
        for (let i = 0; i < ranges.length; i += 1) {
            const range = ranges[i];
            const value = parallelDimensionValue(run, range.dim);
            if (value === null || !Number.isFinite(value)) {
                return;
            }
            const denom = Math.abs(range.max - range.min);
            const normalized = denom <= 1e-12 ? 0.5 : (value - range.min) / denom;
            const x = margin.left + i * axisStep;
            const y = margin.top + (1.0 - normalized) * plotH;
            points.push({ x, y });
        }
        if (points.length < 2) {
            return;
        }

        plotted += 1;
        const selected = appState.selectedRunIds.has(run.run_id);
        ctx.globalAlpha = selected ? selectedAlpha : baseAlpha;
        ctx.strokeStyle = statusColor(getRunStatus(run));
        ctx.lineWidth = selected ? selectedLineWidth : baseLineWidth;
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i += 1) {
            ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.stroke();
    });

    ctx.globalAlpha = 1.0;
    ctx.fillStyle = textMuted;
    ctx.textAlign = 'left';
    ctx.font = '12px Plus Jakarta Sans, sans-serif';
    ctx.fillText(`${plotted} runs plotted`, margin.left, 12);
}

function renderParallelCoordinatesSection(
    runs,
    section,
    dimsMount,
    resetBtn,
    canvas,
    selectedDims,
    onDimsChange,
) {
    if (!section || !dimsMount || !canvas) {
        return selectedDims || [];
    }

    const availableDims = collectParallelDimensions(runs);
    if (availableDims.length < 2) {
        section.classList.add('module-muted');
        dimsMount.innerHTML = '<div class="module-empty">Need at least two numeric dimensions.</div>';
        if (resetBtn) {
            resetBtn.disabled = true;
        }
        drawParallelCoordinatesCanvas([], [], canvas);
        return [];
    }

    section.classList.remove('module-muted');
    if (resetBtn) {
        resetBtn.disabled = false;
    }

    let safeDims = (Array.isArray(selectedDims) ? selectedDims : []).filter((dim) => availableDims.includes(dim));
    if (safeDims.length < 2) {
        safeDims = defaultParallelDimensions(availableDims, pickPrimaryMetricKey(runs));
    }

    dimsMount.innerHTML = availableDims.map((dim) => {
        const active = safeDims.includes(dim) ? 'active' : '';
        return `<button type="button" class="axis-chip ${active}" data-dim="${escapeHtml(dim)}">${escapeHtml(dimensionLabel(dim))}</button>`;
    }).join('');

    dimsMount.querySelectorAll('.axis-chip').forEach((btn) => {
        btn.addEventListener('click', () => {
            const dim = btn.dataset.dim;
            if (!dim) {
                return;
            }
            let next = [...safeDims];
            if (next.includes(dim)) {
                if (next.length <= 2) {
                    return;
                }
                next = next.filter((x) => x !== dim);
            } else {
                next.push(dim);
                if (next.length > 8) {
                    next = next.slice(next.length - 8);
                }
            }
            onDimsChange(next);
        });
    });

    if (resetBtn) {
        resetBtn.onclick = () => {
            onDimsChange(defaultParallelDimensions(availableDims, pickPrimaryMetricKey(runs)));
        };
    }

    drawParallelCoordinatesCanvas(runs, safeDims, canvas);
    return safeDims;
}

function numericFromUnknown(value) {
    if (typeof value === 'number' && Number.isFinite(value)) {
        return value;
    }
    if (typeof value === 'string') {
        const raw = value.trim();
        if (!raw) {
            return null;
        }
        if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(raw)) {
            return null;
        }
        const n = Number(raw);
        return Number.isFinite(n) ? n : null;
    }
    return null;
}

function pearsonAbs(xs, ys) {
    const n = xs.length;
    if (n < 3 || ys.length !== n) {
        return null;
    }
    const meanX = xs.reduce((a, b) => a + b, 0) / n;
    const meanY = ys.reduce((a, b) => a + b, 0) / n;
    let cov = 0;
    let varX = 0;
    let varY = 0;
    for (let i = 0; i < n; i += 1) {
        const dx = xs[i] - meanX;
        const dy = ys[i] - meanY;
        cov += dx * dy;
        varX += dx * dx;
        varY += dy * dy;
    }
    const denom = Math.sqrt(varX * varY);
    if (denom <= 1e-12) {
        return null;
    }
    const corr = cov / denom;
    return Number.isFinite(corr) ? Math.abs(corr) : null;
}

function etaSquared(categories, ys) {
    const n = categories.length;
    if (n < 3 || ys.length !== n) {
        return null;
    }
    const groups = new Map();
    for (let i = 0; i < n; i += 1) {
        const cat = categories[i];
        if (!groups.has(cat)) {
            groups.set(cat, []);
        }
        groups.get(cat).push(ys[i]);
    }
    if (groups.size < 2) {
        return null;
    }

    const mean = ys.reduce((a, b) => a + b, 0) / n;
    let ssTotal = 0;
    ys.forEach((y) => {
        const d = y - mean;
        ssTotal += d * d;
    });
    if (ssTotal <= 1e-12) {
        return null;
    }

    let ssBetween = 0;
    groups.forEach((vals) => {
        const gm = vals.reduce((a, b) => a + b, 0) / vals.length;
        const d = gm - mean;
        ssBetween += vals.length * d * d;
    });

    const score = ssBetween / ssTotal;
    return Number.isFinite(score) ? Math.max(0, Math.min(1, score)) : null;
}

function computeFeatureImportance(runs, targetMetricKey) {
    const rows = runs
        .map((run) => ({ run, y: metricValueForRun(run, targetMetricKey) }))
        .filter((row) => row.y !== null);
    if (rows.length < 3) {
        return [];
    }

    const paramKeys = new Set();
    rows.forEach((row) => {
        Object.keys(row.run.params || {}).forEach((key) => paramKeys.add(key));
    });

    const ranked = [];
    paramKeys.forEach((key) => {
        const numericX = [];
        const numericY = [];
        const catX = [];
        const catY = [];
        let sampleCount = 0;

        rows.forEach((row) => {
            const raw = row.run.params ? row.run.params[key] : undefined;
            if (raw === undefined || raw === null || raw === '') {
                return;
            }
            sampleCount += 1;
            const num = numericFromUnknown(raw);
            if (num !== null) {
                numericX.push(num);
                numericY.push(row.y);
            }
            catX.push(String(raw));
            catY.push(row.y);
        });

        if (sampleCount < 3) {
            return;
        }

        const numScore = (numericX.length >= 3 && new Set(numericX).size > 1)
            ? pearsonAbs(numericX, numericY)
            : null;
        const catScore = (catX.length >= 3 && new Set(catX).size > 1)
            ? etaSquared(catX, catY)
            : null;

        if (numScore === null && catScore === null) {
            return;
        }

        let mode = 'numeric';
        let rawScore = numScore === null ? -1 : numScore;
        if (catScore !== null && (numScore === null || catScore > numScore)) {
            mode = 'categorical';
            rawScore = catScore;
        }

        const coverage = sampleCount / rows.length;
        const score = rawScore * Math.sqrt(coverage);
        ranked.push({
            key,
            mode,
            score,
            rawScore,
            coverage,
            samples: sampleCount,
        });
    });

    return ranked.sort((a, b) => b.score - a.score);
}

function renderFeatureImportanceSection(
    runs,
    section,
    targetSelect,
    canvas,
    listMount,
    targetMetric,
) {
    if (!section || !targetSelect || !canvas || !listMount) {
        return targetMetric || '';
    }

    const metricKeys = extractNumericMetricKeys(runs);
    if (!metricKeys.length) {
        section.classList.add('module-muted');
        targetSelect.innerHTML = '<option value="">No target metric</option>';
        targetSelect.disabled = true;
        listMount.innerHTML = '<li class="module-empty">Need numeric metrics for importance analysis.</li>';
        if (importanceChart) {
            importanceChart.destroy();
            importanceChart = null;
        }
        return '';
    }

    section.classList.remove('module-muted');
    targetSelect.disabled = false;

    const preferred = pickPrimaryMetricKey(runs);
    const safeTarget = metricKeys.includes(targetMetric) ? targetMetric : (preferred || metricKeys[0]);
    targetSelect.innerHTML = metricKeys.map((key) => `<option value="${escapeHtml(key)}">${escapeHtml(key)}</option>`).join('');
    targetSelect.value = safeTarget;

    const ranked = computeFeatureImportance(runs, safeTarget);
    if (!ranked.length) {
        listMount.innerHTML = '<li class="module-empty">Not enough parameter variability for this target.</li>';
        if (importanceChart) {
            importanceChart.destroy();
            importanceChart = null;
        }
        return safeTarget;
    }

    const top = ranked.slice(0, 12);
    const textColor = getComputedStyle(document.body).getPropertyValue('--text-secondary').trim() || '#94a3b8';
    const gridColor = getComputedStyle(document.body).getPropertyValue('--border-color').trim() || '#334155';
    const labels = top.map((entry) => truncateLabel(entry.key, 18));
    const values = top.map((entry) => Number(entry.score.toFixed(6)));
    const colors = top.map((_, idx) => PALETTE[idx % PALETTE.length]);

    if (importanceChart) {
        importanceChart.destroy();
        importanceChart = null;
    }

    importanceChart = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Importance',
                data: values,
                backgroundColor: colors.map((color) => `${color}bb`),
                borderColor: colors,
                borderWidth: 1.2,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    min: 0,
                    max: 1,
                    grid: { color: gridColor },
                    ticks: { color: textColor },
                },
                y: {
                    grid: { display: false },
                    ticks: { color: textColor },
                },
            },
            plugins: {
                legend: { display: false },
            },
        },
    });

    listMount.innerHTML = ranked.slice(0, 8).map((entry) => `
        <li class="importance-item">
            <div class="importance-main">
                <span>${escapeHtml(entry.key)}</span>
                <small>${entry.mode} · ${(entry.coverage * 100).toFixed(0)}% coverage</small>
            </div>
            <div class="importance-score">${escapeHtml(formatMetricValue(entry.score))}</div>
        </li>
    `).join('');

    return safeTarget;
}

function updateCompareButton(compareBtn) {
    const count = appState.selectedRunIds.size;
    compareBtn.textContent = `Compare Selected (${count})`;
    compareBtn.disabled = count < 2;
}

function buildMetricPreview(metrics) {
    const metricEntries = Object.entries(metrics || {}).slice(0, 4);
    if (!metricEntries.length) {
        return '<span class="metric-chip">No metrics</span>';
    }
    return metricEntries
        .map(([k, v]) => {
            return `<span class="metric-chip" title="${escapeHtml(k)}"><span>${escapeHtml(k)}</span><b>${escapeHtml(formatMetricValue(v))}</b></span>`;
        })
        .join('');
}

function appendRunRow(tbody, run, compareBtn) {
    const tr = document.createElement('tr');
    tr.className = 'run-row';
    if (appState.selectedRunIds.has(run.run_id)) {
        tr.classList.add('is-selected');
    }
    tr.onclick = () => loadRun(run.run_id);

    const pickTd = document.createElement('td');
    pickTd.className = 'checkbox-cell';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.className = 'run-picker';
    checkbox.checked = appState.selectedRunIds.has(run.run_id);
    checkbox.addEventListener('click', (e) => e.stopPropagation());
    checkbox.addEventListener('change', (e) => {
        if (e.target.checked) {
            appState.selectedRunIds.add(run.run_id);
            tr.classList.add('is-selected');
        } else {
            appState.selectedRunIds.delete(run.run_id);
            tr.classList.remove('is-selected');
        }
        updateCompareButton(compareBtn);
    });
    pickTd.appendChild(checkbox);

    const statusTd = document.createElement('td');
    statusTd.innerHTML = `<span class="status-badge status-${escapeHtml(run.status || 'RUNNING')}">${escapeHtml(run.status || 'RUNNING')}</span>`;

    const runTd = document.createElement('td');
    runTd.innerHTML = `
        <div class="run-name">${escapeHtml(run.name || 'Unnamed Run')}</div>
        <div class="run-id">${escapeHtml(run.run_id)}</div>
    `;

    const startTd = document.createElement('td');
    startTd.textContent = formatDate(run.start_time);

    const durationTd = document.createElement('td');
    durationTd.textContent = formatDuration(run.duration);

    const metricsTd = document.createElement('td');
    metricsTd.innerHTML = `<div class="metric-preview">${buildMetricPreview(run.metrics)}</div>`;

    const deleteTd = document.createElement('td');
    deleteTd.className = 'delete-cell';
    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.className = 'delete-run-btn';
    deleteBtn.title = 'Delete run';
    deleteBtn.innerHTML = '&#128465;';
    deleteBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        await deleteRun(run.run_id, tr);
    });
    deleteTd.appendChild(deleteBtn);

    tr.append(pickTd, statusTd, runTd, startTd, durationTd, metricsTd, deleteTd);
    tbody.appendChild(tr);
}

function updateRunTable(groups, tbody, compareBtn) {
    tbody.innerHTML = '';

    const totalRows = groups.reduce((acc, g) => acc + g.runs.length, 0);
    if (!totalRows) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="7" style="text-align:center;color:var(--text-secondary);">No runs found.</td>';
        tbody.appendChild(tr);
        return;
    }

    groups.forEach((group) => {
        if (group.label) {
            const groupRow = document.createElement('tr');
            groupRow.className = 'group-row';
            groupRow.innerHTML = `<td colspan="6">${escapeHtml(group.label)} <span>(${group.runs.length})</span></td>`;
            tbody.appendChild(groupRow);
        }

        group.runs.forEach((run) => appendRunRow(tbody, run, compareBtn));
    });
}

function fillKVList(container, values) {
    container.innerHTML = '';
    const entries = Object.entries(values || {});

    if (!entries.length) {
        container.innerHTML = '<div class="kv-key">-</div><div class="kv-val">No data</div>';
        return;
    }

    entries.forEach(([key, value]) => {
        container.innerHTML += `<div class="kv-key">${escapeHtml(key)}</div><div class="kv-val">${escapeHtml(String(value))}</div>`;
    });
}

function renderArchitectureNode(node) {
    const name = escapeHtml(node.name || 'module');
    const type = escapeHtml(node.type || 'Unknown');
    const numParams = Number.isFinite(node.num_params) ? Number(node.num_params) : 0;
    const trainable = Number.isFinite(node.trainable_params) ? Number(node.trainable_params) : 0;
    const children = Array.isArray(node.children) ? node.children : [];

    const line = `
        <div class="architecture-node-line">
            <span class="architecture-node-name">${name}</span>
            <span class="architecture-node-type">[${type}]</span>
            <span class="architecture-node-meta">params=${numParams.toLocaleString()} · trainable=${trainable.toLocaleString()}</span>
        </div>
    `;

    if (!children.length) {
        return `<li class="architecture-node">${line}</li>`;
    }

    const kids = children.map((child) => renderArchitectureNode(child)).join('');
    return `
        <li class="architecture-node">
            <details open>
                <summary>${line}</summary>
                <ul>${kids}</ul>
            </details>
        </li>
    `;
}

async function renderArchitectureView(run, artifacts, mount) {
    if (!mount) {
        return;
    }

    const archJson = artifacts.find((a) => /_architecture\.json$/i.test(a.path));
    if (!archJson) {
        mount.textContent = 'No architecture artifact found for this run.';
        return;
    }

    mount.textContent = 'Loading architecture...';
    try {
        const payload = await fetchArtifactJson(run.run_id, archJson.path);
        const summary = payload && payload.summary ? payload.summary : {};
        const tree = payload && payload.tree ? payload.tree : null;

        if (!tree) {
            mount.textContent = 'Architecture artifact is present, but has invalid format.';
            return;
        }

        const total = Number.isFinite(summary.total) ? Number(summary.total) : 0;
        const trainable = Number.isFinite(summary.trainable) ? Number(summary.trainable) : 0;
        const nonTrainable = Number.isFinite(summary.non_trainable)
            ? Number(summary.non_trainable)
            : Math.max(0, total - trainable);

        mount.innerHTML = `
            <div class="architecture-summary">
                <span class="tag-chip">total: ${total.toLocaleString()}</span>
                <span class="tag-chip">trainable: ${trainable.toLocaleString()}</span>
                <span class="tag-chip">non-trainable: ${nonTrainable.toLocaleString()}</span>
            </div>
            <ul class="architecture-tree">${renderArchitectureNode(tree)}</ul>
        `;
    } catch (err) {
        mount.textContent = `Failed to render architecture: ${err.message || err}`;
    }
}

function renderMainMetricChart(ctx, metricsData) {
    if (metricsChart) {
        metricsChart.destroy();
        metricsChart = null;
    }

    const allMetricEntries = Object.entries(metricsData || {});
    if (!allMetricEntries.length) {
        return;
    }

    const preferredOrder = ['train_loss', 'val_loss'];
    const preferredEntries = preferredOrder
        .map((preferred) => allMetricEntries.find(([key]) => String(key).toLowerCase() === preferred))
        .filter(Boolean);

    const metricEntries = preferredEntries.length ? preferredEntries : allMetricEntries;

    const textColor = getComputedStyle(document.body).getPropertyValue('--text-secondary').trim() || '#94a3b8';
    const gridColor = getComputedStyle(document.body).getPropertyValue('--border-color').trim() || '#334155';

    const datasets = metricEntries.map(([key, points], i) => ({
        label: key,
        data: (points || []).map((p) => ({ x: p.step, y: p.value })),
        borderColor: PALETTE[i % PALETTE.length],
        backgroundColor: 'transparent',
        tension: 0.25,
        borderWidth: 2,
        pointRadius: 1.8,
        pointHoverRadius: 4,
    }));

    metricsChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'nearest', intersect: false },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Step', color: textColor },
                    grid: { color: gridColor },
                    ticks: { color: textColor },
                },
                y: {
                    grid: { color: gridColor },
                    ticks: { color: textColor },
                },
            },
            plugins: {
                legend: { labels: { color: textColor } },
            },
        },
    });
}

function renderPinnedSection(mount) {
    if (!mount) {
        return;
    }

    destroyPinnedCharts();

    const relevantPins = appState.pinnedCharts.filter((pin) => pin.experimentName === appState.currentExperiment);
    if (!relevantPins.length) {
        mount.innerHTML = '';
        return;
    }

    mount.innerHTML = relevantPins.map((pin, idx) => {
        return `
            <article class="pinned-card">
                <div class="pinned-header">
                    <div>
                        <div class="pinned-title">${escapeHtml(pin.metricKey)}</div>
                        <div class="pinned-subtitle">${escapeHtml(pin.runName || pin.runId)}</div>
                    </div>
                    <button class="pin-remove" data-pin-key="${escapeHtml(pin.key)}">Remove</button>
                </div>
                <div class="chart-container tiny">
                    <canvas id="pin-chart-${idx}"></canvas>
                </div>
            </article>
        `;
    }).join('');

    relevantPins.forEach((pin, idx) => {
        const canvas = document.getElementById(`pin-chart-${idx}`);
        if (!canvas) {
            return;
        }

        const lineColor = hashColor(pin.key);
        const points = pin.points || [];
        const chart = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: {
                datasets: [{
                    label: pin.metricKey,
                    data: points.map((p) => ({ x: p.step, y: p.value })),
                    borderColor: lineColor,
                    backgroundColor: 'transparent',
                    tension: 0.22,
                    borderWidth: 2,
                    pointRadius: 0,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false, type: 'linear' },
                    y: { display: false },
                },
            },
        });
        pinnedChartInstances.push(chart);
    });

    mount.querySelectorAll('.pin-remove').forEach((btn) => {
        btn.addEventListener('click', () => {
            const key = btn.dataset.pinKey;
            appState.pinnedCharts = appState.pinnedCharts.filter((pin) => pin.key !== key);
            renderPinnedSection(mount);
        });
    });
}

function extractNumericMetricKeys(runs) {
    const keys = new Set();
    runs.forEach((run) => {
        Object.entries(run.metrics || {}).forEach(([k, v]) => {
            if (typeof v === 'number' && Number.isFinite(v)) {
                keys.add(k);
            }
        });
    });
    return [...keys].sort();
}

function renderSweepSection(runs, section, xSelect, ySelect, topList, canvas) {
    if (!section || !xSelect || !ySelect || !topList || !canvas) {
        return;
    }

    const metricKeys = extractNumericMetricKeys(runs);
    if (metricKeys.length < 2) {
        section.classList.add('module-muted');
        topList.innerHTML = '<li class="module-empty">Need at least two numeric metrics for sweep view.</li>';
        if (sweepChart) {
            sweepChart.destroy();
            sweepChart = null;
        }
        xSelect.innerHTML = '';
        ySelect.innerHTML = '';
        return;
    }

    section.classList.remove('module-muted');

    if (!metricKeys.includes(appState.sweepAxis.x)) {
        appState.sweepAxis.x = metricKeys[0];
    }
    if (!metricKeys.includes(appState.sweepAxis.y) || appState.sweepAxis.y === appState.sweepAxis.x) {
        appState.sweepAxis.y = metricKeys[1] || metricKeys[0];
    }

    xSelect.innerHTML = metricKeys.map((k) => `<option value="${escapeHtml(k)}" ${k === appState.sweepAxis.x ? 'selected' : ''}>X: ${escapeHtml(k)}</option>`).join('');
    ySelect.innerHTML = metricKeys.map((k) => `<option value="${escapeHtml(k)}" ${k === appState.sweepAxis.y ? 'selected' : ''}>Y: ${escapeHtml(k)}</option>`).join('');

    const xKey = appState.sweepAxis.x;
    const yKey = appState.sweepAxis.y;

    const points = runs
        .map((run) => {
            const x = run.metrics ? run.metrics[xKey] : null;
            const y = run.metrics ? run.metrics[yKey] : null;
            if (typeof x !== 'number' || typeof y !== 'number' || !Number.isFinite(x) || !Number.isFinite(y)) {
                return null;
            }
            return {
                x,
                y,
                runId: run.run_id,
                runName: run.name || run.run_id,
                status: run.status || 'UNKNOWN',
            };
        })
        .filter(Boolean);

    const grouped = new Map();
    points.forEach((point) => {
        if (!grouped.has(point.status)) {
            grouped.set(point.status, []);
        }
        grouped.get(point.status).push(point);
    });

    const datasets = [...grouped.entries()].map(([status, arr]) => ({
        label: status,
        data: arr,
        backgroundColor: statusColor(status),
        pointRadius: 4,
        pointHoverRadius: 6,
    }));

    const textColor = getComputedStyle(document.body).getPropertyValue('--text-secondary').trim() || '#94a3b8';
    const gridColor = getComputedStyle(document.body).getPropertyValue('--border-color').trim() || '#334155';

    if (sweepChart) {
        sweepChart.destroy();
        sweepChart = null;
    }

    sweepChart = new Chart(canvas.getContext('2d'), {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: textColor } },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const p = ctx.raw;
                            return `${p.runName} (${p.runId})`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    title: { display: true, text: xKey, color: textColor },
                    ticks: { color: textColor },
                    grid: { color: gridColor },
                },
                y: {
                    title: { display: true, text: yKey, color: textColor },
                    ticks: { color: textColor },
                    grid: { color: gridColor },
                },
            },
            onClick: (_evt, elements) => {
                if (!elements.length) {
                    return;
                }
                const hit = elements[0];
                const p = sweepChart.data.datasets[hit.datasetIndex].data[hit.index];
                if (p && p.runId) {
                    loadRun(p.runId);
                }
            },
        },
    });

    const topPoints = [...points].sort((a, b) => b.y - a.y).slice(0, 5);
    topList.innerHTML = topPoints.length
        ? topPoints.map((p) => `<li><button type="button" class="link-btn" data-run-id="${escapeHtml(p.runId)}">${escapeHtml(p.runName)}</button><span>${escapeHtml(formatMetricValue(p.y))}</span></li>`).join('')
        : '<li class="module-empty">No points with both metrics.</li>';

    topList.querySelectorAll('.link-btn').forEach((btn) => {
        btn.addEventListener('click', () => loadRun(btn.dataset.runId));
    });
}

async function renderCompareSection(section, compareData, onClose = null) {
    if (!section) {
        return;
    }

    if (!compareData || !compareData.runs || compareData.runs.length < 2) {
        section.classList.add('module-hidden');
        section.innerHTML = '';
        if (compareChart) {
            compareChart.destroy();
            compareChart = null;
        }
        return;
    }

    section.classList.remove('module-hidden');

    const metricKeys = compareData.metric_keys || [];
    const selectedMetric = metricKeys[0] || null;

    section.innerHTML = `
        <div class="module-header">
            <h3>Run Comparison</h3>
            <div class="compare-controls">
                <select id="compare-metric" class="sort-select">
                    ${metricKeys.map((k) => `<option value="${escapeHtml(k)}">Metric: ${escapeHtml(k)}</option>`).join('')}
                </select>
                <button id="compare-close" class="action-btn subtle">Close</button>
            </div>
        </div>
        <div class="compare-grid">
            <div class="chart-container compact">
                <canvas id="compare-canvas"></canvas>
            </div>
            <div>
                <table class="compare-table">
                    <thead>
                        <tr>
                            <th>Run</th>
                            <th>Status</th>
                            <th>Duration</th>
                            <th id="compare-value-head">Value</th>
                        </tr>
                    </thead>
                    <tbody id="compare-tbody"></tbody>
                </table>
            </div>
        </div>
    `;

    const metricSelect = section.querySelector('#compare-metric');
    const closeBtn = section.querySelector('#compare-close');
    const tbody = section.querySelector('#compare-tbody');
    const valueHead = section.querySelector('#compare-value-head');
    const canvas = section.querySelector('#compare-canvas');

    closeBtn.onclick = () => {
        section.classList.add('module-hidden');
        section.innerHTML = '';
        if (compareChart) {
            compareChart.destroy();
            compareChart = null;
        }
        if (typeof onClose === 'function') {
            onClose();
        }
    };

    async function updateCompare(metricKey) {
        valueHead.textContent = metricKey || 'Value';

        tbody.innerHTML = compareData.runs.map((run) => {
            const value = run.metrics && metricKey ? run.metrics[metricKey] : undefined;
            return `
                <tr>
                    <td>${escapeHtml(run.name || run.run_id)}</td>
                    <td><span class="status-badge status-${escapeHtml(run.status || 'RUNNING')}">${escapeHtml(run.status || 'RUNNING')}</span></td>
                    <td>${escapeHtml(formatDuration(run.duration))}</td>
                    <td>${escapeHtml(value !== undefined ? formatMetricValue(value) : '-')}</td>
                </tr>
            `;
        }).join('');

        if (!metricKey || !canvas) {
            return;
        }

        const histories = await Promise.all(compareData.runs.map(async (run) => {
            try {
                const hist = await fetchMetricHistory(run.run_id, metricKey);
                let points = (hist.metrics && hist.metrics[metricKey]) || [];
                if (!points.length && run.metrics && typeof run.metrics[metricKey] === 'number') {
                    points = [{ step: 0, value: run.metrics[metricKey] }];
                }
                return { run, points };
            } catch {
                return { run, points: [] };
            }
        }));

        const textColor = getComputedStyle(document.body).getPropertyValue('--text-secondary').trim() || '#94a3b8';
        const gridColor = getComputedStyle(document.body).getPropertyValue('--border-color').trim() || '#334155';

        const datasets = histories.map((entry, i) => ({
            label: entry.run.name || entry.run.run_id,
            data: entry.points.map((p) => ({ x: p.step, y: p.value })),
            borderColor: hashColor(`${entry.run.run_id}:${metricKey}:${i}`),
            backgroundColor: 'transparent',
            tension: 0.25,
            borderWidth: 2,
            pointRadius: 1.5,
        }));

        if (compareChart) {
            compareChart.destroy();
            compareChart = null;
        }

        compareChart = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'nearest', intersect: false },
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Step', color: textColor },
                        ticks: { color: textColor },
                        grid: { color: gridColor },
                    },
                    y: {
                        ticks: { color: textColor },
                        grid: { color: gridColor },
                    },
                },
                plugins: {
                    legend: { labels: { color: textColor } },
                },
            },
        });
    }

    metricSelect.onchange = () => updateCompare(metricSelect.value);
    await updateCompare(selectedMetric);
}

async function pinMetric(run, metricKey, preloadedMetrics = null) {
    const experimentName = appState.currentExperiment || '__global';
    const key = pinKey(experimentName, run.run_id, metricKey);
    if (appState.pinnedCharts.some((pin) => pin.key === key)) {
        return;
    }

    let points = [];
    if (preloadedMetrics && preloadedMetrics[metricKey]) {
        points = preloadedMetrics[metricKey];
    }

    if (!points.length) {
        try {
            const history = await fetchMetricHistory(run.run_id, metricKey);
            points = (history.metrics && history.metrics[metricKey]) || [];
        } catch {
            points = [];
        }
    }

    if (!points.length && run.metrics && typeof run.metrics[metricKey] === 'number') {
        points = [{ step: 0, value: run.metrics[metricKey] }];
    }

    if (!points.length) {
        return;
    }

    appState.pinnedCharts.push({
        key,
        experimentName,
        runId: run.run_id,
        runName: run.name || run.run_id,
        metricKey,
        points,
    });

    const pinMount = document.getElementById('pinned-section');
    if (pinMount) {
        renderPinnedSection(pinMount);
    }
}

// ── Tab strip helper (scroll-based, safe for Chart.js) ──────────────────
async function deleteRun(runId, rowEl) {
    if (!window.confirm(`Delete run "${runId}"?\nThis cannot be undone.`)) return;
    try {
        await fetchJSON(`${API_BASE}/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' });
        rowEl.style.transition = 'opacity 200ms, transform 200ms';
        rowEl.style.opacity = '0';
        rowEl.style.transform = 'translateX(12px)';
        setTimeout(() => {
            rowEl.remove();
            if (appState.currentExperiment) loadExperiment(appState.currentExperiment);
        }, 220);
    } catch (err) {
        alert(`Failed to delete run: ${err.message}`);
    }
}

function initViewTabs(tabStrip, scrollContainer) {
    if (!tabStrip || !scrollContainer) return;
    const tabs = [...tabStrip.querySelectorAll('.view-tab-btn')];
    if (!tabs.length) return;

    const activate = (tab) => {
        tabs.forEach((t) => t.classList.remove('active'));
        tab.classList.add('active');
        const target = scrollContainer.querySelector(tab.dataset.scrollTo);
        if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };

    tabs.forEach((tab) => { tab.onclick = () => activate(tab); });
    tabs[0].classList.add('active');

    // Auto-highlight tab based on scroll position
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((e) => {
            if (e.isIntersecting) {
                const match = tabs.find((t) => scrollContainer.querySelector(t.dataset.scrollTo) === e.target);
                if (match) { tabs.forEach((t) => t.classList.remove('active')); match.classList.add('active'); }
            }
        });
    }, { root: scrollContainer, threshold: 0.15 });

    tabs.forEach((tab) => {
        const el = scrollContainer.querySelector(tab.dataset.scrollTo);
        if (el) observer.observe(el);
    });
}

function renderRunList(runs, expName) {
    setRunSidebarVisible(true);

    const tpl = document.getElementById('tpl-run-list');
    const view = tpl.content.cloneNode(true);

    const title = view.querySelector('#exp-title');
    const subtitle = view.querySelector('#exp-subtitle');
    const searchInput = view.querySelector('#params-search');
    const sortSelect = view.querySelector('#sort-select');
    const groupSelect = view.querySelector('#group-select');
    const compareBtn = view.querySelector('#compare-btn');
    const clearPinsBtn = view.querySelector('#clear-pins-btn');
    const tagFilterInput = view.querySelector('#tag-filter');
    const metricFilterKeySelect = view.querySelector('#metric-filter-key');
    const metricFilterMinInput = view.querySelector('#metric-filter-min');
    const metricFilterMaxInput = view.querySelector('#metric-filter-max');
    const onlySelectedToggle = view.querySelector('#only-selected-runs');
    const selectVisibleBtn = view.querySelector('#select-visible-btn');
    const clearSelectedBtn = view.querySelector('#clear-selected-btn');
    const statusFiltersMount = view.querySelector('#status-filters');
    const summaryMount = view.querySelector('#run-summary');
    const insightsMount = view.querySelector('#insights-strip');
    const leaderboardSection = view.querySelector('#leaderboard-section');
    const leaderboardMetricSelect = view.querySelector('#leaderboard-metric');
    const leaderboardObjectiveSelect = view.querySelector('#leaderboard-objective');
    const leaderboardList = view.querySelector('#leaderboard-list');
    const parallelSection = view.querySelector('#parallel-section');
    const parallelDimsMount = view.querySelector('#parallel-dimensions');
    const parallelResetBtn = view.querySelector('#parallel-reset');
    const parallelCanvas = view.querySelector('#parallel-canvas');
    const importanceSection = view.querySelector('#importance-section');
    const importanceTargetSelect = view.querySelector('#importance-target');
    const importanceCanvas = view.querySelector('#importance-canvas');
    const importanceList = view.querySelector('#importance-list');
    const pinnedMount = view.querySelector('#pinned-section');
    const sweepSection = view.querySelector('#sweep-section');
    const sweepX = view.querySelector('#sweep-x');
    const sweepY = view.querySelector('#sweep-y');
    const sweepTop = view.querySelector('#sweep-top-runs');
    const sweepCanvas = view.querySelector('#sweep-canvas');
    const compareSection = view.querySelector('#compare-section');
    const tbody = view.querySelector('#runs-tbody');

    title.textContent = expName;

    const prefs = loadViewPrefs(expName);
    const urlState = loadViewStateFromUrl(expName);
    let activeRuns = [...runs];
    let runQuery = typeof urlState.runQuery === 'string' && urlState.runQuery !== ''
        ? urlState.runQuery
        : (typeof prefs.runQuery === 'string' ? prefs.runQuery : '');
    let sortMode = typeof urlState.sortMode === 'string' && urlState.sortMode !== ''
        ? urlState.sortMode
        : (typeof prefs.sortMode === 'string' ? prefs.sortMode : '-start_time');
    let groupMode = typeof urlState.groupMode === 'string' && urlState.groupMode !== ''
        ? urlState.groupMode
        : (typeof prefs.groupMode === 'string' ? prefs.groupMode : 'none');
    let activeStatus = typeof urlState.activeStatus === 'string' && urlState.activeStatus !== ''
        ? urlState.activeStatus
        : (typeof prefs.activeStatus === 'string' ? prefs.activeStatus : 'ALL');
    let tagFilter = typeof urlState.tagFilter === 'string' && urlState.tagFilter !== ''
        ? urlState.tagFilter
        : (typeof prefs.tagFilter === 'string' ? prefs.tagFilter : '');
    let metricFilterKey = typeof urlState.metricFilterKey === 'string' && urlState.metricFilterKey !== ''
        ? urlState.metricFilterKey
        : (typeof prefs.metricFilterKey === 'string' ? prefs.metricFilterKey : '');
    let metricMin = (
        urlState.metricMin !== '' && Number.isFinite(Number(urlState.metricMin))
    ) ? Number(urlState.metricMin) : (
        prefs.metricMin !== ''
        && prefs.metricMin !== null
        && prefs.metricMin !== undefined
        && Number.isFinite(Number(prefs.metricMin))
    ) ? Number(prefs.metricMin) : '';
    let metricMax = (
        urlState.metricMax !== '' && Number.isFinite(Number(urlState.metricMax))
    ) ? Number(urlState.metricMax) : (
        prefs.metricMax !== ''
        && prefs.metricMax !== null
        && prefs.metricMax !== undefined
        && Number.isFinite(Number(prefs.metricMax))
    ) ? Number(prefs.metricMax) : '';
    let onlySelected = urlState.onlySelected || Boolean(prefs.onlySelected);
    let leaderboardMetric = typeof urlState.leaderboardMetric === 'string' && urlState.leaderboardMetric !== ''
        ? urlState.leaderboardMetric
        : (typeof prefs.leaderboardMetric === 'string' ? prefs.leaderboardMetric : '');
    let leaderboardObjective = ['auto', 'min', 'max'].includes(urlState.leaderboardObjective)
        ? urlState.leaderboardObjective
        : ['auto', 'min', 'max'].includes(prefs.leaderboardObjective)
        ? prefs.leaderboardObjective
        : 'auto';
    let parallelDims = Array.isArray(urlState.parallelDims) && urlState.parallelDims.length
        ? urlState.parallelDims.filter((dim) => typeof dim === 'string')
        : Array.isArray(prefs.parallelDims)
        ? prefs.parallelDims.filter((dim) => typeof dim === 'string')
        : [];
    let importanceTarget = typeof urlState.importanceTarget === 'string' && urlState.importanceTarget !== ''
        ? urlState.importanceTarget
        : typeof prefs.importanceTarget === 'string'
        ? prefs.importanceTarget
        : '';
    let currentCompareData = null;
    let visibleRuns = [...runs];

    if (Array.isArray(urlState.selectedRunIds) && urlState.selectedRunIds.length) {
        appState.selectedRunIds = new Set(urlState.selectedRunIds);
    }

    if (searchInput) {
        searchInput.value = runQuery;
    }
    if (sortSelect) {
        sortSelect.value = sortMode;
    }
    if (groupSelect) {
        groupSelect.value = groupMode;
    }

    if (tagFilterInput) {
        tagFilterInput.value = tagFilter;
    }
    if (metricFilterMinInput) {
        metricFilterMinInput.value = metricMin === '' ? '' : String(metricMin);
    }
    if (metricFilterMaxInput) {
        metricFilterMaxInput.value = metricMax === '' ? '' : String(metricMax);
    }
    if (onlySelectedToggle) {
        onlySelectedToggle.checked = onlySelected;
    }
    if (leaderboardObjectiveSelect) {
        leaderboardObjectiveSelect.value = leaderboardObjective;
    }

    const persistPrefs = () => {
        saveViewPrefs(expName, {
            activeStatus,
            tagFilter,
            metricFilterKey,
            metricMin,
            metricMax,
            onlySelected,
            runQuery,
            sortMode,
            groupMode,
            leaderboardMetric,
            leaderboardObjective,
            parallelDims,
            importanceTarget,
        });
    };

    const repaint = async () => {
        const sorted = sortRuns(activeRuns, sortMode);
        const metricKeys = extractNumericMetricKeys(sorted);
        metricFilterKey = setMetricSelectOptions(
            metricFilterKeySelect,
            metricKeys,
            metricFilterKey,
            'Metric filter: Any',
        );

        activeStatus = renderStatusFilters(sorted, statusFiltersMount, activeStatus, (nextStatus) => {
            activeStatus = nextStatus;
            persistPrefs();
            repaint();
        });
        const statusFiltered = activeStatus === 'ALL'
            ? sorted
            : sorted.filter((run) => getRunStatus(run) === activeStatus);
        const filtered = applyAdvancedFilters(statusFiltered, {
            tagFilter,
            metricFilterKey,
            metricMin,
            metricMax,
            onlySelected,
            selectedRunIds: appState.selectedRunIds,
        });
        visibleRuns = filtered;

        const grouped = groupRuns(filtered, groupMode);
        updateRunTable(grouped, tbody, compareBtn);
        renderRunSidebar(filtered, compareBtn);
        renderRunSummary(filtered, summaryMount);
        renderInsightsStrip(filtered, insightsMount);
        renderPinnedSection(pinnedMount);

        parallelDims = renderParallelCoordinatesSection(
            filtered,
            parallelSection,
            parallelDimsMount,
            parallelResetBtn,
            parallelCanvas,
            parallelDims,
            (nextDims) => {
                parallelDims = nextDims;
                persistPrefs();
                repaint();
            },
        );

        importanceTarget = renderFeatureImportanceSection(
            filtered,
            importanceSection,
            importanceTargetSelect,
            importanceCanvas,
            importanceList,
            importanceTarget,
        );

        renderSweepSection(filtered, sweepSection, sweepX, sweepY, sweepTop, sweepCanvas);

        const lbState = renderLeaderboardSection(
            filtered,
            leaderboardSection,
            leaderboardMetricSelect,
            leaderboardObjectiveSelect,
            leaderboardList,
            leaderboardMetric,
            leaderboardObjective,
        );
        leaderboardMetric = lbState.metricKey;
        leaderboardObjective = lbState.objectiveMode;

        const filterTokens = [];
        if (activeStatus !== 'ALL') {
            filterTokens.push(`status=${activeStatus}`);
        }
        if (tagFilter.trim()) {
            filterTokens.push(`tag=${tagFilter.trim()}`);
        }
        if (metricFilterKey && (metricMin !== '' || metricMax !== '')) {
            filterTokens.push(`metric=${metricFilterKey}`);
        }
        if (onlySelected) {
            filterTokens.push('only selected');
        }

        subtitle.textContent = filterTokens.length
            ? `${filtered.length}/${sorted.length} runs shown · ${filterTokens.join(' · ')}`
            : `${filtered.length} runs shown`;
        updateCompareButton(compareBtn);

        if (currentCompareData) {
            await renderCompareSection(compareSection, currentCompareData, () => {
                currentCompareData = null;
            });
        }

        appState.currentViewSnapshot = {
            runQuery,
            sortMode,
            groupMode,
            activeStatus,
            tagFilter,
            metricFilterKey,
            metricMin,
            metricMax,
            onlySelected,
            leaderboardMetric,
            leaderboardObjective,
            parallelDims,
            importanceTarget,
            selectedRunIds: [...appState.selectedRunIds].slice(0, 80),
        };

        writeViewStateToUrl(expName, appState.currentViewSnapshot);
        persistPrefs();
    };

    appState.runListRepaint = repaint;
    appState.runDetailRepaint = null;

    searchInput.oninput = debounce(async (e) => {
        const q = e.target.value.trim();
        runQuery = q;
        try {
            activeRuns = await fetchRuns(expName, q || null);
            await repaint();
        } catch (err) {
            console.error('Search failed:', err);
        }
    }, 240);

    sortSelect.onchange = () => {
        sortMode = sortSelect.value;
        repaint();
    };

    groupSelect.onchange = () => {
        groupMode = groupSelect.value;
        repaint();
    };

    if (tagFilterInput) {
        tagFilterInput.oninput = debounce(() => {
            tagFilter = tagFilterInput.value.trim();
            repaint();
        }, 180);
    }

    if (metricFilterKeySelect) {
        metricFilterKeySelect.onchange = () => {
            metricFilterKey = metricFilterKeySelect.value;
            repaint();
        };
    }

    if (metricFilterMinInput) {
        metricFilterMinInput.oninput = debounce(() => {
            const raw = metricFilterMinInput.value.trim();
            metricMin = raw === '' ? '' : Number(raw);
            if (metricMin !== '' && !Number.isFinite(metricMin)) {
                metricMin = '';
            }
            repaint();
        }, 180);
    }

    if (metricFilterMaxInput) {
        metricFilterMaxInput.oninput = debounce(() => {
            const raw = metricFilterMaxInput.value.trim();
            metricMax = raw === '' ? '' : Number(raw);
            if (metricMax !== '' && !Number.isFinite(metricMax)) {
                metricMax = '';
            }
            repaint();
        }, 180);
    }

    if (onlySelectedToggle) {
        onlySelectedToggle.onchange = () => {
            onlySelected = Boolean(onlySelectedToggle.checked);
            repaint();
        };
    }

    if (selectVisibleBtn) {
        selectVisibleBtn.onclick = () => {
            visibleRuns.forEach((run) => appState.selectedRunIds.add(run.run_id));
            repaint();
        };
    }

    if (clearSelectedBtn) {
        clearSelectedBtn.onclick = () => {
            appState.selectedRunIds.clear();
            repaint();
        };
    }

    if (leaderboardMetricSelect) {
        leaderboardMetricSelect.onchange = () => {
            leaderboardMetric = leaderboardMetricSelect.value;
            repaint();
        };
    }

    if (leaderboardObjectiveSelect) {
        leaderboardObjectiveSelect.onchange = () => {
            leaderboardObjective = leaderboardObjectiveSelect.value;
            repaint();
        };
    }

    if (importanceTargetSelect) {
        importanceTargetSelect.onchange = () => {
            importanceTarget = importanceTargetSelect.value;
            repaint();
        };
    }

    sweepX.onchange = () => {
        appState.sweepAxis.x = sweepX.value;
        renderSweepSection(visibleRuns, sweepSection, sweepX, sweepY, sweepTop, sweepCanvas);
    };

    sweepY.onchange = () => {
        appState.sweepAxis.y = sweepY.value;
        renderSweepSection(visibleRuns, sweepSection, sweepX, sweepY, sweepTop, sweepCanvas);
    };

    compareBtn.onclick = async () => {
        const runIds = [...appState.selectedRunIds];
        if (runIds.length < 2) {
            return;
        }
        try {
            currentCompareData = await fetchRunCompare(runIds);
            await renderCompareSection(compareSection, currentCompareData, () => {
                currentCompareData = null;
            });
        } catch (err) {
            console.error('Compare failed:', err);
            compareSection.classList.remove('module-hidden');
            compareSection.innerHTML = `<div class="module-empty">Compare failed: ${escapeHtml(err.message)}</div>`;
        }
    };

    clearPinsBtn.onclick = () => {
        appState.pinnedCharts = appState.pinnedCharts.filter((pin) => pin.experimentName !== appState.currentExperiment);
        renderPinnedSection(pinnedMount);
    };

    if (els.runSidebarClear) {
        els.runSidebarClear.onclick = () => {
            appState.selectedRunIds.clear();
            repaint();
        };
    }

    els.viewContainer.innerHTML = '';
    els.viewContainer.appendChild(view);

    (async () => {
        if (runQuery) {
            try {
                activeRuns = await fetchRuns(expName, runQuery);
            } catch (err) {
                console.error('Initial query load failed:', err);
            }
        }
        await repaint();
    })();
    initViewTabs(els.viewContainer.querySelector('#view-tabs-list'), els.viewContainer);
}

function openDrawer() {
    els.drawerOverlay.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function closeDrawer() {
    els.drawerOverlay.classList.add('hidden');
    document.body.style.overflow = '';

    if (metricsChart) {
        metricsChart.destroy();
        metricsChart = null;
    }

    if (appState.currentExperiment) {
        window.location.hash = `#exp/${encodeURIComponent(appState.currentExperiment)}`;
    } else {
        window.history.pushState('', document.title, window.location.pathname + window.location.search);
    }
}

function renderRunDetail(run, metricsHist, artifacts) {
    const tpl = document.getElementById('tpl-run-detail');
    const view = tpl.content.cloneNode(true);

    view.querySelector('#run-title').textContent = run.name || run.run_id;

    const metricsMap = metricsHist.metrics || {};
    const metricKeys = Object.keys(metricsMap).length ? Object.keys(metricsMap) : Object.keys(run.metrics || {});

    const badges = view.querySelector('#run-badges');
    badges.innerHTML = `
        <span class="status-badge status-${escapeHtml(run.status || 'RUNNING')}">${escapeHtml(run.status || 'RUNNING')}</span>
        <span class="tag-chip">run_id: ${escapeHtml(run.run_id)}</span>
    `;

    if (metricKeys.length) {
        const pinControl = document.createElement('div');
        pinControl.className = 'pin-controls';
        pinControl.innerHTML = `
            <select class="inline-select" id="pin-metric-select">
                ${metricKeys.map((k) => `<option value="${escapeHtml(k)}">${escapeHtml(k)}</option>`).join('')}
            </select>
            <button type="button" class="action-btn" id="pin-metric-btn">Pin Metric</button>
        `;
        badges.appendChild(pinControl);

        const pinBtn = pinControl.querySelector('#pin-metric-btn');
        const pinSelect = pinControl.querySelector('#pin-metric-select');
        pinBtn.onclick = async () => {
            await pinMetric(run, pinSelect.value, metricsMap);
        };
    }

    fillKVList(view.querySelector('#params-list'), run.params || {});

    fillKVList(view.querySelector('#info-list'), {
        experiment_id: run.experiment_id,
        start_time: run.start_time,
        end_time: run.end_time || '-',
        duration: formatDuration(run.duration),
        status: run.status,
    });

    const tagsList = view.querySelector('#tags-list');
    tagsList.innerHTML = '';
    const tagEntries = Object.entries(run.tags || {});
    if (!tagEntries.length) {
        tagsList.innerHTML = '<span class="tag-chip">No tags</span>';
    } else {
        tagEntries.forEach(([k, v]) => {
            tagsList.innerHTML += `<span class="tag-chip">${escapeHtml(k)}: ${escapeHtml(v)}</span>`;
        });
    }

    const artList = view.querySelector('#artifact-list');
    const architectureView = view.querySelector('#architecture-view');
    artList.innerHTML = '';
    if (!artifacts.length) {
        artList.innerHTML = '<li><span>No artifacts found.</span></li>';
    } else {
        artifacts.forEach((art) => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span>${escapeHtml(art.path)} <span style="color:var(--text-muted)">(${escapeHtml(art.type)})</span></span>
                <a href="${API_BASE}/runs/${run.run_id}/artifacts/${encodeURIComponent(art.path).replace(/%2F/g, '/')}" class="artifact-link" target="_blank" rel="noopener noreferrer">Download</a>
            `;
            artList.appendChild(li);
        });
    }

    renderArchitectureView(run, artifacts, architectureView);

    els.drawerContent.innerHTML = '';
    els.drawerContent.appendChild(view);
    openDrawer();

    // Fill run-hero with latest metric values
    const heroEl = els.drawerContent.querySelector('#run-hero');
    if (heroEl) {
        const heroMetrics = Object.entries(run.metrics || {}).slice(0, 5);
        heroEl.innerHTML = heroMetrics.length
            ? heroMetrics.map(([k, v]) => `
                <div class="hero-metric-item">
                    <span class="hero-metric-label">${escapeHtml(k)}</span>
                    <span class="hero-metric-value">${typeof v === 'number' ? v.toFixed(4) : escapeHtml(String(v))}</span>
                </div>`).join('')
            : '<span style="color:var(--text-muted);font-size:12px;padding:8px;">No metrics logged yet</span>';
    }

    // Scroll-based tabs in the drawer
    const drawerEl = els.drawerContent.closest('.drawer') || els.drawerContent.parentElement;
    initViewTabs(els.drawerContent.querySelector('#view-tabs-detail'), drawerEl);

    const canvas = els.drawerContent.querySelector('#metrics-chart');
    appState.runDetailRepaint = () => {
        const liveCanvas = els.drawerContent.querySelector('#metrics-chart');
        if (liveCanvas && typeof Chart !== 'undefined') {
            renderMainMetricChart(liveCanvas.getContext('2d'), metricsMap);
        }
    };
    if (canvas && typeof Chart !== 'undefined') {
        renderMainMetricChart(canvas.getContext('2d'), metricsMap);
    }
}

async function loadRun(runId) {
    window.location.hash = `#run/${encodeURIComponent(runId)}`;
    els.breadcrumbs.innerHTML = '<span class="crumb root">Dashboard</span> / <span class="crumb active">Run</span>';

    try {
        const [run, metrics, artifacts] = await Promise.all([
            fetchRunDetail(runId),
            fetchMetricHistory(runId),
            fetchArtifacts(runId),
        ]);
        renderRunDetail(run, metrics, artifacts);
    } catch (err) {
        console.error('Failed to load run:', err);
        els.viewContainer.innerHTML = `
            <div class="empty-state">
                <h2>Could not load run</h2>
                <p>${escapeHtml(err.message)}</p>
            </div>
        `;
    }
}

async function loadExperiment(name) {
    appState.currentExperiment = name;
    appState.selectedRunIds = new Set();
    window.location.hash = `#exp/${encodeURIComponent(name)}`;
    renderExperiments(appState.experiments);

    els.breadcrumbs.innerHTML = `<span class="crumb root">Dashboard</span> / <span class="crumb active">${escapeHtml(name)}</span>`;

    try {
        const runs = await fetchRuns(name);
        appState.currentRuns = runs;
        setTopbarMeta(`${runs.length} runs · ${name}`);
        renderRunList(runs, name);
    } catch (err) {
        console.error('Failed to load experiment runs:', err);
        setRunSidebarVisible(false);
        if (els.runSidebarList) {
            els.runSidebarList.innerHTML = '';
        }
        els.viewContainer.innerHTML = `
            <div class="empty-state">
                <h2>Could not load runs</h2>
                <p>${escapeHtml(err.message)}</p>
            </div>
        `;
    }
}

async function routeFromHash() {
    const hash = window.location.hash;
    if (hash.startsWith('#run/')) {
        setTopbarMeta('Run Detail');
        await loadRun(decodeURIComponent(hash.slice(5)));
        return;
    }

    if (hash.startsWith('#exp/')) {
        await loadExperiment(decodeURIComponent(hash.slice(5)));
        return;
    }

    if (appState.experiments.length) {
        await loadExperiment(appState.experiments[0].name);
        return;
    }

    setRunSidebarVisible(false);
    if (els.runSidebarList) {
        els.runSidebarList.innerHTML = '';
    }

    setTopbarMeta('No experiments available');

    els.viewContainer.innerHTML = `
        <div class="empty-state">
            <h2>No experiments yet</h2>
            <p>Create your first run to see activity here.</p>
        </div>
    `;
    appState.runListRepaint = null;
    appState.runDetailRepaint = null;
}

async function loadApp() {
    await checkHealth();

    try {
        const experiments = await fetchExperiments();
        appState.experiments = experiments;
        renderExperiments(experiments);
        setTopbarMeta(`${experiments.length} experiment${experiments.length !== 1 ? 's' : ''}`);
        await routeFromHash();
    } catch (err) {
        console.error('Failed to initialize app:', err);
        setTopbarMeta('Dashboard unavailable');
        els.viewContainer.innerHTML = `
            <div class="empty-state">
                <h2>Dashboard unavailable</h2>
                <p>${escapeHtml(err.message)}</p>
            </div>
        `;
    }
}

els.drawerOverlay.onclick = (e) => {
    if (e.target === els.drawerOverlay) {
        closeDrawer();
    }
};

els.drawerClose.onclick = closeDrawer;

document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        openCommandPalette('');
        return;
    }

    if (e.key === 'Escape') {
        closeCommandPalette();
        closeDrawer();
    }
});

if (els.commandPaletteInput) {
    els.commandPaletteInput.addEventListener('input', () => {
        appState.paletteActions = buildCommandPaletteActions(els.commandPaletteInput.value);
        appState.paletteIndex = 0;
        renderCommandPaletteActions();
    });

    els.commandPaletteInput.addEventListener('keydown', async (e) => {
        const max = appState.paletteActions.length - 1;
        if (e.key === 'ArrowDown' && max >= 0) {
            e.preventDefault();
            appState.paletteIndex = Math.min(max, appState.paletteIndex + 1);
            renderCommandPaletteActions();
            return;
        }
        if (e.key === 'ArrowUp' && max >= 0) {
            e.preventDefault();
            appState.paletteIndex = Math.max(0, appState.paletteIndex - 1);
            renderCommandPaletteActions();
            return;
        }
        if (e.key === 'Enter' && max >= 0) {
            e.preventDefault();
            const action = appState.paletteActions[appState.paletteIndex];
            if (action) {
                await action.run();
            }
        }
    });
}

if (els.commandPaletteOverlay) {
    els.commandPaletteOverlay.addEventListener('click', (event) => {
        if (event.target === els.commandPaletteOverlay) {
            closeCommandPalette();
        }
    });
}

if (els.commandPaletteClose) {
    els.commandPaletteClose.onclick = closeCommandPalette;
}

if (els.commandPaletteBtn) {
    els.commandPaletteBtn.onclick = () => openCommandPalette('');
}

if (els.savedViewsBtn) {
    els.savedViewsBtn.onclick = () => openCommandPalette('view');
}

els.refreshBtn.onclick = async () => {
    const hash = window.location.hash;
    if (hash.startsWith('#run/')) {
        await loadRun(decodeURIComponent(hash.slice(5)));
    } else if (appState.currentExperiment) {
        await loadExperiment(appState.currentExperiment);
    } else {
        await loadApp();
    }
};

if (els.themeToggle) {
    els.themeToggle.onclick = () => {
        const next = document.body.dataset.theme === 'dark' ? 'light' : 'dark';
        setTheme(next);
    };
}

if (els.experimentSearch) {
    els.experimentSearch.addEventListener('input', (event) => {
        appState.experimentFilter = event.target.value || '';
        renderExperiments(appState.experiments);
    });
}

window.addEventListener('hashchange', () => {
    routeFromHash();
});

initTheme();
loadApp();
