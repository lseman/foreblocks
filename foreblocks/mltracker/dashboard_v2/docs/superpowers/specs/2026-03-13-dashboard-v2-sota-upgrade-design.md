# dashboard_v2 SOTA Upgrade — Design Spec

**Date:** 2026-03-13
**Status:** Approved
**Scope:** Incremental upgrade of the existing `dashboard_v2` ML experiment tracker UI

---

## Context

`dashboard_v2` is a React 18 + TypeScript + Vite ML experiment tracking dashboard with a custom CSS design system. It ships five analytics modules (leaderboard, sweep scatter, parallel coordinates, feature importance, compare), dark/light theming, a command palette, saved views, and URL-synced state. All logic and components live in a single `App.tsx` (~1500 lines) with hand-rolled SVG charts and a monolithic CSS file.

The goal is an incremental upgrade toward a SOTA production-quality dashboard — better charting, proper component architecture, real-time updates, and a modern design system — without breaking the working dashboard at any intermediate step.

---

## Decisions

| Dimension | Decision | Rationale |
|---|---|---|
| Approach | Incremental upgrade (not rewrite) | Always-working dashboard at every step |
| Charting | ECharts via `echarts-for-react` | Native parallel coords, scatter brushing, zoom/pan; closest to W&B/TensorBoard quality |
| UI foundation | Tailwind CSS v3 + shadcn/ui | Replaces hand-rolled primitives (palette, drawer, chips, buttons) with owned, accessible components |
| State management | Zustand | Lightweight, no boilerplate, replaces prop drilling from monolithic App.tsx |
| Motion | Framer Motion | Micro-interactions, card enters, animated KPI counters |
| Real-time | SSE (Server-Sent Events) | Simpler than WebSocket for unidirectional metric streaming; matches the read-heavy nature of the tracker |

---

## Phased Plan

### Phase 1 — Foundation (Tailwind + shadcn/ui)

**Goal:** Replace hand-rolled primitives with shadcn components. No feature changes.

**New dependencies:**
- `tailwindcss@3` + `postcss` + `autoprefixer`
- `shadcn/ui` (CLI: `npx shadcn@latest init`)
- `@radix-ui/*` (installed by shadcn automatically)
- `clsx` + `tailwind-merge` (for `cn()` utility)
- `lucide-react` (icons)

**shadcn components to install:**
- `button` -> replaces `.action-btn`, `.icon-btn`
- `input` -> replaces `.search-input`
- `select` -> replaces `.sort-select`
- `badge` -> replaces `.status-badge`, `.metric-chip`, `.tag-chip`
- `dialog` -> replaces the command palette overlay
- `command` -> replaces the palette list/search (shadcn Command is built for this)
- `sheet` -> replaces the run detail drawer (Sheet = slide-over panel)
- `tooltip` -> replaces custom hover tooltips on charts/buttons
- `separator` -> replaces border-based dividers

**CSS setup — Tailwind + existing styles coexistence:**

Tailwind's Preflight (`@tailwind base`) resets `margin`, `padding`, `box-sizing`, and element defaults globally, which would break the existing `body { overflow: hidden }`, `html/body/#root { height: 100% }`, and the `background: radial-gradient(...)` on body. To prevent this:

1. Disable Preflight in `tailwind.config.js`. The project has `"type": "module"` in `package.json`, so use ES module syntax:
   ```js
   // tailwind.config.js
   export default {
     corePlugins: { preflight: false },
     content: ['./index.html', './src/**/*.{ts,tsx}'],
     theme: { extend: {} },
     plugins: [],
   }
   ```

2. Create `postcss.config.js` at project root:
   ```js
   export default {
     plugins: {
       tailwindcss: {},
       autoprefixer: {},
     },
   }
   ```

3. Create `src/globals.css` with Tailwind layers + shadcn CSS vars + dark/light theme. `@tailwind base` is intentionally omitted:
   ```css
   @tailwind components;
   @tailwind utilities;

   :root {
     --background: 0 0% 7%;
     --foreground: 240 4% 94%;
     --primary: 37 91% 55%;
     --primary-foreground: 0 0% 9%;
     --border: 240 4% 14%;
     --input: 240 6% 13%;
     --ring: 220 89% 64%;
     --radius: 0.625rem;
   }
   body[data-theme='light'] {
     --background: 240 14% 97%;
     --foreground: 240 9% 9%;
     --primary: 37 91% 55%;
     --border: 240 4% 90%;
   }
   ```
   **Note on focus rings:** Omitting `@tailwind base` may silently break shadcn focus-ring styles (`ring-offset-background`). After installing each shadcn component, verify focus states in-browser. If rings are missing, add the relevant base rule manually to `globals.css` (e.g. `*, *::before, *::after { --tw-ring-offset-color: hsl(var(--background)); }`).

4. Update `main.tsx` imports — both files coexist during the migration:
   ```ts
   import './globals.css'   // Tailwind utilities + shadcn vars
   import './styles.css'    // legacy layout classes — emptied gradually across P1/P2
   ```

**New component files (P1):**
- `src/components/StatusBadge.tsx` — extracted from inline JSX, uses shadcn Badge
- `src/components/MetricChip.tsx` — extracted, uses shadcn Badge variant
- `src/components/KpiCard.tsx` — already extracted; migrate class names to Tailwind
- `src/components/CommandPalette.tsx` — extracted from App.tsx, uses shadcn Dialog + Command
- `src/components/RunDetailDrawer.tsx` — extracted from App.tsx, uses shadcn Sheet

**Migration order within P1:**
1. Install Tailwind + configure `tailwind.config.js` (Preflight off) + create `postcss.config.js`
2. Create `globals.css`, update `main.tsx` imports
3. Migrate `KpiCard.tsx` (already a separate file, safest first target)
4. Migrate `CommandPalette` (high isolation, good test of shadcn Dialog + Command)
5. Migrate `RunDetailDrawer` (uses shadcn Sheet)
6. Replace all button/input/select/badge instances in remaining App.tsx JSX
7. Delete migrated class blocks from `styles.css`

---

### Phase 2 — Component Split + ECharts

**Goal:** Break `App.tsx` into focused components. Replace SVG charts with ECharts.

**New dependencies:**
- `echarts`
- `echarts-for-react`
- `zustand`

**UI types migration — step 0 of Phase 2:**

Before creating the Zustand store, move the following types out of `App.tsx` into `src/types.ts` so they are importable by the store without circular dependencies:
- `ViewStateSnapshot`
- `DashboardModuleId`
- `DashboardModuleLayoutItem`
- `DashboardModuleSpan`
- `SortMode`
- `GroupMode`
- `SweepAxisMode`
- `SavedView`

These are currently local type declarations in `App.tsx`. After moving them, `App.tsx` replaces each declaration with an import from `./types`.

**Run selection — canonical representation:**

`selectedRunIds` is stored as `string[]` everywhere (store, `ViewStateSnapshot`, URL serialization, localStorage). This is required for JSON-serializability. Components that need Set-like lookup derive it locally via `useMemo`:
```typescript
const selectedSet = useMemo(() => new Set(viewState.selectedRunIds), [viewState.selectedRunIds]);
```

**Zustand store (`src/store/useAppStore.ts`):**

```typescript
interface AppState {
  // data
  experiments: Experiment[];
  selectedExp: string | null;
  runs: Run[];
  // selection
  activeRunId: string | null;
  // view config (selectedRunIds lives inside viewState as string[])
  viewState: ViewStateSnapshot;
  moduleLayout: DashboardModuleLayoutItem[];
  theme: 'dark' | 'light';
  // drag state — kept for Phase 2, replaced in Phase 3 by Framer Motion
  draggedModuleId: DashboardModuleId | null;
  dropTargetModuleId: DashboardModuleId | null;

  // data actions
  setExperiment: (name: string) => void;
  setRuns: (runs: Run[]) => void;
  patchRun: (id: string, patch: Partial<Run>) => void;  // used by Phase 4 SSE
  addRun: (run: Run) => void;                            // used by Phase 4 SSE
  removeRun: (id: string) => void;                       // used by Phase 4 SSE

  // selection actions
  setActiveRun: (id: string | null) => void;
  toggleRunSelection: (id: string) => void;

  // view actions
  setViewState: (patch: Partial<ViewStateSnapshot>) => void;
  reorderModules: (fromId: DashboardModuleId, toId: DashboardModuleId) => void;
  setTheme: (t: 'dark' | 'light') => void;

  // drag actions (temporary — removed in Phase 3)
  setDraggedModule: (id: DashboardModuleId | null) => void;
  setDropTargetModule: (id: DashboardModuleId | null) => void;
}
```

**`setActiveRun` and async fetch lifecycle:**

`setActiveRun(id)` only updates store state. The async data fetch lives in a `useEffect` inside `RunDetailDrawer.tsx` watching `activeRunId`:

```typescript
// RunDetailDrawer.tsx
const activeRunId = useAppStore(s => s.activeRunId);
useEffect(() => {
  if (!activeRunId) { setDetail(null); return; }
  let cancelled = false;
  Promise.all([
    getRunDetail(activeRunId),
    getMetricHistory(activeRunId),
    getArtifacts(activeRunId),
  ]).then(([d, h, a]) => {
    if (!cancelled) setDetail({ ...d, history: h, artifacts: a });
  });
  return () => { cancelled = true; };
}, [activeRunId]);
```

**Component split:**

| File | Extracted from | Responsibility |
|---|---|---|
| `src/layout/ExperimentSidebar.tsx` | App.tsx sidebar JSX | Experiment list nav, search |
| `src/layout/RunSidebar.tsx` | App.tsx run-sidebar JSX | Selected runs rail |
| `src/layout/TopBar.tsx` | App.tsx top-bar JSX | Breadcrumbs, actions, connection status |
| `src/components/RunsTable.tsx` | App.tsx runs table JSX | Filterable/sortable/groupable run table |
| `src/modules/LeaderboardModule.tsx` | App.tsx leaderboard section | Top-N runs by metric |
| `src/modules/SweepModule.tsx` | App.tsx sweep section | Scatter plot wrapper |
| `src/modules/ParallelModule.tsx` | App.tsx parallel section | Parallel coords wrapper |
| `src/modules/ImportanceModule.tsx` | App.tsx importance section | Param importance bar chart |
| `src/modules/CompareModule.tsx` | App.tsx compare section | Side-by-side run table |

**ECharts chart components — prop APIs:**

All chart components receive pre-computed data from their parent module. Modules handle `useMemo` data-wrangling; chart components are pure renderers.

```typescript
// MetricHistoryChart.tsx
interface MetricHistoryChartProps {
  series: Array<{ name: string; color: string; data: Array<[number, number]> }>; // [step, value]
  theme: 'dark' | 'light';
}

// SweepScatterChart.tsx
interface SweepScatterChartProps {
  points: Array<{ run: Run; x: number; y: number; color: string }>;
  xLabel: string;
  yLabel: string;
  theme: 'dark' | 'light';
  onSelect?: (runIds: string[]) => void;  // ECharts brush callback
}

// ParallelCoordsChart.tsx
interface ParallelCoordsChartProps {
  dimensions: string[];
  series: Array<{ run: Run; values: number[]; color: string }>;
  theme: 'dark' | 'light';
}

// ImportanceBarChart.tsx
interface ImportanceBarChartProps {
  items: Array<{ label: string; score: number; metricKey: string }>;
  theme: 'dark' | 'light';
}
```

**ECharts theme config:**
- Create `src/charts/echartsTheme.ts` — dark theme object matching existing palette (#f6a623, #4f8ef7, #34d399, bg #1a1a1f)
- Register on app init: `echarts.registerTheme('mltracker', theme)`

**Drag-and-drop in Phase 2:**

The existing HTML5 drag state (`draggedModuleId`, `dropTargetModuleId`) is lifted into the Zustand store (see above). Module components receive drag handlers as props from the App.tsx shell. These store fields and handlers are removed entirely in Phase 3 when Framer Motion Reorder takes over.

**Migration order within P2:**
0. Move UI types (`ViewStateSnapshot`, `DashboardModuleId`, etc.) from `App.tsx` to `types.ts`
1. Add Zustand, create store, thread into App.tsx including drag state lift
2. Extract layout components (ExperimentSidebar, RunSidebar, TopBar)
3. Extract RunsTable
4. Extract modules one by one
5. Add ECharts + theme; implement MetricHistoryChart first (drawer, low risk)
6. Wire SweepScatterChart -> SweepModule
7. Wire ParallelCoordsChart -> ParallelModule (most complex, last)
8. Wire ImportanceBarChart -> ImportanceModule

---

### Phase 3 — Visual Polish

**Goal:** Add motion, animated metrics, richer card aesthetics.

**New dependencies:**
- `framer-motion`

**Additions:**
- `AnimatePresence` + `motion.div` on module cards — fade+slide on mount/unmount
- `motion.div` on run rows — staggered entrance when experiment loads
- KPI counter animation — `useMotionValue` + `useSpring` + `useTransform` for count-up on mount
- Drag-and-drop reorder — replace HTML5 drag (store fields `draggedModuleId`/`dropTargetModuleId` removed) with `Reorder.Group`; `reorderModules` store action still called on drop to persist order
- Topbar connection pill — pulsing animation on RUNNING status
- Sheet drawer — slide easing tuned via CSS (Radix already handles open/close animation)
- Background gradient — animated shift on theme toggle

---

### Phase 4 — Real-time SSE

**Goal:** Live metric streaming for running experiments without polling.

**Backend requirement:**
- New endpoint: `GET /experiments/{name}/stream` returning `text/event-stream`
- Events:
  - `event: run_update` — `data: {"run_id": "...", "metrics": {...}, "status": "RUNNING"}`
  - `event: run_added` — `data: {"run": {...}}`
  - `event: run_deleted` — `data: {"run_id": "..."}`
  - `event: heartbeat` — `data: {}` every 15s
- Server sends `retry: 0` to suppress native `EventSource` reconnect (client handles it)

**Frontend SSE implementation — `fetch`-based (not native `EventSource`):**

Native `EventSource` reconnect is fixed-interval and not overridable. Using `fetch` with a `ReadableStream` decoder allows exponential backoff:

```typescript
function useSSE(experimentName: string | null): { isLive: boolean }
// - Connects when experimentName is non-null
// - Reconnect: initial 1s, multiplier 2x, max 30s, +10% random jitter
// - Cancellation via AbortController on unmount
// - isLive: true when connection is open and heartbeats are received
```

**Event dispatch:**
- `run_update` -> `store.patchRun(run_id, { metrics, status })`
- `run_added` -> `store.addRun(run)`
- `run_deleted` -> `store.removeRun(run_id)`; if `activeRunId === run_id` call `store.setActiveRun(null)`; remove from `viewState.selectedRunIds` via `store.setViewState`

**UI changes:**
- `src/charts/RunSparkline.tsx` — ECharts `line` chart (60px height) on RUNNING run rows, last 20 values of primary metric
- RUNNING status badge — animated pulse dot via Framer Motion `animate={{ scale: [1, 1.4, 1] }}` on loop
- Top bar connection pill shows "Live" when `isLive === true`

---

## File Structure Summary

```
src/
  api/
    client.ts               (unchanged)
  charts/
    echartsTheme.ts         (P2)
    MetricHistoryChart.tsx  (P2)
    SweepScatterChart.tsx   (P2)
    ParallelCoordsChart.tsx (P2)
    ImportanceBarChart.tsx  (P2)
    RunSparkline.tsx        (P4)
  components/
    ui/                     (shadcn generated, P1)
    CommandPalette.tsx      (P1)
    KpiCard.tsx             (P1, migrate existing)
    MetricChip.tsx          (P1)
    RunDetailDrawer.tsx     (P1)
    StatusBadge.tsx         (P1)
  hooks/
    useSSE.ts               (P4)
  layout/
    ExperimentSidebar.tsx   (P2)
    RunSidebar.tsx          (P2)
    TopBar.tsx              (P2)
  modules/
    CompareModule.tsx       (P2)
    ImportanceModule.tsx    (P2)
    LeaderboardModule.tsx   (P2)
    ParallelModule.tsx      (P2)
    SweepModule.tsx         (P2)
  store/
    useAppStore.ts          (P2)
  App.tsx                   (shrinks to routing shell ~150 lines)
  globals.css               (P1, Tailwind utilities + shadcn vars, no Preflight)
  main.tsx                  (import order updated in P1)
  styles.css                (gradually emptied P1-P2, deleted at end of P2)
  types.ts                  (unchanged)
postcss.config.js           (P1, project root)
tailwind.config.js          (P1, project root, Preflight disabled)
```

---

## What Is Not Changing

- `api/client.ts` — already clean, no changes needed
- `types.ts` — data types stable; `ViewStateSnapshot.selectedRunIds` stays `string[]`
- URL hash state scheme (`#exp/<name>`) — preserved for compatibility
- localStorage keys — preserved so saved views survive the upgrade
- Python backend mltracker server — unchanged until Phase 4 SSE endpoint

---

## Success Criteria Per Phase

| Phase | Done when |
|---|---|
| P1 | All replaced class names (`.action-btn`, `.icon-btn`, `.search-input`, `.sort-select`, `.status-badge`, `.metric-chip`, `.tag-chip`, palette overlay styles, drawer styles) are absent from `styles.css`; no visual regression |
| P2 | `App.tsx` < 200 lines; all 5 modules render ECharts; Zustand is sole source of truth; no view-state `useState` in App.tsx |
| P3 | Module cards animate in/out; KPI counters count up on load; drag reorder uses Framer Motion Reorder; no HTML5 drag handlers remain |
| P4 | RUNNING runs update metrics live without page refresh; sparklines on RUNNING run rows; SSE reconnects with exponential backoff after disconnect |
