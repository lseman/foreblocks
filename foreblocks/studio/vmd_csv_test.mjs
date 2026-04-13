import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const { computeVmdDiagnostics } = await import('./lib/diagnostics-vmd.js');

const csvPath = path.resolve(__dirname, 'dist', 'sample_reservoir.csv');
const text = await fs.promises.readFile(csvPath, 'utf8');
const lines = text.trim().split(/\r?\n/);
const header = lines[0].split(',');
console.log('header', header);
const rows = lines.slice(1).map((line) => line.split(',').map((cell) => cell.trim()));
const fields = ['inflow_m3s', 'reservoir_level_pct', 'rainfall_mm', 'temperature_c'];
for (const field of fields) {
  const idx = header.indexOf(field);
  if (idx < 0) continue;
  const series = rows.map((row) => ({ value: Number(row[idx]) }));
  const out = computeVmdDiagnostics(series);
  console.log('field', field, { available: out.available, modeCount: out.modeCount, anyNaNResidual: out.residual?.values.some((v) => Number.isNaN(v)), anyNaNComponents: out.components.some((c) => c.values.some((v) => Number.isNaN(v))), finalDiff: out.finalDiff, reason: out.reason });
}
