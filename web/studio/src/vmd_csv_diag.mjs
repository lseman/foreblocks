import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const { computeVmdDiagnostics } = await import('./lib/diagnostics-vmd.js');

const csvPath = path.resolve(__dirname, 'dist', 'sample_reservoir.csv');
const text = await fs.promises.readFile(csvPath, 'utf8');
const lines = text.trim().split(/\r?\n/);
const header = lines[0].split(',');
const rows = lines.slice(1).map((line) => line.split(',').map((cell) => cell.trim()));
const fields = ['inflow_m3s', 'reservoir_level_pct', 'rainfall_mm', 'temperature_c'];
for (const field of fields) {
  const idx = header.indexOf(field);
  const series = rows.map((row) => ({ value: Number(row[idx]) }));
  const out = computeVmdDiagnostics(series);
  const residualNaN = out.residual?.values.reduce((acc, v, i) => (Number.isNaN(v) ? acc.concat(i) : acc), []);
  const compNaN = out.components.map((c, k) => ({ k, indices: c.values.reduce((acc, v, i) => (Number.isNaN(v) ? acc.concat(i) : acc), []) }));
  console.log('FIELD', field);
  console.log(' available', out.available);
  console.log(' anyNaNResidual', residualNaN.length > 0, residualNaN.slice(0, 10));
  console.log(' anyNaNComponents', compNaN.filter((x) => x.indices.length > 0));
  console.log('first component NaN lengths', compNaN.map((x) => x.indices.length));
  console.log('---');
}
