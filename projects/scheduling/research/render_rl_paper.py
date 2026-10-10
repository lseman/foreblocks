"""Generate paper2 tables and numerical macros from saved experiment records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LABELS = {
    'legacy_ppo': 'Original PPO', 'stable_ppo': 'Stabilized PPO',
    'graph_ppo': 'Constraint-graph PPO', 'graph_preference': 'Graph preference',
    'graph_preference_ppo': 'Graph preference + PPO',
    'uniform': 'Uniform', 'utility': 'Utility rule', 'bandit': 'Contextual bandit',
    'profile_heuristic': 'Profile heuristic', 'profile_only': 'Uniform profile-only',
    'untrained': 'Untrained attention policy', 'preference': 'Rate preference',
    'ppo': 'Portfolio PPO', 'preference_ppo': 'Rate preference + PPO',
}


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def mean(rows, split, method):
    by_instance = {}
    for row in rows:
        if row['split']==split and row['method']==method:
            by_instance.setdefault(row['instance'], []).append(row['improvement'])
    return float(np.mean([np.mean(v) for v in by_instance.values()]))


def summary_table(rows, methods, path, caption, label):
    lines = [r'\begin{table}[tb]', r'\centering', r'\small',
             '\\caption{'+caption+'}', '\\label{'+label+'}',
             r'\begin{tabular}{lrr}', r'\toprule',
             r'Selector & Development & Larger instances \\', r'\midrule']
    for method in methods:
        lines.append(f"{LABELS[method]} & {mean(rows, 'development', method):.6f} & {mean(rows, 'ood', method):.6f} \\")
        # The source string above contains one trailing slash; LaTeX needs two.
        lines[-1] += '\\'
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    path.write_text('\n'.join(lines)+'\n')


def comparison_table(comparisons, selected, path, caption, label):
    lines = [r'\begin{table}[tb]', r'\centering', r'\small',
             '\\caption{'+caption+'}', '\\label{'+label+'}',
             r'\resizebox{\linewidth}{!}{%', r'\begin{tabular}{llrr}', r'\toprule',
             r'Split & Comparison & Difference & Marginal 95\% interval \\', r'\midrule']
    for c in comparisons:
        if (c['method'], c['baseline']) not in selected:
            continue
        lo, hi = c['delta']['ci95']
        split = 'Development' if c['split']=='development' else 'Larger'
        lines.append(f"{split} & {LABELS[c['method']]} vs. {LABELS[c['baseline']]} & "
                     f"{c['delta']['mean']:.6f} & [{lo:.6f}, {hi:.6f}] " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'}', r'\end{table}']
    path.write_text('\n'.join(lines)+'\n')


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--diagnostic-results', type=Path, default=root/'research/results/rl-upgrade-20261006')
    parser.add_argument('--portfolio-results', type=Path, default=root/'research/results/rl-portfolio-20261006')
    parser.add_argument('--paper', type=Path, default=root/'paper2')
    args = parser.parse_args()
    tables = args.paper/'tables'
    tables.mkdir(parents=True, exist_ok=True)
    diagnostic = records(args.diagnostic_results/'evaluation.jsonl')
    comparisons = json.loads((args.diagnostic_results/'analysis.json').read_text())
    methods = ('uniform', 'utility', 'bandit', 'legacy_ppo', 'stable_ppo', 'graph_ppo',
               'graph_preference', 'graph_preference_ppo', 'profile_heuristic')
    caption = ('MILP-only selection study: mean normalized incumbent improvement at 1.2 s. '
               'Each instance is averaged over two policy/evaluation seeds before aggregation. '
               'Development comprises six instances; larger-instance evaluation comprises three. '
               'All 216 runs, including validation, returned feasible schedules. '
               'The profile heuristic uses a different move portfolio.')
    summary_table(diagnostic, methods, tables/'milp_summary.tex', caption, 'tab:milp')
    comparison_table(comparisons, [('stable_ppo', 'legacy_ppo'), ('graph_ppo', 'stable_ppo'),
                                  ('graph_preference_ppo', 'graph_preference')],
                     tables/'milp_pairs.tex', 'Exploratory paired differences in normalized improvement. '
                     'Intervals bootstrap instance means and have no multiplicity correction.', 'tab:milppairs')
    macros = {
        'MILPLegacyIID': mean(diagnostic, 'development', 'legacy_ppo'),
        'MILPStableIID': mean(diagnostic, 'development', 'stable_ppo'),
        'MILPGraphIID': mean(diagnostic, 'development', 'graph_ppo'),
        'MILPHeuristicOOD': mean(diagnostic, 'ood', 'profile_heuristic'),
        'MILPWarmOOD': mean(diagnostic, 'ood', 'graph_preference_ppo'),
    }
    portfolio = records(args.portfolio_results/'evaluation.jsonl')
    integrity = json.loads((args.portfolio_results/'integrity.json').read_text())
    if len(portfolio) != integrity['runs'] or not integrity['all_feasible']:
        raise RuntimeError('Portfolio results are incomplete or infeasible')
    plan = json.loads((args.portfolio_results/'plan.json').read_text())
    methods = plan['methods']
    summary_table(portfolio, methods, tables/'portfolio_summary.tex',
                  f"Profile/MILP portfolio study: mean normalized incumbent improvement at {plan['eval_budget']:g} s. "
                  'Uniform, contextual bandit, untrained and learned policies share the candidate-action interface. '
                  'The profile-only control excludes MILP actions; the original profile heuristic uses its own controller. '
                  'Development comprises six instances and larger-instance evaluation three, each with two seeds. '
                  f"All {len(portfolio)} runs returned feasible schedules.", 'tab:portfolio')
    comparison_table(json.loads((args.portfolio_results/'analysis.json').read_text()),
                     [('ppo', 'untrained'), ('preference_ppo', 'preference'),
                      ('preference_ppo', 'bandit'), ('preference_ppo', 'profile_heuristic')],
                     tables/'portfolio_pairs.tex', 'Exploratory paired portfolio comparisons in normalized improvement. '
                     'Marginal intervals bootstrap instance means; three larger instances provide limited resampling support.',
                     'tab:portfoliopairs')
    for macro, split, method in [('PortfolioPPOIID', 'development', 'ppo'),
                                 ('PortfolioWarmIID', 'development', 'preference_ppo'),
                                 ('PortfolioBanditIID', 'development', 'bandit'),
                                 ('PortfolioHeuristicIID', 'development', 'profile_heuristic'),
                                 ('PortfolioWarmOOD', 'ood', 'preference_ppo'),
                                 ('PortfolioHeuristicOOD', 'ood', 'profile_heuristic')]:
        macros[macro] = mean(portfolio, split, method)
    lines = ['% Generated from saved experiment records; do not edit numbers manually.']
    lines += ['\\newcommand{\\'+name+'}{'+f'{value:.6f}'+'}' for name, value in macros.items()]
    lines += ['\\newcommand{\\PortfolioRuns}{'+str(len(portfolio))+'}',
              '\\newcommand{\\PortfolioEpisodes}{'+str(plan['episodes'])+'}']
    (args.paper/'numbers.tex').write_text('\n'.join(lines)+'\n')
    print(f'Generated four tables and numerical macros under {args.paper}')


if __name__ == '__main__':
    main()
