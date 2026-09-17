# ONTS Neural Scheduling

Neural schedulers for the Offline Nanosatellite Task Scheduling (ONTS) MILP in
`original.py`.

## Structure

```text
scheduling/
├── environments/
│   ├── base.py
│   └── onts_env.py              # ONTS env, JSON loader, instance-pool env
├── examples/
│   └── train_onts.py            # PPO training entrypoint
├── instances/
│   ├── convert_examples.py      # .jl -> JSON converter
│   ├── generate_onts_instances.py
│   └── examples/                # original and converted ONTS instances
├── models/
│   ├── nco_model.py
│   ├── pointer_net.py
│   ├── bipartite_gnn.py
│   └── adapters.py
├── tests/
│   ├── test_bipartite_gnn.py
│   ├── test_graph_converter.py
│   └── test_modular.py
├── graph_converter.py
├── train.py                     # PPO trainer
└── original.py                  # reference MILP formulation
```

## Instances

Convert bundled Julia-style examples to JSON:

```bash
python instances/convert_examples.py instances/examples --out instances/examples
```

Generate feasible training instances:

```bash
.venv/bin/python instances/generate_onts_instances.py \
  --count 100 \
  --out instances/generated \
  --jobs 9 \
  --horizon 97
```

The generator prefers PuLP/CBC when available and falls back to SciPy MILP.

## Training

Train from one instance:

```bash
python examples/train_onts.py \
  --instance instances/examples/125_9.json \
  --encoder transformer
```

Train from a directory of generated instances:

```bash
python examples/train_onts.py \
  --instance-dir instances/generated \
  --encoder bipartite
```

The `bipartite` encoder enables graph observations with task nodes, valid
inequality constraint nodes, and typed bidirectional edges.

## Tests

```bash
python -m pytest tests -q
```
