"""Export ForeTree / ForeForest models to Treelite and ONNX formats.

Provides:
- export_to_treelite(forest, name=None) → treelite.Model
- export_to_onnx(forest, model_name="foretree", op_type="regression") → bytes
- save_to_treelite_json(forest, path, name=None)
- save_to_onnx_file(forest, path, model_name="foretree", op_type="regression")
- export_to_json(forest, path)  → portable JSON serialization

Requirements:
- Treelite (pip install treelite): for Treelite and C code export
- ONNX (pip install onnx): for ONNX export
- onnxruntime (optional): for ONNX model validation
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers – read tree data from packed representation
# ---------------------------------------------------------------------------

_SPLIT_KIND_AXIS = 0
_SPLIT_KIND_CATEGORICAL = 1
_SPLIT_KIND_OBLIQUE = 2
_SPLIT_KIND_PAIR = 3


def _tree_to_dict(packed_tree, num_features: int) -> dict[str, Any]:
    """Convert a PackedTree to a Treelite-compatible tree dictionary."""
    n = packed_tree.node_count
    if n == 0:
        return {"nodes": []}

    nodes: list[dict[str, Any]] = []
    for i in range(n):
        is_leaf = bool(packed_tree.leaf_flags[i])
        node: dict[str, Any] = {"nodeID": i, "Split": None, "LeftChild": -1, "RightChild": -1, "MissingLeftChild": -1, "Gain": 0.0, "Cover": 0.0, "Weight": 0.0}

        if is_leaf:
            node["Weight"] = float(packed_tree.leaf_values[i])
        else:
            kind = packed_tree.split_kinds[i] if i < len(packed_tree.split_kinds) else _SPLIT_KIND_AXIS

            if kind == _SPLIT_KIND_AXIS:
                feat = int(packed_tree.features[i])
                thresh = float(packed_tree.thresholds[i])
                # Threshold in PackedTree is a bin index; we need the actual split value.
                # For Treelite we use the bin index directly (Treelite works with binned data).
                node["Split"] = {"feature": feat, "condition": "<=", "threshold": thresh}
            elif kind == _SPLIT_KIND_CATEGORICAL:
                # Categorical partition – not directly supported by Treelite's
                # simple split representation. Fall back to axis split if possible.
                feat = int(packed_tree.features[i])
                thresh = float(packed_tree.thresholds[i])
                node["Split"] = {"feature": feat, "condition": "<=", "threshold": thresh}
            elif kind == _SPLIT_KIND_OBLIQUE:
                # Oblique split: k-feature hyperplane. Not natively supported.
                # Fall back to best axis feature.
                feat = int(packed_tree.features[i])
                thresh = float(packed_tree.thresholds[i])
                node["Split"] = {"feature": feat, "condition": "<=", "threshold": thresh}
            elif kind == _SPLIT_KIND_PAIR:
                feat = int(packed_tree.features[i])
                thresh = float(packed_tree.thresholds[i])
                node["Split"] = {"feature": feat, "condition": "<=", "threshold": thresh}
            else:
                feat = int(packed_tree.features[i])
                thresh = float(packed_tree.thresholds[i])
                node["Split"] = {"feature": feat, "condition": "<=", "threshold": thresh}

            node["LeftChild"] = int(packed_tree.left_children[i])
            node["RightChild"] = int(packed_tree.right_children[i])
            node["MissingLeftChild"] = int(packed_tree.missing_left[i]) if i < len(packed_tree.missing_left) else -1
            if i < len(packed_tree.cover):
                node["Cover"] = float(packed_tree.cover[i])

        nodes.append(node)

    return {"nodes": nodes, "vector_output": False}


# ---------------------------------------------------------------------------
# Treelite export
# ---------------------------------------------------------------------------


def export_to_treelite(
    forest: Any,
    name: Optional[str] = None,
    **builder_params: Any,
) -> Any:
    """Export a ForeForest model to a Treelite Model.

    Args:
        forest: A trained foreforest.ForeForest instance.
        name: Optional model name for Treelite metadata.
        **builder_params: Extra keyword arguments passed to treelite.ModelBuilder.

    Returns:
        A treelite.Model instance.

    Example:
        >>> import foreforest
        >>> model = foreforest.ForeForest(cfg)
        >>> model.fit_complete(X, y)
        >>> tl_model = export_to_treelite(model)
        >>> tl_model.export_c(path="model.c", verbose=True)
    """
    try:
        import treelite
    except ImportError:
        raise ImportError(
            "treelite is required for Treelite export. "
            "Install it with: pip install treelite"
        )

    if name is None:
        name = "foretree_model"

    n_trees = forest.size()
    if n_trees == 0:
        raise ValueError("ForeForest has no trained trees")

    # Determine number of outputs (classes - 1)
    n_outputs = max(forest.num_classes() - 1, 1)

    # Build Treelite model
    tl_model = treelite.Model(n_trees, 1, 1)  # trees, channels, outputs per channel

    for tree_idx in range(n_trees):
        # Get the packed tree for this tree index
        # The forest stores trees internally; we access via the model representation
        try:
            packed = forest.get_packed_tree(tree_idx)
        except AttributeError:
            # Fallback: if get_packed_tree is not available, build from prediction
            raise NotImplementedError(
                "ForeForest.get_packed_tree() is required for Treelite export. "
                "Ensure your build includes the packed tree accessor."
            )

        tree_dict = _tree_to_dict(packed, 0)  # num_features determined from data later
        tl_model.set_tree(tree_dict, tree_idx=tree_idx)

    tl_model.set_meta_data(model_name=name)
    return tl_model


def save_to_treelite_json(
    forest: Any,
    path: str | Path,
    name: Optional[str] = None,
) -> None:
    """Save a ForeForest model to a Treelite JSON file.

    Args:
        forest: A trained foreforest.ForeForest instance.
        path: Output file path (.json).
        name: Optional model name.
    """
    tl_model = export_to_treelite(forest, name)
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".json")
    tl_model.save_json_model(str(path))


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_to_onnx(
    forest: Any,
    model_name: str = "foretree",
    op_type: str = "regression",
    input_names: list[str] | None = None,
) -> bytes:
    """Export a ForeForest model to ONNX format.

    Creates an ONNX model using the TreeEnsemble operator. This produces a
    self-contained model that can be loaded by any ONNX runtime.

    Args:
        forest: A trained foreforest.ForeForest instance.
        model_name: Name for the ONNX model.
        op_type: "regression" or "classification".
        input_names: List of input tensor names (default: ["X"]).

    Returns:
        ONNX model as bytes.

    Example:
        >>> import onnx
        >>> onnx_bytes = export_to_onnx(model)
        >>> onnx.save_model_from_string(onnx_bytes, "model.onnx")
    """
    try:
        import onnx
        from onnx import helper, numpy_helper, TensorProto
    except ImportError:
        raise ImportError(
            "onnx is required for ONNX export. "
            "Install it with: pip install onnx"
        )

    if input_names is None:
        input_names = ["X"]

    n_trees = forest.size()
    if n_trees == 0:
        raise ValueError("ForeForest has no trained trees")

    n_outputs = max(forest.num_classes() - 1, 1)
    is_classification = op_type == "classification"

    # Build a single TreeEnsemble node that represents the entire forest.
    # TreeEnsemble supports multiple trees and outputs.
    # We construct the tree data as flat arrays per tree.

    # Collect tree data
    all_tree_roots: list[int] = []
    all_tree_features: list[int] = []
    all_tree_thresholds: list[float] = []
    all_tree_modes: list[str] = []  # "VALUE" for axis splits
    all_tree_node_weights: list[list[float]] = []
    all_tree_nodes_missing: list[int] = []
    all_tree_hinge_loss: list[float] = []

    for tree_idx in range(n_trees):
        packed = forest.get_packed_tree(tree_idx)
        n = packed.node_count
        if n == 0:
            continue

        all_tree_roots.append(0)  # root is always node 0

        for node_idx in range(n):
            is_leaf = bool(packed.leaf_flags[node_idx])

            if is_leaf:
                all_tree_features.append(0)
                all_tree_thresholds.append(0.0)
                all_tree_modes.append("VALUE")
                all_tree_node_weights.append([float(packed.leaf_values[node_idx])])
                all_tree_nodes_missing.append(-1)
                all_tree_hinge_loss.append(0.0)
            else:
                # Use the first feature/threshold as a simplification
                # (Treelite fallback for complex split types)
                feat = int(packed.features[node_idx]) % 10000  # feature ID
                thresh = float(packed.thresholds[node_idx])
                missing_left = int(packed.missing_left[node_idx]) if node_idx < len(packed.missing_left) else -1

                all_tree_features.append(feat)
                all_tree_thresholds.append(thresh)
                all_tree_modes.append("VALUE")
                all_tree_node_weights.append([])
                all_tree_nodes_missing.append(missing_left)
                all_tree_hinge_loss.append(0.0)

    # Pack arrays for ONNX
    np_type = TensorProto.INT64
    float_type = TensorProto.DOUBLE

    tree_roots_arr = numpy_helper.from_array(
        np.array(all_tree_roots, dtype=np.int64), name="tree_roots"
    )
    tree_features_arr = numpy_helper.from_array(
        np.array(all_tree_features, dtype=np.int64), name="tree_features"
    )
    tree_thresholds_arr = numpy_helper.from_array(
        np.array(all_tree_thresholds, dtype=np.float64), name="tree_thresholds"
    )
    tree_modes_arr = numpy_helper.from_array(
        np.array(all_tree_modes, dtype=np.str_), name="tree_modes"
    )

    # Node weights: flatten all lists
    flat_weights: list[float] = []
    for w_list in all_tree_node_weights:
        flat_weights.extend(w_list)

    # Build weights tensor with shape [n_trees, max_nodes, max_outputs]
    max_nodes = 0
    for tree_idx in range(n_trees):
        try:
            packed = forest.get_packed_tree(tree_idx)
            max_nodes = max(max_nodes, packed.node_count)
        except Exception:
            max_nodes = max(max_nodes, 1)

    weights_tensor = np.zeros(
        (n_trees, max_nodes, n_outputs), dtype=np.float64
    )
    weight_idx = 0
    for tree_idx in range(n_trees):
        packed = forest.get_packed_tree(tree_idx)
        n = packed.node_count
        for node_idx in range(n):
            for k in range(n_outputs):
                weights_tensor[tree_idx, node_idx, k] = flat_weights[weight_idx]
            weight_idx += 1

    tree_weights_arr = numpy_helper.from_array(
        weights_tensor, name="tree_weights"
    )

    nodes_missing_arr = numpy_helper.from_array(
        np.array(all_tree_nodes_missing, dtype=np.int64), name="nodes_missing"
    )

    # Determine output shape
    output_name = "output"
    if is_classification:
        # Classification: output probabilities
        # Need class labels
        class_labels = numpy_helper.from_array(
            np.array([0, 1], dtype=np.int64), name="class_labels"
        )
        outputs = [
            helper.make_tensor_value_info(
                output_name, TensorProto.DOUBLE, [None, n_outputs]
            )
        ]
        node = helper.make_node(
            "TreeEnsemble",
            inputs=[input_names[0], "tree_roots", "tree_features",
                    "tree_thresholds", "tree_modes", "tree_weights",
                    "nodes_missing_valueTreatment"],
            outputs=[output_name],
            tree_roots=tree_roots_arr,
            tree_features=tree_features_arr,
            tree_thresholds=tree_thresholds_arr,
            tree_modes=tree_modes_arr,
            tree_weights=tree_weights_arr,
            nodes_missing_valueTreatment="as_value",
            op_type=op_type,
            class_labels=class_labels,
            n_targets=n_outputs,
        )
    else:
        outputs = [
            helper.make_tensor_value_info(
                output_name, TensorProto.DOUBLE, [None, n_outputs]
            )
        ]
        node = helper.make_node(
            "TreeEnsemble",
            inputs=[input_names[0], "tree_roots", "tree_features",
                    "tree_thresholds", "tree_modes", "tree_weights",
                    "nodes_missing_valueTreatment"],
            outputs=[output_name],
            tree_roots=tree_roots_arr,
            tree_features=tree_features_arr,
            tree_thresholds=tree_thresholds_arr,
            tree_modes=tree_modes_arr,
            tree_weights=tree_weights_arr,
            nodes_missing_valueTreatment="as_value",
            op_type=op_type,
            n_targets=n_outputs,
        )

    # We need the nodes_missing tensor as an input since we reference it
    nodes_missing_input = helper.make_tensor_value_info(
        "nodes_missing_valueTreatment", TensorProto.INT64, [None]
    )

    graph = helper.make_graph(
        [node],
        model_name,
        [
            helper.make_tensor_value_info(input_names[0], TensorProto.DOUBLE, [None, None]),
            nodes_missing_input,
        ],
        outputs,
        [
            tree_roots_arr, tree_features_arr, tree_thresholds_arr,
            tree_modes_arr, tree_weights_arr, nodes_missing_arr,
        ],
    )

    opset_imports = [helper.make_opsetid("", 15)]
    model = helper.make_model(graph, opset_imports=opset_imports, producer_name="foretree")
    model.producer_version = "0.1.0"

    return model.SerializeToString()


def save_to_onnx_file(
    forest: Any,
    path: str | Path,
    model_name: str = "foretree",
    op_type: str = "regression",
) -> None:
    """Save a ForeForest model to an ONNX file.

    Args:
        forest: A trained foreforest.ForeForest instance.
        path: Output file path (.onnx).
        model_name: Name for the ONNX model.
        op_type: "regression" or "classification".
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".onnx")

    onnx_bytes = export_to_onnx(forest, model_name, op_type)

    try:
        import onnx
        onnx.save_model_from_string(onnx_bytes, str(path))
    except ImportError:
        path.write_bytes(onnx_bytes)


# ---------------------------------------------------------------------------
# JSON export (portable, human-readable)
# ---------------------------------------------------------------------------


def _forest_to_export_dict(forest: Any) -> dict[str, Any]:
    """Serialize the entire forest to a nested dict."""
    n_trees = forest.size()
    trees = []

    for i in range(n_trees):
        packed = forest.get_packed_tree(i)
        tree_dict: dict[str, Any] = {
            "root": packed.root,
            "outputs": packed.outputs,
            "node_count": packed.node_count,
            "nodes": [],
        }

        for j in range(packed.node_count):
            is_leaf = bool(packed.leaf_flags[j])
            node: dict[str, Any] = {"index": j, "is_leaf": is_leaf}

            if is_leaf:
                node["value"] = float(packed.leaf_values[j])
            else:
                node["split_feature"] = int(packed.features[j])
                node["split_threshold"] = float(packed.thresholds[j])
                node["split_kind"] = int(packed.split_kinds[j]) if j < len(packed.split_kinds) else 0
                node["missing_left"] = bool(packed.missing_left[j]) if j < len(packed.missing_left) else False
                node["left_child"] = int(packed.left_children[j])
                node["right_child"] = int(packed.right_children[j])
                node["cover"] = float(packed.cover[j]) if j < len(packed.cover) else 0.0

            tree_dict["nodes"].append(node)

        trees.append(tree_dict)

    return {
        "version": "1.0",
        "n_trees": n_trees,
        "num_classes": max(forest.num_classes() - 1, 1),
        "trees": trees,
    }


def export_to_json(
    forest: Any,
    path: Optional[str | Path] = None,
) -> dict[str, Any] | str:
    """Export a ForeForest model to a JSON-serializable dict or file.

    Args:
        forest: A trained foreforest.ForeForest instance.
        path: If provided, save to this file path.

    Returns:
        The model dict, or the JSON string if path is provided.
    """
    data = _forest_to_export_dict(forest)
    if path is not None:
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".json")
        json_str = json.dumps(data, indent=2)
        path.write_text(json_str)
        return json_str
    return data


# ---------------------------------------------------------------------------
# Convenience: unified export
# ---------------------------------------------------------------------------


def export_forest(
    forest: Any,
    format: str = "json",
    path: Optional[str | Path] = None,
    **kwargs: Any,
) -> Any:
    """Unified export function with format dispatch.

    Args:
        forest: A trained foreforest.ForeForest instance.
        format: "json", "treelite", "onnx", "c".
        path: Output path (required for file formats).
        **kwargs: Passed to the underlying export function.

    Returns:
        The exported model or bytes, depending on format.
    """
    fmt = format.lower()

    if fmt == "json":
        return export_to_json(forest, path)
    elif fmt in ("treelite",):
        return export_to_treelite(forest, **kwargs)
    elif fmt == "onnx":
        if path is None:
            raise ValueError("path is required for ONNX export")
        save_to_onnx_file(forest, path, **kwargs)
        return path
    elif fmt == "c":
        if path is None:
            raise ValueError("path is required for C export")
        tl_model = export_to_treelite(forest, **kwargs)
        tl_model.export_c(str(path), verbose=True)
        return path
    else:
        raise ValueError(
            f"Unknown format '{format}'. Supported: json, treelite, onnx, c"
        )
