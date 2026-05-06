#!/usr/bin/env python3
“”“dequantize_onnx.py

Fold QuantizeLinear / DequantizeLinear ops in an ONNX graph into FP32
constants and drop them, producing a quantization-free (pseudo-float)
ONNX model that downstream tools (notably onnx2torch) can consume.

## Why this exists

TFLite EdgeTPU models are INT8. tf2onnx faithfully translates the
quantization scheme into the ONNX graph as QuantizeLinear /
DequantizeLinear nodes. onnx2torch has no converter registered for those
ops, so conversion fails:

```
NotImplementedError: Converter is not implemented
    (OperationDescription(operation_type=DequantizeLinear, ...))
```

This pass replaces every Q/DQ pair with the FP32 tensor it represents:

```
fp32_value = scale * (int_value - zero_point)
```

The model’s numerical behavior is preserved at INT8 fidelity, but the
graph contains only float ops – exactly what the team’s existing
convert_onnx_pytorch.py expects.

## Scope (deliberately narrow)

Handles the patterns tf2onnx produces for TFLite-INT8 models:

```
1. Constant initializer -> DequantizeLinear -> consumer
   (weights / biases stored as INT8 with explicit dequant on read)
2. Producer -> QuantizeLinear -> DequantizeLinear -> consumer
   (activation re-quantization between layers; potentially chained
    across multiple consecutive layers)
```

Does NOT attempt to handle mixed-precision QLinearConv / QLinearMatMul
fused ops – tf2onnx does not produce these for TFLite input.

If any Q/DQ ops remain after the pass, the script exits non-zero
(unless –no-strict) so unhandled patterns fail loud rather than silent.

## Usage

```
python model_sources/MobileNetV2/scripts/dequantize_onnx.py \\
    --input  model_sources/MobileNetV2/weights/model.onnx \\
    --output model_sources/MobileNetV2/weights/model.fp32.onnx
```

## Exit codes

```
0  success -- output ONNX is Q/DQ free
1  filesystem / load error
2  graph still contains Q/DQ ops after the pass (unhandled pattern)
3  validation of the output ONNX failed
```

“””

from **future** import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import GraphProto, ModelProto, NodeProto, TensorProto, numpy_helper

LOG = logging.getLogger(“dequantize_onnx”)

_QUANT_OP = “QuantizeLinear”
_DEQUANT_OP = “DequantizeLinear”

# —————————————————————————

# Initializer index

# —————————————————————————

@dataclass
class InitializerIndex:
“”“Random-access view over a graph’s initializers.”””

```
by_name: dict[str, TensorProto] = field(default_factory=dict)

@classmethod
def build(cls, graph: GraphProto) -> "InitializerIndex":
    return cls(by_name={init.name: init for init in graph.initializer})

def get_array(self, name: str) -> np.ndarray | None:
    tensor = self.by_name.get(name)
    if tensor is None:
        return None
    return numpy_helper.to_array(tensor)

def add(self, tensor: TensorProto) -> None:
    self.by_name[tensor.name] = tensor

def remove(self, name: str) -> None:
    self.by_name.pop(name, None)
```

# —————————————————————————

# Quantization math

# —————————————————————————

def _dequantize_values(
int_values: np.ndarray,
scale: np.ndarray,
zero_point: np.ndarray,
) -> np.ndarray:
“”“Apply fp32 = scale * (int_value - zero_point) with broadcasting.

```
scale and zero_point may be scalars (per-tensor) or 1-D vectors
(per-channel). NumPy broadcasting handles both correctly when the
shapes are compatible.
"""
return (
    int_values.astype(np.float32) - zero_point.astype(np.float32)
) * scale.astype(np.float32)
```

# —————————————————————————

# Stats

# —————————————————————————

@dataclass
class FoldStats:
dequant_constants_folded: int = 0
quant_dequant_pairs_folded: int = 0
nodes_removed: int = 0
initializers_added: int = 0
initializers_removed: int = 0

# —————————————————————————

# Pass 1: pure DequantizeLinear on constants

# —————————————————————————

def _fold_pure_dequantize(
graph: GraphProto,
inits: InitializerIndex,
stats: FoldStats,
) -> None:
“”“Replace DequantizeLinear(const_int, const_scale, const_zp) with one FP32 constant.”””
nodes_to_remove: list[NodeProto] = []
initializers_to_drop: set[str] = set()

```
for node in graph.node:
    if node.op_type != _DEQUANT_OP:
        continue
    if len(node.input) < 3:
        continue

    x_name, scale_name, zp_name = node.input[0], node.input[1], node.input[2]
    x_arr = inits.get_array(x_name)
    scale_arr = inits.get_array(scale_name)
    zp_arr = inits.get_array(zp_name)

    # All three inputs must be constants for this fold to apply.
    if x_arr is None or scale_arr is None or zp_arr is None:
        continue

    fp32 = _dequantize_values(x_arr, scale_arr, zp_arr)
    out_name = node.output[0]
    new_init = numpy_helper.from_array(fp32, name=out_name)

    inits.add(new_init)
    graph.initializer.append(new_init)
    stats.initializers_added += 1

    nodes_to_remove.append(node)
    initializers_to_drop.update({x_name, scale_name, zp_name})
    stats.dequant_constants_folded += 1

_remove_nodes(graph, nodes_to_remove, stats)
_drop_orphan_initializers(graph, inits, initializers_to_drop, stats)
```

# —————————————————————————

# Pass 2: Q -> DQ activation pairs

# —————————————————————————

def _fold_quantize_then_dequantize(
graph: GraphProto,
inits: InitializerIndex,
stats: FoldStats,
) -> None:
“”“Rewire ‘x -> Quantize -> Dequantize -> consumer’ to ‘x -> consumer’.

```
We discard the quantization grid intentionally: the framework's
PT2E pass re-quantizes on the export side, so carrying TFLite's
quantization through PyTorch is double-work and a source of drift.
"""
producer: dict[str, NodeProto] = {}
for node in graph.node:
    for out in node.output:
        producer[out] = node

nodes_to_remove: list[NodeProto] = []
initializers_to_drop: set[str] = set()
rewires: dict[str, str] = {}  # dq_output -> q_input

for node in graph.node:
    if node.op_type != _DEQUANT_OP:
        continue

    intermediate = node.input[0]
    prev = producer.get(intermediate)
    if prev is None or prev.op_type != _QUANT_OP:
        continue

    passthrough_src = prev.input[0]
    rewires[node.output[0]] = passthrough_src

    nodes_to_remove.extend([prev, node])
    # Drop the scale / zero_point initializers once we confirm
    # they're no longer referenced.
    initializers_to_drop.update(prev.input[1:])
    initializers_to_drop.update(node.input[1:])
    stats.quant_dequant_pairs_folded += 1

_apply_rewires(graph, _resolve_rewire_chains(rewires))
_remove_nodes(graph, nodes_to_remove, stats)
_drop_orphan_initializers(graph, inits, initializers_to_drop, stats)
```

# —————————————————————————

# Graph mutation primitives

# —————————————————————————

def _resolve_rewire_chains(rewires: dict[str, str]) -> dict[str, str]:
“”“Collapse ‘a -> b -> c’ chains into ‘a -> c’.

```
Necessary because chained Q/DQ patterns at consecutive layer
boundaries produce intermediate-tensor rewires whose targets are
themselves about to be rewritten in the same batch. Without this,
_apply_rewires would point consumers at tensor names whose
producers have just been deleted.
"""
resolved: dict[str, str] = {}
for src in rewires:
    target = rewires[src]
    seen = {src}
    # Follow the chain while the target is itself a rewire source.
    # The seen-set guards against pathological cycles, which
    # well-formed ONNX graphs should never produce.
    while target in rewires and target not in seen:
        seen.add(target)
        target = rewires[target]
    resolved[src] = target
return resolved
```

def _apply_rewires(graph: GraphProto, rewires: dict[str, str]) -> None:
“”“Substitute every reference to a rewired tensor with its resolved source.”””
if not rewires:
return
for node in graph.node:
for i, name in enumerate(node.input):
if name in rewires:
node.input[i] = rewires[name]
for value_info in graph.output:
if value_info.name in rewires:
value_info.name = rewires[value_info.name]

def _remove_nodes(
graph: GraphProto,
to_remove: Iterable[NodeProto],
stats: FoldStats,
) -> None:
“”“Remove nodes by identity, preserving relative order of survivors.”””
targets = {id(n) for n in to_remove}
if not targets:
return
keep = [n for n in graph.node if id(n) not in targets]
removed = len(graph.node) - len(keep)
del graph.node[:]
graph.node.extend(keep)
stats.nodes_removed += removed

def _drop_orphan_initializers(
graph: GraphProto,
inits: InitializerIndex,
candidates: Iterable[str],
stats: FoldStats,
) -> None:
“”“Remove initializers that no surviving node still references.”””
candidate_set = set(candidates)
if not candidate_set:
return

```
referenced = {name for node in graph.node for name in node.input}
referenced.update(out.name for out in graph.output)

keep: list[TensorProto] = []
removed_count = 0
for init in graph.initializer:
    if init.name in candidate_set and init.name not in referenced:
        inits.remove(init.name)
        removed_count += 1
        continue
    keep.append(init)

del graph.initializer[:]
graph.initializer.extend(keep)
stats.initializers_removed += removed_count
```

# —————————————————————————

# Top-level pass

# —————————————————————————

def _count_qdq(graph: GraphProto) -> int:
return sum(1 for n in graph.node if n.op_type in (_QUANT_OP, _DEQUANT_OP))

def fold_quantization(model: ModelProto) -> FoldStats:
“”“Run the dequantization passes to a fixed point.

```
Folding a constant DequantizeLinear can expose a previously-blocked
Q -> DQ pair on the next iteration, so we loop until the Q/DQ count
stops decreasing. MAX_ITERS guards against any pathological
no-progress case.
"""
MAX_ITERS = 8

stats = FoldStats()
graph = model.graph
inits = InitializerIndex.build(graph)

last_count = _count_qdq(graph)
for _ in range(MAX_ITERS):
    _fold_pure_dequantize(graph, inits, stats)
    _fold_quantize_then_dequantize(graph, inits, stats)
    current = _count_qdq(graph)
    if current >= last_count:
        break
    last_count = current

return stats
```

def remaining_qdq_nodes(model: ModelProto) -> list[NodeProto]:
return [n for n in model.graph.node if n.op_type in (_QUANT_OP, _DEQUANT_OP)]

# —————————————————————————

# CLI

# —————————————————————————

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
p = argparse.ArgumentParser(
description=“Fold Q/DQ ops in an ONNX graph into FP32 constants.”,
)
p.add_argument(”–input”, type=Path, required=True, help=“Path to the input .onnx file.”)
p.add_argument(”–output”, type=Path, required=True, help=“Path to write the cleaned .onnx file.”)
p.add_argument(
“–strict”,
action=“store_true”,
default=True,
help=“Exit non-zero if any Q/DQ ops remain after the pass (default: on).”,
)
p.add_argument(
“–no-strict”,
dest=“strict”,
action=“store_false”,
help=“Allow leftover Q/DQ ops in the output (warns instead of failing).”,
)
p.add_argument(”–verbose”, action=“store_true”, help=“Enable DEBUG logging.”)
return p.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
args = parse_args(argv)
logging.basicConfig(
level=logging.DEBUG if args.verbose else logging.INFO,
format=”%(levelname)-7s %(message)s”,
)

```
input_path = args.input.expanduser().resolve()
output_path = args.output.expanduser().resolve()

if not input_path.is_file():
    LOG.error("input file not found: %s", input_path)
    return 1

LOG.info("loading: %s", input_path)
try:
    model = onnx.load(str(input_path))
except Exception as e:
    LOG.error("failed to load ONNX: %s", e)
    return 1

n_qdq_before = _count_qdq(model.graph)
LOG.info("Q/DQ nodes before: %d", n_qdq_before)

stats = fold_quantization(model)

leftover = remaining_qdq_nodes(model)
LOG.info("Q/DQ nodes after:  %d", len(leftover))
LOG.info(
    "folded: %d pure-dequantize, %d Q->DQ pairs | "
    "nodes removed: %d | initializers added: %d, removed: %d",
    stats.dequant_constants_folded,
    stats.quant_dequant_pairs_folded,
    stats.nodes_removed,
    stats.initializers_added,
    stats.initializers_removed,
)

if leftover:
    for n in leftover[:5]:
        LOG.warning("leftover %s: name=%r inputs=%s", n.op_type, n.name, list(n.input))
    if len(leftover) > 5:
        LOG.warning("(+%d more)", len(leftover) - 5)
    if args.strict:
        LOG.error(
            "strict mode: %d Q/DQ node(s) remain. These are patterns this "
            "script does not handle; the model would still fail in onnx2torch. "
            "Inspect with Netron and extend the pass if needed.",
            len(leftover),
        )
        return 2

LOG.info("validating output model...")
try:
    onnx.checker.check_model(model)
except onnx.checker.ValidationError as e:
    LOG.error("output model failed validation: %s", e)
    return 3

output_path.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model, str(output_path))
LOG.info("wrote: %s", output_path)
return 0
```

if **name** == “**main**”:
sys.exit(main())