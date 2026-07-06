"""Dump the dtype structure of an ONNX graph — diagnose fp16-conversion issues.

    python3 scripts/inspect_onnx.py models/depth_v2_small_fp16.onnx

Shows graph I/O dtypes, an initializer dtype histogram, Cast-node count, the
first handful of nodes (op + I/O), and the first Conv's activation-input vs
weight dtype — enough to see exactly why a strongly-typed TensorRT parse rejects
the graph (e.g. an fp32 activation flowing into an fp16-weight conv).
"""

import sys
from collections import Counter

import onnx
from onnx import TensorProto

_DT = {TensorProto.FLOAT: "fp32", TensorProto.FLOAT16: "fp16",
       TensorProto.INT64: "i64", TensorProto.INT32: "i32",
       TensorProto.BOOL: "bool", TensorProto.DOUBLE: "f64"}


def _name(e):
    return _DT.get(e, f"dt{e}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "models/depth_v2_small_fp16.onnx"
    g = onnx.load(path).graph

    print(f"== {path} ==")
    print("INPUTS:")
    for i in g.input:
        print(f"   {i.name:40s} {_name(i.type.tensor_type.elem_type)}")
    print("OUTPUTS:")
    for o in g.output:
        print(f"   {o.name:40s} {_name(o.type.tensor_type.elem_type)}")

    hist = Counter(_name(init.data_type) for init in g.initializer)
    print(f"INITIALIZERS: {dict(hist)}")

    casts = [n for n in g.node if n.op_type == "Cast"]
    print(f"CAST nodes: {len(casts)}")
    for n in casts[:4]:
        to = next((a.i for a in n.attribute if a.name == "to"), None)
        print(f"   cast {list(n.input)} -> {list(n.output)}  to={_name(to)}")

    print("FIRST 6 NODES:")
    for k, n in enumerate(g.node[:6]):
        print(f"   [{k}] {n.op_type:12s} in={list(n.input)} out={list(n.output)}")

    inits = {init.name: init for init in g.initializer}
    for n in g.node:
        if n.op_type == "Conv":
            w = n.input[1] if len(n.input) > 1 else None
            wd = _name(inits[w].data_type) if w in inits else "??(not-initializer)"
            print(f"FIRST CONV: {n.name}\n   act-in={n.input[0]}  weight={w} ({wd})")
            break


if __name__ == "__main__":
    main()
