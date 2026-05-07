def replace_edgetpu_ops(m):
    """Replace TFL_edgetpu-custom-op nodes with native ONNX ops.

    The EdgeTPU compiler bundles small subgraphs into custom ops carrying
    binary TPU bytecode that ONNX runtime can't validate. Heuristically map
    them to native ops based on node name.

    Multi-output Squeeze ops are reconstructed as Squeeze -> Unsqueeze chain:
        output[0] = Squeeze(input, axes=[0])
        output[1] = Unsqueeze(output[0], axes=[0])  # restores input shape
    Other multi-output ops fall back to Identity aliases on output[0].
    The parity check is your validation gate."""
    NAME_TO_OP = [
        ("squeeze",   "Squeeze"),
        ("transpose", "Transpose"),
        ("reshape",   "Reshape"),
    ]
    SQUEEZE_AXES = [0]  # heuristic for SSD batch-strip; adjust if parity fails

    new_nodes = []
    new_initializers = []
    replaced = 0

    for node in m.graph.node:
        if not node.op_type.startswith("TFL_"):
            new_nodes.append(node)
            continue

        new_op = "Identity"
        for pat, op in NAME_TO_OP:
            if pat in node.name.lower():
                new_op = op
                break

        n_outputs = len(node.output)
        log.info("  replacing %s (was %s, %d outputs) -> %s",
                 node.name, node.op_type, n_outputs, new_op)

        if n_outputs == 1:
            node.op_type = new_op
            node.domain = ""
            del node.attribute[:]
            new_nodes.append(node)

        elif new_op == "Squeeze" and n_outputs == 2:
            # Squeeze(input) -> output[0], then Unsqueeze(output[0]) -> output[1]
            axes_name = f"{node.name}_axes"
            new_initializers.append(onnx.helper.make_tensor(
                name=axes_name,
                data_type=TensorProto.INT64,
                dims=[len(SQUEEZE_AXES)],
                vals=SQUEEZE_AXES,
            ))
            new_nodes.append(onnx.helper.make_node(
                "Squeeze",
                inputs=[node.input[0], axes_name],
                outputs=[node.output[0]],
                name=node.name,
            ))
            new_nodes.append(onnx.helper.make_node(
                "Unsqueeze",
                inputs=[node.output[0], axes_name],
                outputs=[node.output[1]],
                name=f"{node.name}_unsqueeze",
            ))
            log.info("    output[0] = Squeeze(input, axes=%s)", SQUEEZE_AXES)
            log.info("    output[1] = Unsqueeze(output[0], axes=%s) "
                     "[shape-restoring hedge]", SQUEEZE_AXES)

        else:
            # Fallback: primary op + Identity aliases
            primary = onnx.helper.make_node(
                new_op,
                inputs=list(node.input),
                outputs=[node.output[0]],
                name=node.name,
            )
            new_nodes.append(primary)
            for i, extra_out in enumerate(node.output[1:], start=1):
                new_nodes.append(onnx.helper.make_node(
                    "Identity",
                    inputs=[node.output[0]],
                    outputs=[extra_out],
                    name=f"{node.name}_alias_{i}",
                ))

        replaced += 1

    if replaced:
        del m.graph.node[:]
        m.graph.node.extend(new_nodes)
        m.graph.initializer.extend(new_initializers)
        log.warning("  replaced %d EdgeTPU op(s) - parity check is your "
                    "validation gate", replaced)
    return m
