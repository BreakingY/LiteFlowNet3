import onnx

import sys
from pathlib import Path


src_path = sys.argv[1]
p = Path(src_path)
dst_path = p.with_stem(p.stem + "_fixed")

model = onnx.load(src_path, load_external_data=True)


# ============================================================
# 1. 建立 Identity 映射
# ============================================================

identity_map = {}

for node in model.graph.node:

    if node.op_type != "Identity":
        continue

    if len(node.input) != 1 or len(node.output) != 1:
        continue

    identity_map[node.output[0]] = node.input[0]


# ============================================================
# 2. 获取 initializer
# ============================================================

initializer_names = {
    initializer.name
    for initializer in model.graph.initializer
}


# ============================================================
# 3. 递归追溯 Identity，找到真正的 initializer
# ============================================================

def resolve_initializer(name):

    visited = set()

    while name not in initializer_names:

        if name in visited:
            return None

        visited.add(name)

        source = identity_map.get(name)

        if source is None:
            return None

        name = source

    return name


print("=" * 80)
print("ONNX TRT 8.4 Compatibility Fix")
print("=" * 80)


# ============================================================
# 4. Fix ConvTranspose bias
# ============================================================

base_bias = "model.netTwo.netUpflow.bias"

count_convtranspose = 0

print()
print("=" * 80)
print("Fix ConvTranspose bias")
print("=" * 80)


for node in model.graph.node:

    if node.op_type != "ConvTranspose":
        continue

    if len(node.input) < 3:
        continue

    old_bias = node.input[2]

    source_bias = resolve_initializer(old_bias)

    if source_bias == base_bias:

        print()
        print("Node:", node.name)
        print("  old bias:", old_bias)
        print("  new bias:", source_bias)

        node.input[2] = source_bias

        count_convtranspose += 1


# ============================================================
# 5. Fix Conv / ConvTranspose weight
# ============================================================

count_weight = 0

print()
print("=" * 80)
print("Fix Conv / ConvTranspose weight")
print("=" * 80)


for node in model.graph.node:

    if node.op_type not in ("Conv", "ConvTranspose"):
        continue

    if len(node.input) < 2:
        continue

    old_weight = node.input[1]

    if old_weight in initializer_names:
        continue

    source_weight = resolve_initializer(old_weight)

    if source_weight is None:
        continue

    print()
    print("Node:", node.name)
    print("  op          :", node.op_type)
    print("  old weight  :", old_weight)
    print("  source      :", source_weight)
    print("  status      : Identity -> initializer")
    print("  fix         :", old_weight, "=>", source_weight)

    node.input[1] = source_weight

    count_weight += 1


# ============================================================
# 6. Fix Conv bias
# ============================================================

count_conv = 0

print()
print("=" * 80)
print("Fix Conv bias")
print("=" * 80)


for node in model.graph.node:

    if node.op_type != "Conv":
        continue

    if len(node.input) < 3:
        continue

    old_bias = node.input[2]

    if old_bias in initializer_names:
        continue

    source_bias = resolve_initializer(old_bias)

    if source_bias is None:
        continue

    print()
    print("Node:", node.name)
    print("  old bias:", old_bias)
    print("  source  :", source_bias)
    print("  status  : Identity -> initializer")
    print("  fix     :", old_bias, "=>", source_bias)

    node.input[2] = source_bias

    count_conv += 1


# ============================================================
# 7. 检查所有 Conv / ConvTranspose weight 和 bias
# ============================================================

print()
print("=" * 80)
print("Final Conv / ConvTranspose check")
print("=" * 80)


bad_weights = []
bad_biases = []


for node in model.graph.node:

    if node.op_type not in ("Conv", "ConvTranspose"):
        continue

    # 检查 weight
    if len(node.input) >= 2:

        weight = node.input[1]

        if weight not in initializer_names:

            bad_weights.append(
                (node.op_type, node.name, weight)
            )

            print()
            print("[WARNING] Weight is NOT initializer")
            print("  op     :", node.op_type)
            print("  node   :", node.name)
            print("  weight :", weight)

    # 检查 bias
    if len(node.input) >= 3:

        bias = node.input[2]

        if bias not in initializer_names:

            bad_biases.append(
                (node.op_type, node.name, bias)
            )

            print()
            print("[WARNING] Bias is NOT initializer")
            print("  op    :", node.op_type)
            print("  node  :", node.name)
            print("  bias  :", bias)


# ============================================================
# 8. 删除已经不再使用的 Identity
# ============================================================

print()
print("=" * 80)
print("Remove unused Identity")
print("=" * 80)


used_inputs = set()

for node in model.graph.node:
    for inp in node.input:
        used_inputs.add(inp)


# graph output 也属于有效引用
graph_outputs = {
    output.name for output in model.graph.output
}

remove_identity_nodes = []

for node in model.graph.node:

    if node.op_type != "Identity":
        continue

    if len(node.output) != 1:
        continue

    output_name = node.output[0]

    if output_name not in used_inputs and output_name not in graph_outputs:
        remove_identity_nodes.append(node)


for node in remove_identity_nodes:

    print()
    print("Remove Identity:")
    print("  node  :", node.name)
    print("  input :", node.input[0])
    print("  output:", node.output[0])

    model.graph.node.remove(node)


# ============================================================
# 9. 保存
# ============================================================

print()
print("=" * 80)
print("结果")
print("=" * 80)

print("修改 ConvTranspose bias 数量:", count_convtranspose)
print("修改 Conv / ConvTranspose weight 数量:", count_weight)
print("修改 Conv bias 数量:", count_conv)

print("剩余非 initializer weight:", len(bad_weights))
print("剩余非 initializer bias  :", len(bad_biases))


if bad_weights or bad_biases:

    print()
    print("!!! 仍然存在非 initializer 的 weight 或 bias")

else:

    print()
    print("所有 Conv / ConvTranspose weight 和 bias 都是 initializer")


onnx.save(model, dst_path)

print()
print("保存到:", dst_path)
print("=" * 80)