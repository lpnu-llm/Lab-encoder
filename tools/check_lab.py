"""Instructor checks: fill exercise AST nodes in memory, never change the lab."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from percent_to_ipynb import split_cells
from plot_figures import residual, depth

ROOT = Path(__file__).resolve().parents[1]
SOLUTIONS = '''
def residual_add(x, branch, alpha=1.0):
    if len(x) != len(branch):
        raise ValueError("Shape mismatch")
    return [a + alpha * b for a, b in zip(x, branch)]

def scalar_block(x, w, residual=True, alpha=1.0):
    return x + alpha * w * x if residual else w * x

def block_backward(x, w, upstream, residual=True, alpha=1.0):
    if residual:
        return upstream * (1 + alpha * w), upstream * alpha * x
    return upstream * w, upstream * x

def network_loss_grad(x, weights, target, residual=True, alpha=1.0):
    states = [x]
    for w in weights:
        states.append(scalar_block(states[-1], w, residual, alpha))
    loss = 0.5 * (states[-1] - target) ** 2
    gradient = states[-1] - target
    weight_gradients = [0.0] * len(weights)
    for i in reversed(range(len(weights))):
        gradient, weight_gradients[i] = block_backward(states[i], weights[i], gradient, residual, alpha)
    return loss, gradient, weight_gradients

def depth_gradient(depth, w, residual, alpha=1.0):
    gradient = 1.0
    for _ in range(depth):
        gradient, _ = block_backward(0.0, w, gradient, residual, alpha)
    return gradient

def encoder_residuals(x, attention, ffn, norm1, norm2):
    h = residual_add(x, attention(norm1(x)))
    return residual_add(h, ffn(norm2(h)))

def layer_norm(x, gamma, beta, eps=1e-5):
    if not x or len(x) != len(gamma) or len(x) != len(beta) or eps <= 0:
        raise ValueError("Invalid shape or epsilon")
    mean = sum(x) / len(x)
    variance = sum((v - mean) ** 2 for v in x) / len(x)
    return [g * (v - mean) / (variance + eps) ** 0.5 + b
            for v, g, b in zip(x, gamma, beta)]

def layer_norm_tokens(tokens, gamma, beta, eps=1e-5):
    return [layer_norm(row, gamma, beta, eps) for row in tokens]

def dropout(x, p, training, rng):
    if not 0 <= p < 1:
        raise ValueError("Invalid probability")
    if not training or p == 0:
        return list(x)
    return [0.0 if rng.random() < p else v / (1-p) for v in x]

def encoder_with_dropout(x, attention, ffn, norm1, norm2, p, training, rng):
    h = residual_add(x, dropout(attention(norm1(x)), p, training, rng))
    return residual_add(h, dropout(ffn(norm2(h)), p, training, rng))

def rms_norm(x, gamma, eps=1e-5):
    if not x or len(x) != len(gamma) or eps <= 0:
        raise ValueError("Invalid shape or epsilon")
    denominator = (sum(v * v for v in x) / len(x) + eps) ** 0.5
    return [g * v / denominator for v, g in zip(x, gamma)]

def rms_norm_tokens(tokens, gamma, eps=1e-5):
    return [rms_norm(row, gamma, eps) for row in tokens]
'''


def main():
    source = (ROOT / "lab_encoder.py").read_text(encoding="utf-8")
    notebook = json.loads((ROOT / "lab_encoder.ipynb").read_text(encoding="utf-8"))
    assert notebook["cells"] == split_cells(source), "Regenerate notebook"
    solutions = {node.name: node for node in ast.parse(SOLUTIONS).body}
    module = ast.parse(source)
    replaced = set()
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in solutions:
            assert any(isinstance(child, ast.Constant) and child.value is Ellipsis for child in ast.walk(node))
            node.body = solutions[node.name].body
            replaced.add(node.name)
    assert replaced == set(solutions)
    module = ast.fix_missing_locations(module)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(module, str(ROOT / "lab_encoder.py"), "exec"), {"__name__": "__main__"})
    print(output.getvalue(), end="")
    # SVGs must be parseable and match their deterministic generators.
    for name, expected in [("residual", residual()), ("depth", depth())]:
        content = (ROOT / f"figures/{name}.svg").read_text(encoding="utf-8")
        ET.fromstring(content)
        assert content == expected
    # Every code cell is syntactically valid before students fill it in.
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<notebook cell>", "exec")
    print(f"✓ {len(solutions)} вправ; notebook і SVG відповідають джерелам")


if __name__ == "__main__":
    main()
