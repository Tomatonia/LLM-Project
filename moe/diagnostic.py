"""
Compare MoE model against dense baseline on a single prompt.
Run on the training server:  python moe/1.py
"""
import os, sys, json, torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from moe.moe_layer import convert_to_moe

MODEL_DIR = "moe/qwen_moe_init"
with open(os.path.join(MODEL_DIR, "moe_config.json")) as f:
    cfg = json.load(f)

print("Config:")
for k, v in cfg.items():
    print(f"  {k}: {v}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
prompt = "Problem: What is 2+2?\nSolution:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")


def load_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()


ref = load_model()                      # dense baseline
moe = load_model()
convert_to_moe(moe, cfg)                # apply shared+MoE

# ─── 1. Full-model logit comparison ─────────────────────────
with torch.no_grad():
    out_moe = moe(**inputs)
    out_ref = ref(**inputs)

diff = (out_moe.logits - out_ref.logits).abs().mean().item()
scale = out_ref.logits.abs().mean().item()
print(f"\nFull model — |logit diff|: {diff:.4f}  scale: {scale:.4f}  "
      f"rel: {diff / scale * 100:.2f}%")

top5_moe = [tokenizer.decode([t]) for t in out_moe.logits[0, -1].topk(5).indices.tolist()]
top5_ref = [tokenizer.decode([t]) for t in out_ref.logits[0, -1].topk(5).indices.tolist()]
print(f"Top-5 MoE: {top5_moe}")
print(f"Top-5 ref: {top5_ref}")

# ─── 2. Per-MoE-layer MLP comparison ────────────────────────
mlp_inputs = {}

def make_hook(i):
    def hook(module, input, output):
        mlp_inputs[i] = input[0].detach().clone()
    return hook

hooks = []
for i in cfg["moe_layers"]:
    hooks.append(ref.model.layers[i].mlp.register_forward_hook(make_hook(i)))

with torch.no_grad():
    ref(**inputs)
for h in hooks:
    h.remove()

print("\nPer-layer MLP (same input → dense FFN vs MoE):")
for i in cfg["moe_layers"]:
    x = mlp_inputs[i]
    with torch.no_grad():
        dense_out = ref.model.layers[i].mlp(x)
        moe_out = moe.model.layers[i].mlp(x)
    d = (moe_out - dense_out).abs().mean().item()
    s = dense_out.abs().mean().item()
    print(f"  Layer {i:2d}: diff={d:.4f}  scale={s:.4f}  rel={d / s * 100:.1f}%")

# ─── 3. Expert weight check ─────────────────────────────────
print("\nExpert weight check (first MoE layer):")
layer_idx = cfg["moe_layers"][0]
block = moe.model.layers[layer_idx].mlp

# Shared expert
dense_gate = ref.model.layers[layer_idx].mlp.gate_proj.weight
shared_gate = block.shared.gate_proj.weight
s_is = shared_gate.shape[0]
md = (shared_gate - dense_gate[:s_is]).abs().max().item()
print(f"  Shared ({s_is} dims): max|diff|={md:.6f}  mean|w|={shared_gate.abs().mean().item():.4f}")

# Routed experts
for e in range(len(block.routed)):
    r_gate = block.routed[e].gate_proj.weight
    r_is = r_gate.shape[0]
    print(f"  Routed[{e}] ({r_is} dims): mean|w|={r_gate.abs().mean().item():.4f}  "
          f"nonzero={(r_gate.abs().sum(dim=1) > 0).sum().item()}/{r_is}")

# ─── 4. Router & dispatch check ─────────────────────────────
print("\nRouter / dispatch (first MoE layer):")
gate_w = block.router.gate.weight
print(f"  gate: shape={list(gate_w.shape)}  mean={gate_w.mean().item():.6f}  "
      f"std={gate_w.std().item():.6f}")

with torch.no_grad():
    dm, cw, rp, df = block.router(mlp_inputs[layer_idx].reshape(-1, cfg["hidden_size"]))
    for e in range(cfg["num_experts"]):
        n = dm[:, e, :].any(dim=-1).sum().item()
        print(f"  Routed expert {e}: {n} tokens  (capacity={dm.shape[-1]})")
    print(f"  Dropped: {df:.4f}")
    print(f"  Router prob range: [{rp.min().item():.4f}, {rp.max().item():.4f}]")

# ─── 5. Generate a short completion ─────────────────────────
print("\nGeneration test (MoE, greedy):")
moe.eval()
gen = moe.generate(**inputs, max_new_tokens=64, do_sample=False,
                   pad_token_id=tokenizer.pad_token_id,
                   eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(gen[0], skip_special_tokens=True))
