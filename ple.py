"""
How to call Gemma 4 Per-Layer Embeddings (PLE).

Ref: https://sebastianraschka.com/llm-architecture-gallery/per-layer-embeddings/
Module names follow transformers 5.x `models/gemma4/modeling_gemma4.py`.

Pipeline (E4B: num_hidden_layers=42):
  1. token-identity:  embed_tokens_per_layer(input_ids)            -> [B, T, L, D_ple]
  2. context:         per_layer_model_projection(inputs_embeds)    -> [B, T, L, D_ple]
  3. combine:         (context + token_identity) * 2**-0.5
  4. in layer i:      h = h + norm(proj(act(gate(h)) * ple[:, :, i]))
"""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration


MODEL_ID = "google/gemma-4-E4B-it"

model = Gemma4ForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model.eval()

lm = model.model.language_model  # Gemma4TextModel
cfg = lm.config
print("num_hidden_layers           =", cfg.num_hidden_layers)
print("hidden_size                 =", cfg.hidden_size)
print("hidden_size_per_layer_input =", cfg.hidden_size_per_layer_input)

input_ids = tokenizer("a child sleds down a snowy hill", return_tensors="pt").input_ids


# ============================================================
# 1. Built-in API
# ============================================================
with torch.no_grad():
    inputs_embeds = lm.embed_tokens(input_ids)                     # [B, T, H]
    token_ple = lm.get_per_layer_inputs(input_ids, inputs_embeds)  # [B, T, L, D_ple]
    ple = lm.project_per_layer_inputs(inputs_embeds, token_ple)    # [B, T, L, D_ple]

print("inputs_embeds :", tuple(inputs_embeds.shape))
print("token_ple     :", tuple(token_ple.shape))
print("ple           :", tuple(ple.shape))


# ============================================================
# 2. Same thing, step by step
# ============================================================
with torch.no_grad():
    B, T = input_ids.shape
    L, D = cfg.num_hidden_layers, cfg.hidden_size_per_layer_input

    # token-identity part; embed_tokens_per_layer already scales by sqrt(D)
    manual_token_ple = lm.embed_tokens_per_layer(input_ids).reshape(B, T, L, D)

    # context part
    ctx = lm.per_layer_model_projection(inputs_embeds) * lm.per_layer_model_projection_scale
    ctx = lm.per_layer_projection_norm(ctx.reshape(B, T, L, D))

    manual_ple = (ctx + manual_token_ple) * lm.per_layer_input_scale

torch.testing.assert_close(manual_ple, ple)
print("manual PLE matches project_per_layer_inputs()")


# ============================================================
# 3. How decoder layer i consumes its PLE slice
#    (the extra residual update at the end of Gemma4TextDecoderLayer.forward)
# ============================================================
def apply_ple_branch(layer, hidden_states, per_layer_input):
    residual = hidden_states
    x = layer.per_layer_input_gate(hidden_states)  # H -> D_ple
    x = layer.act_fn(x)
    x = x * per_layer_input                         # gate the PLE vector
    x = layer.per_layer_projection(x)               # D_ple -> H
    x = layer.post_per_layer_input_norm(x)
    return residual + x


layer_idx = 0
layer = lm.layers[layer_idx]
with torch.no_grad():
    out = apply_ple_branch(layer, inputs_embeds, ple[:, :, layer_idx, :])
print(f"layer {layer_idx} PLE branch output:", tuple(out.shape))


# ============================================================
# 4. Passing precomputed PLE through the full model
#    (per_layer_inputs is only allowed with inputs_embeds, not input_ids)
# ============================================================
with torch.no_grad():
    outputs = lm(inputs_embeds=inputs_embeds, per_layer_inputs=token_ple)
print("last_hidden_state:", tuple(outputs.last_hidden_state.shape))
