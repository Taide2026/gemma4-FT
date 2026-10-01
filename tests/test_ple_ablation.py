import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ple_ablation import disable_ple, freeze_ple


def make_tiny_language_model():
    config = Gemma4TextConfig(
        vocab_size=32,
        vocab_size_per_layer_input=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        hidden_size_per_layer_input=8,
        max_position_embeddings=64,
    )
    return Gemma4TextModel(config).eval()


class PleAblationTest(unittest.TestCase):
    def test_disable_ple_zeroes_real_decoder_input(self):
        torch.manual_seed(3)
        language_model = make_tiny_language_model()
        wrapper = SimpleNamespace(model=SimpleNamespace(language_model=language_model), config=SimpleNamespace())
        input_ids = torch.tensor([[2, 3, 4]])

        with torch.no_grad():
            with_ple = language_model(input_ids=input_ids, use_cache=False).last_hidden_state
        disable_ple(wrapper)
        with torch.no_grad():
            without_ple = language_model(input_ids=input_ids, use_cache=False).last_hidden_state

        self.assertFalse(torch.allclose(with_ple, without_ple))
        self.assertFalse(wrapper.config.ple_enabled)
        self.assertFalse(language_model.embed_tokens_per_layer.weight.requires_grad)
        self.assertFalse(language_model.layers[0].per_layer_input_gate.weight.requires_grad)

    def test_freeze_ple_keeps_output_and_freezes_only_ple(self):
        torch.manual_seed(3)
        language_model = make_tiny_language_model()
        wrapper = SimpleNamespace(model=SimpleNamespace(language_model=language_model), config=SimpleNamespace())
        input_ids = torch.tensor([[2, 3, 4]])

        with torch.no_grad():
            before = language_model(input_ids=input_ids, use_cache=False).last_hidden_state
        freeze_ple(wrapper)
        with torch.no_grad():
            after = language_model(input_ids=input_ids, use_cache=False).last_hidden_state

        torch.testing.assert_close(before, after)
        self.assertFalse(language_model.embed_tokens_per_layer.weight.requires_grad)
        self.assertFalse(language_model.layers[0].per_layer_projection.weight.requires_grad)
        self.assertTrue(language_model.embed_tokens.weight.requires_grad)
        self.assertTrue(language_model.layers[0].mlp.gate_proj.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
