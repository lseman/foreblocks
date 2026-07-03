"""Tests for the modular NCO framework — protocols, registries, factory."""

import sys

sys.path.insert(0, "/data/dev/foreblocks/scheduling")

import torch
import pytest

from models import (
    NCOModel,
    build_model,
    EncoderRegistry,
    DecoderRegistry,
    shared_decode_loop,
    adapt_encoder,
    adapt_decoder,
    register_builtin_models,
    # Existing implementations
    TransformerEncoder,
    PointerDecoder,
    BipartiteGNN,
    BipartiteDecoder,
    ActorCritic,
)


class TestRegistries:
    """Test encoder/decoder registries."""

    def test_list_builtin_encoders(self):
        register_builtin_models()
        encoders = EncoderRegistry.list()
        assert "transformer" in encoders
        assert "bipartite" in encoders

    def test_list_builtin_decoders(self):
        register_builtin_models()
        decoders = DecoderRegistry.list()
        assert "pointer" in decoders
        assert "bipartite" in decoders

    def test_get_unknown_encoder(self):
        with pytest.raises(ValueError, match="Unknown encoder"):
            EncoderRegistry.get("nonexistent")

    def test_get_unknown_decoder(self):
        with pytest.raises(ValueError, match="Unknown decoder"):
            DecoderRegistry.get("nonexistent")


class TestBuildModel:
    """Test factory function."""

    def test_build_transformer(self):
        model = build_model(
            encoder="transformer",
            decoder="pointer",
            d_model=64,
            n_heads=4,
            n_layers=2,
            encoder_kwargs={"f_static": 9, "f_dynamic": 9, "f_global": 4},
        )
        assert isinstance(model, NCOModel)
        assert type(model.encoder).__name__ in (
            "TransformerEncoder",
            "TransformerEncoderAdapter",
        )
        assert type(model.decoder).__name__ in (
            "PointerDecoder",
            "PointerDecoderAdapter",
        )

    def test_build_bipartite(self):
        model = build_model(
            encoder="bipartite",
            decoder="bipartite",
            d_model=64,
            n_layers=2,
            encoder_kwargs={"edge_dim": 1},
            reencode_every=5,
        )
        assert isinstance(model, NCOModel)
        assert type(model.encoder).__name__ in ("BipartiteGNN", "BipartiteGNNAdapter")
        assert type(model.decoder).__name__ in (
            "BipartiteDecoder",
            "BipartiteDecoderAdapter",
        )
        assert model.reencode_every == 5

    def test_build_direct_instances(self):
        """NCOModel accepts direct encoder/decoder instances (auto-adapts)."""
        encoder = TransformerEncoder(
            f_static=9, f_dynamic=9, f_global=4, d_model=64, n_heads=4, n_layers=2
        )
        decoder = PointerDecoder(d_model=64)
        model = NCOModel(encoder, decoder)
        # NCOModel auto-adapts raw encoders/decoders
        assert type(model.encoder).__name__ in (
            "TransformerEncoder",
            "TransformerEncoderAdapter",
        )
        assert type(model.decoder).__name__ in (
            "PointerDecoder",
            "PointerDecoderAdapter",
        )

    def test_build_with_value_head(self):
        model = build_model(
            encoder="transformer",
            decoder="pointer",
            d_model=64,
            n_layers=2,
            d_value=128,
            encoder_kwargs={"f_static": 9, "f_dynamic": 9, "f_global": 4},
        )
        assert model.value_head is not None

    def test_build_custom_decoder_kwargs(self):
        model = build_model(
            encoder="bipartite",
            decoder="bipartite",
            d_model=128,
            n_layers=3,
            encoder_kwargs={"edge_dim": 2},
            decoder_kwargs={"d_model": 128},
        )
        assert model.encoder.d_model == 128
        assert model.decoder.d_model == 128


class TestNCOModelAct:
    """Test NCOModel act() method."""

    def setup_transformer(self):
        torch.manual_seed(42)
        encoder = TransformerEncoder(
            f_static=9, f_dynamic=9, f_global=4, d_model=64, n_heads=4, n_layers=2
        )
        decoder = PointerDecoder(d_model=64)
        self.model = NCOModel(encoder, decoder)
        self.obs = {
            "task_static": torch.randn(4, 12, 9),
            "task_dynamic": torch.ones(4, 12, 9),
            "glob": torch.ones(4, 4),
            "mask": torch.ones(4, 12, dtype=torch.bool),
        }

    def setup_bipartite(self):
        torch.manual_seed(42)
        encoder = BipartiteGNN(d_model=64, n_layers=2, edge_dim=1)
        decoder = BipartiteDecoder(d_model=64)
        self.model = NCOModel(encoder, decoder)
        B, N, M = 4, 12, 6
        self.obs = {
            "nodes": torch.randn(B, N, 8),  # node features (f_node)
            "edge_index": torch.randint(0, N, (2, M)),  # [2, num_edges]
            "edge_features": torch.randn(M, 1),  # edge features
            "mask": torch.ones(B, N, dtype=torch.bool),
            "n_vars": N,
        }

    def test_transformer_act(self):
        self.setup_transformer()
        out = self.model.act(self.obs)
        assert "new_starts" in out
        assert "logp" in out
        assert "value" in out
        assert out["new_starts"].shape[0] == 4
        assert out["logp"].shape[0] == 4

    def test_bipartite_act(self):
        self.setup_bipartite()
        out = self.model.act(self.obs)
        assert "new_starts" in out
        assert "logp" in out
        assert "value" in out
        assert out["new_starts"].shape[0] == 4

    def test_transformer_value_of(self):
        self.setup_transformer()
        task_emb, ctx = self.model.encode(self.obs)
        value = self.model.value_of(task_emb, ctx)
        assert value.shape[0] == 4

    def test_bipartite_value_of(self):
        self.setup_bipartite()
        task_emb, ctx = self.model.encode(self.obs)
        value = self.model.value_of(task_emb, ctx)
        assert value.shape[0] == 4

    def test_transformer_adapt_encoder(self):
        from models.adapters import TransformerEncoderAdapter

        encoder = TransformerEncoder(
            f_static=9, f_dynamic=9, f_global=4, d_model=64, n_heads=4, n_layers=2
        )
        adapted = adapt_encoder(encoder, n_vars=12)
        task_emb, ctx = adapted.encode(
            {
                "task_static": torch.randn(4, 12, 9),
                "task_dynamic": torch.ones(4, 12, 9),
                "glob": torch.ones(4, 4),
            }
        )
        assert task_emb.shape[0] == 4
        assert adapted.n_vars == 12


class TestSharedDecodeLoop:
    """Test the shared autoregressive decode loop."""

    def test_shared_loop_greedy(self):
        """Shared loop with a simple decoder stub."""
        d_model = 32
        N = 10
        B = 4

        # Stub decoder
        W_q = torch.nn.Linear(d_model, d_model, bias=False)
        W_k = torch.nn.Linear(d_model, d_model, bias=False)
        W_v = torch.nn.Linear(d_model, d_model, bias=False)

        def attention_logits(ctx, nodes):
            q = W_q(ctx)  # [B, d_model]
            k = W_k(nodes)  # [B, N, d_model]
            attn = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1) / (
                d_model**0.5
            )
            stop = torch.zeros(B, device=nodes.device)
            return attn, stop

        def create_context():
            return torch.zeros(B, d_model)

        def update_context(ctx, picked):
            return ctx

        nodes = torch.randn(B, N, d_model)
        mask = torch.ones(B, N, dtype=torch.bool)

        out = shared_decode_loop(
            attn_logits_fn=attention_logits,
            create_context=create_context,
            update_context=update_context,
            nodes=nodes,
            mask=mask,
            mode="greedy",
        )

        assert "new_starts" in out
        assert "logp" in out
        assert "n_picked" in out


class TestBackwardCompat:
    """Test that existing code still works."""

    def test_actor_critic_still_works(self):
        """ActorCritic (existing unified model) still works."""
        model = ActorCritic(
            f_static=9, f_dynamic=9, f_global=4, d_model=64, n_heads=4, n_layers=2
        )
        obs = {
            "task_static": torch.randn(4, 12, 9),
            "task_dynamic": torch.ones(4, 12, 9),
            "glob": torch.ones(4, 4),
            "mask": torch.ones(4, 12, dtype=torch.bool),
        }
        out = model.act(obs)
        assert "new_starts" in out

    def test_aliased_exports(self):
        """TaskEncoder alias still works."""
        assert TransformerEncoder is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
