import pytest
import torch

from tinygpt.model import GPT, GPTConfig, apply_rope, rope_tables

POS = ["rope", "learned"]


def small(**kw):
    cfg = dict(vocab_size=64, block_size=32, n_layer=2, n_head=4, n_embd=32)
    cfg.update(kw)
    return GPTConfig(**cfg)


@pytest.mark.parametrize("pos_emb", POS)
def test_shapes(pos_emb):
    torch.manual_seed(0)
    model = GPT(small(pos_emb=pos_emb))
    idx = torch.randint(0, 64, (3, 20))
    logits, loss = model(idx)
    assert logits.shape == (3, 20, 64)
    assert loss is None
    _, loss = model(idx, idx)
    assert loss.shape == ()
    with pytest.raises(AssertionError):
        model(torch.zeros(1, 33, dtype=torch.long))


def test_untrained_loss_is_near_uniform():
    # Small initial weights give nearly flat logits, so the loss starts at
    # about ln(vocab_size). A much higher start means a bad init.
    torch.manual_seed(0)
    model = GPT(small(vocab_size=4096, n_embd=64))
    # Targets independent of the inputs. With tied weights, predicting the
    # input token itself scores well even untrained (its embedding is still in
    # the residual stream), so idx as its own target would come out lower.
    idx, targets = torch.randint(0, 4096, (2, 4, 32))
    _, loss = model(idx, targets)
    assert abs(loss.item() - torch.log(torch.tensor(4096.0)).item()) < 0.1


def test_output_layer_shares_the_embedding():
    model = GPT(small())
    assert model.lm_head.weight is model.tok_emb.weight


@pytest.mark.parametrize("pos_emb", POS)
def test_parameter_count(pos_emb):
    # Per block: qkv 3C^2, attention output C^2, MLP 8C^2, two layer norms 4C.
    # Plus the embedding (counted once, it is shared with the output layer),
    # the final layer norm, and a learned position table if there is one.
    V, C, L, T = 4096, 384, 6, 256
    model = GPT(GPTConfig(vocab_size=V, n_embd=C, n_layer=L, n_head=6, block_size=T, pos_emb=pos_emb))
    expected = V * C + L * (12 * C * C + 4 * C) + 2 * C + (T * C if pos_emb == "learned" else 0)
    assert model.num_params() == expected
    assert model.num_params() == {"rope": 12_199_680, "learned": 12_297_984}[pos_emb]  # the README's figures


def test_residual_projections_start_smaller():
    # The outputs that add into the residual stream start at 0.02 / sqrt(2 x
    # layers); everything else at 0.02.
    torch.manual_seed(0)
    model = GPT(small(n_embd=128, n_layer=8))
    for name, p in model.named_parameters():
        if p.dim() < 2:
            continue
        expected = 0.02 / 4 if name.endswith("proj.weight") else 0.02
        assert p.std().item() == pytest.approx(expected, rel=0.1), name


@pytest.mark.parametrize("pos_emb", POS)
@pytest.mark.parametrize("attn", ["sdpa", "matmul"])
def test_future_tokens_do_not_change_past_logits(pos_emb, attn):
    torch.manual_seed(0)
    model = GPT(small(pos_emb=pos_emb, attn=attn)).eval()
    idx = torch.randint(0, 64, (2, 32))
    for t in [1, 7, 31]:
        changed = idx.clone()
        changed[:, t:] = torch.randint(0, 64, (2, 32 - t))
        assert not torch.equal(changed[:, t:], idx[:, t:])
        a, _ = model(idx)
        b, _ = model(changed)
        assert torch.allclose(a[:, :t], b[:, :t], atol=1e-6, rtol=0)
        assert not torch.allclose(a[:, t:], b[:, t:])


def test_past_tokens_do_change_future_logits():
    # The other half of the mask test: a model that ignored context entirely
    # would also pass the test above.
    torch.manual_seed(0)
    model = GPT(small()).eval()
    idx = torch.randint(0, 64, (1, 32))
    changed = idx.clone()
    changed[0, 0] = (idx[0, 0] + 1) % 64
    a, _ = model(idx)
    b, _ = model(changed)
    assert not torch.allclose(a[0, -1], b[0, -1])


@pytest.mark.parametrize("pos_emb", POS)
def test_sdpa_matches_matmul_attention(pos_emb):
    torch.manual_seed(0)
    a = GPT(small(pos_emb=pos_emb, attn="sdpa")).eval()
    b = GPT(small(pos_emb=pos_emb, attn="matmul")).eval()
    b.load_state_dict(a.state_dict())
    idx = torch.randint(0, 64, (2, 32))
    assert torch.allclose(a(idx)[0], b(idx)[0], atol=1e-5)


def test_rope_scores_depend_only_on_distance():
    torch.manual_seed(0)
    cos, sin = rope_tables(16, 64)
    q, k = torch.randn(16), torch.randn(16)

    def score(m, n):
        qm = apply_rope(q, cos[m], sin[m])
        kn = apply_rope(k, cos[n], sin[n])
        return (qm * kn).sum()

    assert torch.allclose(score(10, 3), score(50, 43), atol=1e-5)
    assert torch.allclose(score(0, 0), (q * k).sum(), atol=1e-6)
    assert not torch.allclose(score(10, 3), score(10, 4))


@pytest.mark.parametrize("pos_emb", POS)
def test_model_uses_token_order(pos_emb):
    # Without position information, one layer of causal attention sees the
    # tokens before the last one as an unordered set: swapping tokens 0 and 1
    # would leave the last logits unchanged. (With more layers the causal mask
    # alone leaks some order, so this uses one layer.) fp64, so that "unchanged"
    # means rounding error and the check can tell it from a real difference.
    torch.manual_seed(0)
    model = GPT(small(pos_emb=pos_emb, n_layer=1)).double().eval()
    idx = torch.randperm(64)[:32].unsqueeze(0)
    swapped = idx.clone()
    swapped[0, [0, 1]] = idx[0, [1, 0]]

    def change():
        return (model(idx)[0][0, -1] - model(swapped)[0][0, -1]).abs().max().item()

    assert change() > 1e-6
    # The control: switch the positions off and the swap no longer matters.
    with torch.no_grad():
        if pos_emb == "rope":
            model.rope_cos.fill_(1.0)
            model.rope_sin.zero_()
        else:
            model.pos_emb.weight.zero_()
    assert change() < 1e-12


@pytest.mark.parametrize("pos_emb", POS)
def test_memorizes_one_sequence(pos_emb):
    # A permutation, so each token has exactly one successor and greedy
    # decoding from the first token should replay the whole sequence. This
    # checks that the model can learn at all; a model without positions could
    # also pass it, which is what test_model_uses_token_order is for.
    torch.manual_seed(0)
    model = GPT(small(pos_emb=pos_emb))
    seq = torch.randperm(64)[:33].unsqueeze(0)
    x, y = seq[:, :-1], seq[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    out = model.eval().generate(seq[:, :1], max_new_tokens=32, temperature=0)
    assert torch.equal(out, seq)


class CountingGPT(GPT):
    """Always predicts the previous token plus one, so greedy output is known."""

    def forward(self, idx, targets=None):
        nxt = (idx + 1) % self.cfg.vocab_size
        return torch.nn.functional.one_hot(nxt, self.cfg.vocab_size).float(), None


def test_generate_stops_once_every_row_has_stopped():
    model = CountingGPT(small()).eval()
    start = torch.tensor([[0], [5]])
    # Row 1 reaches 7 after two tokens and row 0 after seven, so generation
    # has to keep going for row 0 and then stop.
    out = model.generate(start, max_new_tokens=20, temperature=0, stop_token=7)
    assert out.tolist() == [list(range(0, 8)), list(range(5, 13))]


def test_generate_without_stop_token_runs_to_the_limit():
    model = CountingGPT(small()).eval()
    out = model.generate(torch.tensor([[0]]), max_new_tokens=10, temperature=0)
    assert out.tolist() == [list(range(11))]


def test_generate_runs_past_the_block_size():
    torch.manual_seed(0)
    model = GPT(small(block_size=8)).eval()
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=20, top_k=5)
    assert out.shape == (1, 21)
