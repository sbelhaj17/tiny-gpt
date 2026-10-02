import json
import time

import numpy as np
import pytest
import torch

import train
from tinygpt.data import Batches
from train import TrainConfig, lr_at

VOCAB = 64


@pytest.fixture
def data_dir(tmp_path):
    # The same permutation over and over: easy to learn, so the loss has
    # somewhere to go in a few dozen steps.
    rng = np.random.default_rng(0)
    perm = rng.permutation(VOCAB).astype(np.uint16)
    np.tile(perm, 200).tofile(tmp_path / "train.bin")
    np.tile(perm, 20).tofile(tmp_path / "val.bin")
    meta = {"vocab_size": VOCAB, "eot": VOCAB - 1, "val": {"tokens": 1280, "bytes": 5120}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path


def tiny_args(data_dir, out_dir, **kw):
    args = dict(
        data_dir=data_dir, out_dir=out_dir, device="cpu", dtype="fp32", optimizer="foreach",
        n_layer=1, n_head=2, n_embd=16, block_size=16, batch_size=4,
        lr=1e-2, min_lr=1e-3, warmup_steps=2, max_steps=8,
        log_interval=2, eval_interval=4, eval_batches=2,
    )
    args.update(kw)
    return [f"--{k}={v}" for k, v in args.items()]


def read_log(out_dir):
    with open(out_dir / "log.jsonl") as f:
        return [json.loads(line) for line in f]


def test_lr_schedule():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=1000)
    assert lr_at(0, cfg) == pytest.approx(1e-5)
    assert lr_at(99, cfg) == pytest.approx(1e-3)
    assert lr_at(550, cfg) == pytest.approx(5.5e-4)  # halfway down the cosine
    assert lr_at(1000, cfg) == pytest.approx(1e-4)
    after = [lr_at(s, cfg) for s in range(100, 1001)]
    assert all(a >= b for a, b in zip(after, after[1:]))


def test_batches_are_shifted_windows(data_dir):
    b = Batches(data_dir / "train.bin", batch_size=8, block_size=16, device="cpu")
    tokens = np.fromfile(data_dir / "train.bin", dtype=np.uint16)
    starts = b.random_starts()
    x, y = b.get(starts)
    assert x.shape == y.shape == (8, 16)
    assert x.dtype == torch.long
    for row, s in enumerate(starts):
        assert x[row].tolist() == tokens[s : s + 16].tolist()
        assert y[row].tolist() == tokens[s + 1 : s + 17].tolist()


def test_validation_windows_are_fixed(data_dir):
    a = Batches(data_dir / "val.bin", 4, 16, "cpu", seed=1).fixed_starts(3)
    b = Batches(data_dir / "val.bin", 4, 16, "cpu", seed=2).fixed_starts(3)
    assert all((s == t).all() for s, t in zip(a, b))


def test_training_lowers_validation_loss(data_dir, tmp_path):
    out = tmp_path / "run"
    train.main(tiny_args(data_dir, out, n_embd=32, max_steps=100, eval_interval=100))
    evals = [r for r in read_log(out) if "val_loss" in r]
    assert [r["step"] for r in evals] == [0, 100]
    assert evals[0]["val_loss"] > 3.5  # about ln(64) = 4.16
    assert evals[1]["val_loss"] < 1.0
    ckpt = torch.load(out / "ckpt.pt", weights_only=False)
    assert ckpt["step"] == 100


def test_resume_matches_an_uninterrupted_run(data_dir, tmp_path):
    # Constant learning rate, so stopping at step 4 does not change the
    # schedule. Restoring the model, the optimizer and the data sampler's
    # position must then give exactly the weights of a run that never stopped.
    const = dict(lr=1e-2, min_lr=1e-2, warmup_steps=0)
    train.main(tiny_args(data_dir, tmp_path / "a", max_steps=8, **const))
    train.main(tiny_args(data_dir, tmp_path / "b", max_steps=4, **const))
    train.main(tiny_args(data_dir, tmp_path / "b", max_steps=8, resume=True, **const))

    a = torch.load(tmp_path / "a" / "ckpt.pt", weights_only=False)
    b = torch.load(tmp_path / "b" / "ckpt.pt", weights_only=False)
    assert a["step"] == b["step"] == 8
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k
    assert a["val_loss"] == b["val_loss"]


def test_logged_time_and_speed_count_only_training(data_dir, tmp_path, monkeypatch):
    # A fake clock: each training step takes 0.25 s and each evaluation 100 s.
    # Evaluations must not count toward training time or speed.
    now = [0.0]
    real_step, real_eval = train.train_step, train.evaluate

    def step(*args):
        now[0] += 0.25
        return real_step(*args)

    def evaluate(*args):
        now[0] += 100.0
        return real_eval(*args)

    monkeypatch.setattr(train, "clock", lambda: now[0])
    monkeypatch.setattr(train, "train_step", step)
    monkeypatch.setattr(train, "evaluate", evaluate)
    out = tmp_path / "run"
    train.main(tiny_args(data_dir, out, max_steps=8))
    recs = read_log(out)
    assert [r["step"] for r in recs if "tok_per_s" in r] == [2, 4, 6, 8]
    tokens_per_step = 4 * 16  # batch_size x block_size in tiny_args
    for r in recs:
        assert r["elapsed"] == pytest.approx(0.25 * r["step"])
        if "tok_per_s" in r:
            assert r["tok_per_s"] == pytest.approx(tokens_per_step / 0.25)


def test_full_validation_covers_every_window_once(data_dir):
    from evaluate import full_loss
    from tinygpt.model import GPT, GPTConfig

    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=VOCAB, block_size=16, n_layer=1, n_head=2, n_embd=16)).eval()
    tokens = np.fromfile(data_dir / "val.bin", dtype=np.uint16)[: 16 * 7 + 5]
    # batch size 3 does not divide the 7 windows, so the last batch is short
    loss, count = full_loss(model, tokens, block_size=16, batch_size=3, device="cpu")
    assert count == 7 * 16
    by_hand = []
    for i in range(7):
        w = torch.from_numpy(tokens[i * 16 : i * 16 + 17].astype(np.int64))[None]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            by_hand.append(model(w[:, :-1], w[:, 1:])[1].item())
    assert loss == pytest.approx(np.mean(by_hand), rel=1e-6)


def test_training_clock_is_monotonic():
    # time.time() is the wall clock: it can be set back or forward, and it
    # kept counting while the laptop slept with its lid closed. A monotonic
    # clock that nothing can adjust is perf_counter or monotonic.
    info = time.get_clock_info(train.clock.__name__)
    assert info.monotonic and not info.adjustable


def test_fp16_with_fused_adamw_is_refused_on_mps():
    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError):
        train.make_optimizer(model, TrainConfig(device="mps", dtype="fp16", optimizer="fused"))
    train.make_optimizer(model, TrainConfig(device="cpu", dtype="fp16", optimizer="fused"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
@pytest.mark.parametrize("fused", [True, False])
def test_overflowed_fp16_step_on_mps(fused):
    # The reason for the check above. A gradient scaler has to skip a step
    # whose gradients overflowed. foreach AdamW does; fused AdamW on MPS does
    # not. If the fused case starts failing, torch has fixed the bug and
    # make_optimizer no longer needs to refuse the combination.
    p = torch.nn.Parameter(torch.ones(4, device="mps"))
    opt = torch.optim.AdamW([p], lr=0.1, fused=fused, foreach=None if fused else True)
    scaler = torch.amp.GradScaler("mps", init_scale=4.0)
    scaler.scale(p.sum() * float("inf")).backward()
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    skipped = torch.equal(p.detach().cpu(), torch.ones(4))
    assert skipped != fused


def test_config_file_then_flags(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"n_layer": 3, "lr": 5e-4, "compile": True}))
    cfg = train.parse_config([str(path), "--lr", "2e-3", "--compile", "false"])
    assert (cfg.n_layer, cfg.lr, cfg.compile) == (3, 2e-3, False)
    assert cfg.n_embd == TrainConfig().n_embd
    path.write_text(json.dumps({"n_layers": 3}))  # a typo must not be ignored
    with pytest.raises(TypeError):
        train.parse_config([str(path)])
