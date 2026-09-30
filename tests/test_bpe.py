import random

from prepare import read_blocks, stories
from tinygpt.bpe import EOT, SPLIT, Tokenizer, merge, train_bpe

TEXT = [
    "Once upon a time, there was a little dog named Max. Max liked to play.",
    "One day, Lily found a shiny red ball in the park. She was very happy!",
    "Tom said, \"Let's share the toys.\" They played together all day.",
] * 20


def random_text(rng, n):
    # ASCII, whitespace runs, accents, curly quotes, CJK and emoji
    alphabet = "ab AB09_'\"!?.,\n\t  éüßç“”’–中文😀"
    return "".join(rng.choice(alphabet) for _ in range(n))


def test_merge_replaces_non_overlapping_pairs_left_to_right():
    assert merge([1, 1, 1, 2, 1, 1], (1, 1), 9) == [9, 1, 2, 9]


def test_split_covers_every_character():
    rng = random.Random(0)
    for _ in range(200):
        s = random_text(rng, 50)
        assert "".join(SPLIT.findall(s)) == s


def test_round_trip_on_arbitrary_text():
    tok = train_bpe(TEXT, 300)
    rng = random.Random(1)
    for _ in range(200):
        s = random_text(rng, 80)
        assert tok.decode(tok.encode(s)) == s


def test_vocab_layout():
    tok = train_bpe(TEXT, 300)
    assert tok.vocab_size == 300
    assert tok.eot == 299
    assert len(tok.merges) == len(set(tok.merges)) == 300 - 257
    assert tok.decode([tok.eot]) == EOT


def test_common_words_become_single_tokens():
    tok = train_bpe(TEXT, 400)
    assert len(tok.encode(" little")) == 1
    assert len(tok.encode("Once upon a time")) < len("Once upon a time") / 2


def test_encoding_matches_training_segmentation():
    # Training applies merge 0 to every word, then merge 1, and so on. The
    # encoder takes a shortcut (merge whichever learned pair is earliest) and
    # has to end up with exactly the same tokens.
    tok = train_bpe(TEXT, 330)
    words = set(SPLIT.findall(" ".join(TEXT)))
    for w in words:
        ids = list(w.encode("utf-8"))
        for i, pair in enumerate(tok.merges):
            ids = merge(ids, pair, 256 + i)
        assert list(tok._bpe(w)) == ids


def test_save_and_load(tmp_path):
    tok = train_bpe(TEXT, 320)
    tok.save(tmp_path / "tok.json")
    again = Tokenizer.load(tmp_path / "tok.json")
    s = "The little dog played with a ball."
    assert again.encode(s) == tok.encode(s)


def test_read_blocks_cuts_on_story_boundaries(tmp_path):
    parts = [f"Story {i}.\nIt has two lines." for i in range(50)]
    path = tmp_path / "stories.txt"
    path.write_text(f"\n{EOT}\n".join(parts) + f"\n{EOT}\n")
    got = [s for block in read_blocks(path, block_chars=64) for s in stories(block)]
    assert got == parts
