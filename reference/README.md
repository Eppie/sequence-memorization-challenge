# Third-party source — not my work

Everything in this directory, and `../reference_post.md`, is by **Linus Linsefors and Lucius
Bushnaq**, fetched from their post

> *Challenge: Hand coding weights for efficient sequence memorisation*, LessWrong,
> 2026-07-23 —
> <https://www.lesswrong.com/posts/KWtchKwwnJkd4bwCi/challenge-hand-coding-weights-for-efficient-sequence-1>

and its linked code. It is kept here for one purpose: `../tests/test_matches_reference.py`
checks my reimplementation against it directly, rather than against my reading of the post.
That check is what lets the reproduction claim "fact generation is byte-identical, and the
construction's weights are bit-identical wherever the frequency ranking is unambiguous"
mean something.

No licence was stated at the source. It is included here under the assumption that
verbatim inclusion for verification is fair use; if either author would prefer it removed,
open an issue and I will replace it with a fetch script and a checksum.

Nothing in this directory is imported by `handcode/`. It is used only by the differential
test.
