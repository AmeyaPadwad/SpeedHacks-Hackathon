# SOMA design decisions

Short records of choices made while building v0.1. Newest last.

## D1. Self-contained project inside `SOMA/`
SOMA lives entirely in `SOMA/` on the `SOMA` branch, with its own `pyproject.toml`
and a local `.venv`. It does not import from the Pette app.

## D2. PyTorch is the model, not a remote LLM
Every learnable component (visual, text/layout, fusion, temporal, memory fusion,
policy) is a local `nn.Module`. No API call substitutes for `SomaModel`.

## D3. Bounded action vocabulary with index 0 = `NO_SUGGESTION`
Policy logits are `[B, A+1]`; index 0 is a trainable abstain logit. Blocked or
padded candidates get a large negative logit (-1e4, safe in fp16), so if every
candidate is blocked, abstention takes the probability mass with no NaNs.

## D4. Masks: `True` = PAD
We follow PyTorch's `key_padding_mask` convention everywhere. All-masked
attention inputs (empty OCR, empty memory) get explicit handling: a learned
null key/value (`NO_MEMORY`) or a guarded pooling path, never softmax over
nothing.

## D5. Hashed tokenizer instead of a shipped vocabulary
OCR text and memory summaries are tokenized with a stable blake2b hash into a
fixed vocabulary. There is no vocab file to ship and no training-data leakage
through a vocabulary, and IDs stay stable across processes. Collisions are
acceptable at v0.1 scale.

## D6. Consent is per-scope and defaults to off
`soma.consent.ConsentManager` has independent scopes (capture, app monitoring,
folder indexing, feedback storage, training, remote inference). Pause blocks
capture at once. Refusing a home directory or drive root is enforced in code.

## D7. Single visual resolution first
Following plan section 2, v0.1 uses one patch resolution. Multi-scale is an M5
ablation, benchmarked rather than assumed.

## D8. Memory is external to weights
SQLite holds metadata, provenance and feedback, and a vector index holds
embeddings. Permission filtering happens before vector scoring, so blocked items
are never scored at all. Deletion purges rows, vectors and derived files.
