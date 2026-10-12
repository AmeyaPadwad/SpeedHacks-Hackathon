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

## D9. Retrieval query lives in the TextEmbedder space
The vector index is built with `TextEmbedder`, so the runtime builds its query by
embedding the redacted app name, title and OCR text with that same embedder,
not with `SomaModel.query_embedding`. Setting `TextEmbedder(dim=d_model)` makes
retrieved vectors feed straight into memory fusion. A learned alignment
(contrastive retrieval loss, `lambda_retrieval > 0`) waits for real retrieval
supervision.

## D10. One screen encode per frame
The runtime calls `encode_screen` once per frame and reuses that state for
history, temporal, memory fusion and the policy. A test checks that this
equals `model(batch)`.

## D11. Pause is a hard boundary
Pause revokes capture at once through `ConsentManager`, clears the rolling
history and bumps a generation counter. The runtime drops any result computed
across a pause, and such a result is never recorded as shown.

## D12. Feedback labels
Accept means the shown candidate (weight 1.0) and edit means the shown
candidate (weight 0.5). Reject is labelled NO_SUGGESTION (weight 0.5) and
dismisses the key in the gate. Ignored is stored as ambiguous and is never a
negative. Training snapshots have their pixels zeroed.

## D13. Replay ratio 3:1 for continual updates
With feedback and replay mixed 1:1, the synthetic forgetting check failed
(about 0.21 drop on prior competencies) and the gate rolled back. Using 3x
replay kept forgetting within tolerance. Training only the policy head, or
fusion plus policy, also forgot more than training all parameters with replay.

## D14. Retrieval always returns top-k (known limitation)
`Retriever.retrieve` has no minimum-relevance floor, so a
COMPARE_WITH_PRIOR_WORK candidate exists on most screens and the policy has to
learn to abstain. This keeps the decision in the learnable ranker, but a
calibrated relevance floor is a candidate M5 experiment.
