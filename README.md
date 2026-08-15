# Token Saver

Token Saver is a local Rust retrieval service that builds a **custom Abstract Semantic Graph (ASG)** from a workspace, ranks code with **weighted Personalized PageRank (PPR)**, fuses independent retrievers with **custom Reciprocal Rank Fusion (RRF)**, and emits lossless, token-budgeted context for coding assistants.

## What is improved

- **Sparse semantic ASG:** stores complete functions, structs, enums, traits, impls, modules, constants, and types—not every parser node. Stable IDs include impl scope, so methods such as `A::new` and `B::new` cannot collide.
- **Cross-file relationships:** resolves `calls`, `references`, `contains`, and `implements` edges before ranking.
- **Query-aware weighted PPR:** call/reference/implementation edges can carry more importance than lexical containment. Semantic/BM25 candidates and the current cursor node form the per-query teleport vector.
- **Real BM25:** uses corpus document frequency and length normalization rather than treating term frequency as IDF.
- **Custom weighted RRF:** configurable stream weights plus bounded score-aware blending; deterministic tie handling and bad-score sanitization are built in.
- **Honest lossless compression:** a compact alias is used only when the body *plus its alias legend* is smaller than the original. Hydration is prefix-safe.
- **Budgeted context:** surrounding source and ranked dependencies are packed under configurable token limits.
- **Safer server:** workspace path containment, generation-safe debouncing, finite SSE streams, health/search endpoints, configurable workspace/bind address, and no hard-coded development path.
- **Private indexing:** AST chunks use Merkle change detection, local trigrams, AES-GCM encryption, keyed path aliases, and incremental vector synchronization.

## Run

A stable Rust toolchain is required.

```bash
cp token-saver.example.yaml token-saver.yaml
cargo run --release
```

Without a config file, the current directory is indexed and the service listens on `0.0.0.0:8080`.

Environment overrides:

```bash
TOKEN_SAVER_CONFIG=/path/to/token-saver.yaml cargo run --release
TOKEN_SAVER_WORKSPACE=/path/to/repository TOKEN_SAVER_BIND=0.0.0.0:9000 cargo run --release
```

## API

### Health

```bash
curl http://localhost:8080/health
```

### Inspect hybrid retrieval

```bash
curl -s http://localhost:8080/v1/search \
  -H 'content-type: application/json' \
  -d '{"query":"personalized page rank edge weights","top_k":5}'
```

Each hit reports the stable ASG tracker, global PageRank, fused score, raw stream scores, and per-stream RRF contributions.

### Stream autocomplete context

Paths may be absolute or relative to the configured workspace; paths outside that workspace are rejected.

```bash
curl -N http://localhost:8080/v1/autocomplete \
  -H 'content-type: application/json' \
  -d '{
    "file_path":"src/search/mod.rs",
    "line":130,
    "column":8,
    "query":"fuse semantic lexical structural results",
    "request_id":"editor-42"
  }'
```

The response is `text/event-stream` and contains local source context followed by the highest-value compressed dependencies.

## Configuration

See [`token-saver.example.yaml`](token-saver.example.yaml). The most important knobs are:

- `search.ppr.edge_weights`: structural meaning assigned to each ASG edge kind.
- `search.context_seed_weight`: strength of the editor cursor in query PPR.
- `search.rrf.weights`: `[semantic, bm25, structural]` trust weights.
- `search.rrf.score_alpha`: confidence scaling in `[0, 1]`; `0` is pure weighted RRF.
- `context.max_context_tokens`: total approximate prompt budget.
- `context.dependency_token_budget`: cap reserved for retrieved ASG dependencies.

## Test

```bash
cargo fmt --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

All embeddings and retrieval in the default service are local. `MemoryVectorStore` is an interface-compatible stand-in; integrate a persistent vector backend by implementing the `VectorStore` trait and supplying your own encryption key from a secret manager.
