# Barkov

A Markov chain text generator. Heavily inspired by https://github.com/jsvine/markovify, but with a different philosophy and a much faster build/gen pipeline.

## Philosophy

> This implementation is quite barebones and does not come with tokenization or validation code. You can choose to tokenize your text however you want, and validate a sentence in whichever way you see fit. If you don't want to use the chain struct that I've defined, and want to use your own, fine — there's a `GenerativeChain` interface you need to satisfy.

Every optimisation in barkov is opt-in. You don't have to use token interning, xxh3 hashing, slice pools, or the built-in `NGramSet` validator. If you want the simple string chain and nothing else, you get exactly that.

## Installation

```bash
go get github.com/soumitradev/barkov/v2
```

## Three tiers of usage

The API is layered so that each tier can use only what it needs. Runnable examples are in `examples/`.

### Tier 1 — Drop-in (`examples/simple`)

Pure strings, no extras. Three lines of setup.

```go
chain := barkov.InitChain(2).BuildCompressed(corpus)
out, err := barkov.Gen(context.Background(), chain)
```

### Tier 2 — Tuned (`examples/optimized`)

Same surface, every performance lever pulled: token interning, packed state keys, xxh3-hashed validator, threaded retries.

```go
chain, vocab := interned.InitChain(stateSize)
encoded := vocab.InternCorpus(corpus)
compressed := chain.BuildCompressed(encoded)

validator := nhash.New(encoded, compressed.MaxOverlap(), interned.PackedEncoder{}, xxh3.XXH3{}).Validator()

out, err := barkov.Gen(ctx, compressed,
    barkov.WithValidator(validator),
    barkov.WithThreaded[interned.TokenID](),
)
```

> [!TIP]
> If your stateSize is exactly 4, swap `chain.BuildCompressed(encoded)` for `interned.BuildCompressedIndexed(encoded)` — it's roughly twice as fast on large corpora (see `benchstat/e2e_v2.0.0-beta.4.txt`, `interned/plain-24`). The returned chain satisfies `GenerativeChain[interned.TokenID]`, so the rest of your code is unchanged.

### Tier 3 — Custom (`examples/custom`)

Bring your own token type. Implement `StateEncoder[T]` (and optionally `AppendEncoder[T]` for the zero-alloc fast path).

```go
type int64Encoder struct{}
func (int64Encoder) Encode(tokens []int64) string { ... }
func (int64Encoder) Decode(state string) []int64 { ... }

chain := barkov.NewChain(barkov.ChainConfig[int64]{
    StateSize: 2,
    Sentinels: barkov.Sentinels[int64]{Begin: -1, End: -2},
    Encoder:   int64Encoder{},
})
compressed := chain.BuildCompressed(corpus)
```

## Escape hatches

Everything concrete has an interface to swap it out.

| Concrete choice | Escape hatch |
| --- | --- |
| `Vocabulary` for interning | Roll your own interner — `barkov` only requires `StateEncoder[T]` |
| `PackedEncoder` 4-byte LE encoding | `StateEncoder[T]` + optional `AppendEncoder[T]` |
| xxh3 hashing | `hashers.Hasher` — ship `xxh3`, `xxhash64`, or `fnv`, or supply your own |
| Built-in `NGramSet` validator | `func([]T) bool` — pass any function to `WithValidator` |
| `nhash.HashNGramSet` | Same — `Validator()` returns a plain closure |
| `stuck.Cache` retry detector | `StuckDetector` interface; nil disables |
| `sync.Pool` slice reuse | `SlicePool[T]` interface; default `NoPool` (no reuse) |
| `Chain` struct itself | `GenerativeChain[T]` interface with exported `Move` |

## Subpackages

| Path | What it gives you | Needed for |
| --- | --- | --- |
| `github.com/soumitradev/barkov/v2` | Core: `Chain[T]`, `CompressedChain[T]`, `Gen`, `GenIter`, `NGramSet[T]`, `SepEncoder` | Everything |
| `.../v2/interned` | `Vocabulary`, `TokenID`, `PackedEncoder`, `IndexedCompressedChain` (stateSize=4) | Tier 2 |
| `.../v2/nhash` | `HashNGramSet[T]` — hash-keyed validator | Tier 2 with a hashed validator |
| `.../v2/hashers` | `Hasher` interface | Implementers |
| `.../v2/hashers/xxh3` | Default high-speed hasher (via `github.com/zeebo/xxh3`) | Tier 2 |
| `.../v2/hashers/xxhash64` | Alternative hasher (via `github.com/cespare/xxhash/v2`) | Tier 2 |
| `.../v2/hashers/fnv` | Zero-dep hasher (stdlib `hash/fnv`) | Tier 2 when you don't want external deps |
| `.../v2/stuck` | TTL-bounded stuck-seed cache | Tier 2 with aggressive retry loops |

## Performance

Benchmark data lives in `benchstat/`. The end-to-end path (build + compress + generate one sentence) on the stateSize=4 interned path is roughly 2× faster than `v2.0.0-beta.2` and allocates ~99% fewer allocs/op. See `benchstat/e2e_v2.0.0-beta.4.txt` for the latest numbers.

## Compatibility

`v1.x` (the pre-generics string-only API) is retained on `main` for critical bug fixes. It is frozen at `v1.0.3` for features.

`v2.x` lives at module path `github.com/soumitradev/barkov/v2`. `MIGRATION.md` has the full v1 → v2 changelog with before/after snippets.
