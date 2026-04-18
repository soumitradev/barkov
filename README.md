# Barkov

A Markov chain text generator. Heavily inspired by https://github.com/jsvine/markovify, but with a different philosophy and a much faster build/gen pipeline.

## Philosophy

> This implementation is quite barebones and does not come with tokenization or validation code. You can choose to tokenize your text however you want, and validate a sentence in whichever way you see fit. If you don't want to use the chain struct that I've defined, and want to use your own, fine, there's a `GenerativeChain` interface you need to satisfy.

For simple use, the core string chain is three lines of setup and fast enough for most corpora.

When performance and memory matter more, opt into token interning via the `interned` package. At stateSize=4 with `interned.BuildCompressedIndexed`, the build-and-generate pipeline runs roughly 1.7x faster and uses about 35% less memory than the plain string chain on a novel-length corpus (see `benchstat/pipeline_simple_vs_maxopt.txt`), against the same `Gen` / `GenIter` / `GenerativeChain[T]` surface.

## Installation

```bash
go get github.com/soumitradev/barkov/v2
```

## Three tiers of usage

The API is layered so that each tier can use only what it needs. Runnable examples are in `examples/`.

### Tier 1: Drop-in (`examples/simple`)

Pure strings, no extras. Three lines of setup.

```go
chain := barkov.InitChain(4).BuildCompressed(corpus)
out, err := barkov.Gen(context.Background(), chain)
```

### Tier 2: Tuned (`examples/optimized`)

Same surface, with token interning and packed state keys for a faster build path and lower memory footprint.

```go
chain, vocab := interned.InitChain(4)
encoded := vocab.InternCorpus(corpus)
compressed := chain.BuildCompressed(encoded)

out, err := barkov.Gen(context.Background(), compressed)
```

> [!TIP]
> For stateSizes 2–8, swap `chain.BuildCompressed(encoded)` for `interned.BuildCompressedIndexedN(encoded)` (where `N` matches your stateSize, e.g. `BuildCompressedIndexed5`). `BuildCompressedIndexed` without a suffix is shorthand for the N=4 case. Indexed chains are barkov's fastest build-and-gen path, with lower memory on large corpora. The returned chain satisfies `GenerativeChain[interned.TokenID]`, so the rest of your code is unchanged.

### Tier 3: Custom (`examples/custom`)

Bring your own token type. Implement `StateEncoder[T]` (and optionally `AppendEncoder[T]` for the zero-alloc fast path). Since Go maps can't be keyed on slices, the returned `string` is just packed bytes used as a map key.

```go
type int64Encoder struct{}
func (int64Encoder) Encode(tokens []int64) string { ... }
func (int64Encoder) Decode(state string) []int64 { ... }

chain := barkov.NewChain(barkov.ChainConfig[int64]{
    StateSize: 4,
    Sentinels: barkov.Sentinels[int64]{Begin: -1, End: -2},
    Encoder:   int64Encoder{},
})
compressed := chain.BuildCompressed(corpus)
```

## Validators

A validator is any function `func([]T) bool`. `Gen` calls it on each candidate n-gram as the chain walks; if it returns `false`, the candidate is rejected and `Gen` retries. Pass one with `barkov.WithValidator(fn)`. Without it, every candidate is accepted.

Validators are arbitrary. You might want to write a validator to reject profanity, cap sentence length, block specific tokens, enforce POS patterns etc.

### Anti-verbatim helpers

The single most common validator is anti-verbatim: reject any output that reproduces a corpus n-gram exactly. This matters when you're publishing derivative text and an accidental verbatim reproduction would defeat the point.

Barkov ships two implementations of this specific check:

- `NGramSet[T]` (core package): stores every corpus n-gram and rejects exact matches.
- `nhash.HashNGramSet[T]`: stores 64-bit digests instead, roughly halving memory on large corpora in exchange for a ~1-in-2^64 chance of rejecting a generation for matching a digest collision rather than a real n-gram.

Pick hashed unless your corpus is small enough that memory doesn't matter.

The hash function itself is pluggable via `hashers.Hasher`: `xxh3` is the default high-speed choice, `xxhash64` is an alternative, and `fnv` is zero-dep (stdlib only).

> [!NOTE]
> `barkov.WithThreaded()` fans generation out across `runtime.NumCPU() * 8` goroutines and returns the first candidate a validator accepts. On a typical corpus where the validator accepts almost every candidate, this is strictly slower than sequential (the goroutine fan-out costs more than it saves). It only pays off when the validator is rejecting most candidates, e.g. a tiny corpus where short outputs are verbatim reproductions of the source, or a very strict custom validator.

## Escape hatches

Everything concrete has an interface to swap it out.

| Concrete choice | Escape hatch |
| --- | --- |
| `Vocabulary` for interning | Roll your own interner. `barkov` only requires `StateEncoder[T]` |
| `PackedEncoder` 4-byte LE encoding | `StateEncoder[T]` + optional `AppendEncoder[T]` |
| xxh3 hashing | `hashers.Hasher`: ship `xxh3`, `xxhash64`, or `fnv`, or supply your own |
| Built-in `NGramSet` validator | `func([]T) bool`: pass any function to `WithValidator` |
| `nhash.HashNGramSet` | Same. `Validator()` returns a plain closure |
| `stuck.Cache` retry detector | `StuckDetector` interface; nil disables |
| `sync.Pool` slice reuse | `SlicePool[T]` interface; default `NoPool` (no reuse) |
| `Chain` struct itself | `GenerativeChain[T]` interface with exported `Move` |

## Subpackages

| Path | What it gives you | Needed for |
| --- | --- | --- |
| `github.com/soumitradev/barkov/v2` | Core: `Chain[T]`, `CompressedChain[T]`, `Gen`, `GenIter`, `NGramSet[T]`, `SepEncoder` | Everything |
| `.../v2/interned` | `Vocabulary`, `TokenID`, `PackedEncoder`, `IndexedCompressedChainN` for stateSizes 2–8 (unsuffixed alias = N=4) | Tier 2 |
| `.../v2/nhash` | `HashNGramSet[T]`: hash-keyed validator | Tier 2 with a hashed validator |
| `.../v2/hashers` | `Hasher` interface | Implementers |
| `.../v2/hashers/xxh3` | Default high-speed hasher (via `github.com/zeebo/xxh3`) | Tier 2 |
| `.../v2/hashers/xxhash64` | Alternative hasher (via `github.com/cespare/xxhash/v2`) | Tier 2 |
| `.../v2/hashers/fnv` | Zero-dep hasher (stdlib `hash/fnv`) | Tier 2 when you don't want external deps |
| `.../v2/stuck` | TTL-bounded stuck-seed cache | Tier 2 with aggressive retry loops |

## Performance

Benchmark data lives in `benchstat/`.

The Tier 2 best-case pipeline (stateSize=4 with `interned.BuildCompressedIndexed`) runs roughly 1.7x faster and uses about 35% less memory than the plain Tier 1 string pipeline on a novel-length corpus. See `benchstat/pipeline_simple_vs_maxopt.txt`. Other stateSizes (2, 3, 5–8) use the same indexed path via `BuildCompressedIndexedN`.

Allocations per op go up on the interned path because interning has fixed per-build setup cost (vocabulary, packed keys) that the plain string path skips. The trade is fewer bytes and faster wall-clock at the cost of more, smaller allocations.

## Compatibility

> [!IMPORTANT]
> If you're on `v1.x`, upgrade. On the same corpus, the `v2.x` best-case pipeline is roughly 4x faster, uses about 3x less memory, and allocates ~50x fewer times per op than `v1.0.3`. See `benchstat/v1_vs_v2.txt`. `MIGRATION.md` has before/after snippets for every API change.

`v1.x` (the pre-generics string-only API) is retained on `main` for critical bug fixes. It is frozen at `v1.0.3` for features.

`v2.x` lives at module path `github.com/soumitradev/barkov/v2`.
