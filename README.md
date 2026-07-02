# Barkov

A Markov chain text generator. Heavily inspired by https://github.com/jsvine/markovify, but with a different philosophy and a much faster build/gen pipeline.

## Philosophy

> This implementation is quite barebones and does not come with tokenization or validation code. You can choose to tokenize your text however you want, and validate a sentence in whichever way you see fit. If you don't want to use the chain struct that I've defined, and want to use your own, fine, there's a `GenerativeChain` interface you need to satisfy.

Barkov is layered like zap: a sugared `text` package for the common case — tokenized messages in, non-parroting sentences out — sitting over an unsugared core of generics, pluggable encoders, and functional options for when you need control. Reach for `text` first; drop to the core when you outgrow it.

## Installation

```bash
go get github.com/soumitradev/barkov/v2
```

## Quick start

Three lines from a tokenized corpus to sentences that don't parrot the source:

```go
gen, _ := text.New(corpus) // corpus is [][]string: one message's tokens per row
sentence, _ := gen.Sentence(context.Background())
```

`text.New` builds the vocabulary, interns the corpus, and picks the fastest engine for you, with anti-verbatim rejection on by default so outputs don't reproduce the source verbatim. Options tune it without ceremony:

```go
gen, _ := text.New(corpus,
    text.Order(4),               // context window (stateSize), 2–8
    text.Timeout(5*time.Second), // per-Generate budget
    text.Threads(8),             // fan attempts out across goroutines
)
out, _ := gen.Generate(ctx, text.Seed([]string{"once", "upon"}))
```

When you outgrow the facade, `gen.Chain()` and `gen.Vocab()` hand you the underlying core objects to use with everything below. Full example: `examples/text`.

## When you need control: three tiers

Drop below `text` for a custom token type, your own validator, or direct access to the chain. The core API is layered so each tier uses only what it needs. Runnable examples are in `examples/`.

### Tier 1: Drop-in (`examples/simple`)

Pure strings, no extras. Three lines of setup.

```go
chain := barkov.InitChain(4).BuildCompressed(corpus)
out, err := barkov.Gen(context.Background(), chain)
```

### Tier 2: Tuned (`examples/optimized`)

Same surface, with token interning and packed state keys for a faster build path and lower memory footprint.

```go
vocab := interned.NewVocabulary()
encoded := vocab.InternCorpus(corpus)
compressed := interned.Build(4, encoded) // fastest build-and-gen path, stateSize 1–8

out, err := barkov.Gen(context.Background(), compressed)
```

> [!TIP]
> `interned.Build(stateSize, encoded)` is barkov's fastest build-and-gen path at stateSizes 1–8, with lower memory on large corpora. The rest of your code is unchanged. If you need the concrete affordances, assert for the interface you want: `compressed.(barkov.RNGSettable).SetRNG(r)` for a deterministic RNG, or `compressed.(barkov.FastMoverKey[[4]interned.TokenID, interned.TokenID])` for direct `MoveKey`.

### Tier 3: Custom (`examples/custom`)

Bring your own token type. Implement `StateEncoder[T]` (and optionally `AppendEncoder[T]` for the zero-alloc fast path). The returned `string` is just packed bytes used as a map key.[^1]

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

### Validator shapes

`WithValidator` slides a fixed `StateSize+2` window: outputs shorter than that are never checked, and the width is fixed regardless of how the validator was built. Two options remove those footguns:

- `WithNGramValidator(v)` takes a `WindowValidator` that declares its own width through `N()`, so `Gen` slides a window of exactly that size. `NGramSet` and `nhash.HashNGramSet` both satisfy it. Use it instead of hand-wiring an `NGramSet` of a non-default width into `WithValidator`, which silently never matches.
- `WithOutputValidator(v)` runs once on the complete output, so it catches whole-sentence reproductions and the short outputs the sliding window skips. It makes generation non-streaming by definition.

The `text` package wires both for you: an `nhash.HashNGramSet` at `Order+2` plus a whole-message check, so short outputs that reproduce a complete corpus message are caught too.

### Anti-verbatim helpers

The single most common validator is anti-verbatim: reject any output that reproduces a corpus n-gram exactly. This matters when you're publishing derivative text and an accidental verbatim reproduction would defeat the point.

Barkov ships two implementations of this specific check:

- `NGramSet[T]` (core package): stores every corpus n-gram and rejects exact matches.
- `nhash.HashNGramSet[T]`: stores 64-bit digests instead, roughly halving memory on large corpora in exchange for a ~1-in-2^64 false-positive rate per candidate.

For large corpora, prefer `HashNGramSet`. Use `NGramSet` only on small corpora where the memory cost is trivial.

The hash function itself is pluggable via `hashers.Hasher`: `xxh3` is the default high-speed choice, `xxhash64` is an alternative, and `fnv` is zero-dep (stdlib only).

> [!NOTE]
> `barkov.WithThreaded()` is a validator-adjacent knob: it fans generation across `runtime.NumCPU() * 8` goroutines and returns the first candidate the validator accepts. It only pays off when the validator rejects most candidates — strict custom validators, or tiny corpora where short outputs reproduce the source verbatim. With a permissive validator the fan-out costs more than it saves.

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
| `interned.Build`'s concrete indexed type | Assert `barkov.RNGSettable` (SetRNG) or `barkov.FastMoverKey[[N]TokenID, TokenID]` (direct MoveKey) |

## Subpackages

| Path | What it gives you | Needed for |
| --- | --- | --- |
| `.../v2/text` | `text.New` → `Generator`: strings in, non-parroting sentences out | The common case |
| `github.com/soumitradev/barkov/v2` | Core: `Chain[T]`, `CompressedChain[T]`, `Gen`, `GenIter`, `NGramSet[T]`, `SepEncoder` | Everything below `text` |
| `.../v2/interned` | `Vocabulary`, `TokenID`, `PackedEncoder`, `Build` for stateSizes 1–8 | Tier 2 |
| `.../v2/nhash` | `HashNGramSet[T]`: hash-keyed validator | Tier 2 with a hashed validator |
| `.../v2/hashers` | `Hasher` interface | Implementers |
| `.../v2/hashers/xxh3` | Default high-speed hasher (via `github.com/zeebo/xxh3`) | Tier 2 |
| `.../v2/hashers/xxhash64` | Alternative hasher (via `github.com/cespare/xxhash/v2`) | Tier 2 |
| `.../v2/hashers/fnv` | Zero-dep hasher (stdlib `hash/fnv`) | Tier 2 when you don't want external deps |
| `.../v2/stuck` | TTL-bounded stuck-seed cache | Tier 2 with aggressive retry loops |

## Performance

Benchmark data lives in `benchstat/`.

We've observed the Tier 2 interned pipeline running roughly 1.7x faster and using about 35% less memory than the Tier 1 string pipeline on a full-length novel (~630k tokens). See `benchstat/pipeline_simple_vs_maxopt.txt`.

Allocations per op go up on the interned path because interning has fixed per-build setup cost (vocabulary, packed keys) that the plain string path skips. The trade is fewer bytes and faster wall-clock at the cost of more, smaller allocations.

## Compatibility

> [!IMPORTANT]
> If you're on `v1.x`, upgrade. On the same corpus, we've observed the `v2.x` best-case pipeline via `interned.Build` running roughly 4.8x faster, using about 4x less memory, and allocating ~57x fewer times per op than `v1.0.3`. See `benchstat/v1_vs_v2.txt`. `MIGRATION.md` has before/after snippets for every API change.

`v1.x` (the pre-generics string-only API) lives on `main`, feature-frozen at `v1.0.3`; critical bug fixes only.

`v2.x` lives at module path `github.com/soumitradev/barkov/v2`.

[^1]: Go maps can't be keyed on slices, so state keys have to be strings.
