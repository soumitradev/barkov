# Barkov

A markov chain text generator. Heavily inspired by https://github.com/jsvine/markovify, but much faster, and with a few more bells and whistles. Gives way more control to the user while keeping the simplicity of the API.

## Installation

```bash
go get github.com/soumitradev/barkov/v2
```

## Usage

Barkov doesn't implement a tokenizer, you pick your own tokenizer and pass barkov a slice of tokens.

```go
gen, err := text.New(corpus) // corpus is a 2D slice of tokens: [][]string
sentence, err := gen.Sentence(ctx)
```

`text.New` builds a variable-order markov chain from the corpus and steers generation so the output doesn't reproduce corpus text verbatim. The rest of this README is for when you need more control.

Options, if you need them:

```go
gen, err := text.New(corpus,
    text.Order(4),               // context size, 2-8 (default 4)
    text.Timeout(5*time.Second), // per-sentence budget (default 10s)
    text.Threads(8),             // parallel attempts (default sequential)
)
out, err := gen.Generate(ctx, text.Seed([]string{"once", "upon"}))
```

`gen.Chain()` and `gen.Vocab()` hand you the underlying engine and vocabulary if you want to work with them directly. Full program: `examples/text`.

## Why

Markovify was slow, and it didn't give me control over tokenization or validation without overriding its classes. Plus I feel like the memory layout was very bloated and I didn't want to use Python either. So this library stays barebones about both: tokenize your text however you want, validate a sentence however you see fit. If you don't want the chain struct either, fine, there's a `GenerativeChain` interface to satisfy.

Not in scope: tokenization (you bring the tokens), combining models.

## The core API

`text` is a thin wrapper over the core package, a few hundred lines with no magic in it. Drop to the core when you want a custom token type, your own validator, or direct access to the chain.

Strings, no extras:

```go
chain := barkov.InitChain(4).BuildCompressed(corpus)
out, err := barkov.Gen(ctx, chain)
```

The same thing with tokens interned to integer IDs first (less memory, faster builds on large corpora):

```go
vocab := interned.NewVocabulary()
encoded := vocab.InternCorpus(corpus)
chain := interned.Build(4, encoded) // stateSize 1-8
out, err := barkov.Gen(ctx, chain)
```

And the engine `text` uses, exposed directly: a variable-order chain that backs off to shorter contexts where the corpus is thin and steers around verbatim reproductions while it walks.

```go
chain := interned.BuildBackoff(interned.BackoffConfig{}, encoded) // zero value = defaults
out, err := barkov.Gen(ctx, chain)
```

Why not always run a fixed-order chain? On real text most states have exactly one continuation (~97% at stateSize 4 on prose), so the walk replays the corpus verbatim with a random branch every once in a while. Backoff uses the longest context with enough observations at each step, so the walk makes real choices where the corpus genuinely branches.

For a custom token type, implement `StateEncoder[T]` (and `AppendEncoder[T]` for the zero-alloc path) and pass them to `barkov.NewChain(ChainConfig[T]{...})`. See `examples/custom`. You can find more details about all this in `examples/`.

## Validators

A validator is any `func([]T) bool`. `Gen` calls it on a sliding window as the chain walks; `false` rejects the candidate and retries. Install one with `barkov.WithValidator`.

Two footguns, and the options that remove them:

- `WithValidator` slides a `StateSize+2` window regardless of what your validator expects. If the validator knows its own width (`NGramSet`, `nhash.HashNGramSet`), use `WithNGramValidator` instead, and `Gen` slides exactly `v.N()` tokens.
- The sliding window never sees outputs shorter than itself. `WithOutputValidator` runs once on the complete output and catches those.

The stock anti-verbatim validators are `NGramSet` (exact, fine on small corpora) and `nhash.HashNGramSet` (64-bit digests, roughly half the memory, ~1-in-2^64 false-positive rate). Under `BuildBackoff` and `text` steering they're redundant for n <= MaxVerbatim, since steering filters verbatim continuations during the walk. Validators remain the tool for everything else: profanity, length caps, POS patterns, whatever you like.

Hashing is pluggable via `hashers.Hasher`: `xxh3` is the default fast choice, `xxhash64` an alternative, `fnv` zero-dep. `WithThreaded` fans generation across goroutines and takes the first accepted candidate; it only pays off when the validator rejects most attempts.

## Escape hatches

Everything concrete has an interface. This way if you don't like a util we write you can just plug and play your own. `StateEncoder[T]` for state keys, `hashers.Hasher` for digests, `StuckDetector` for retry detection, `SlicePool[T]` for allocation reuse, `GenerativeChain[T]` for the chain itself. The concrete interned chain's extras are reachable through `barkov.RNGSettable` (`SetRNG`) and `barkov.FastMoverKey` (`MoveKey`). Godoc has the details.

## Subpackages

- `.../v2/text`: the quick start above. Start here.
- `.../v2`: the core API: `Chain[T]`, `CompressedChain[T]`, `Gen`, `NGramSet`, encoders.
- `.../v2/interned`: `Vocabulary`, `TokenID`, `PackedEncoder`, `Build` (stateSizes 1-8), `BuildBackoff`.
- `.../v2/nhash`: `HashNGramSet[T]`.
- `.../v2/hashers/{xxh3,xxhash64,fnv}`: hashers for `nhash`.
- `.../v2/stuck`: TTL-bounded stuck-seed cache for aggressive retry loops.

## Performance

Benchmark data lives in `benchstat/`. On a full-length novel (~630k tokens), the interned pipeline runs about 2.9x faster and uses about two-thirds less memory than the plain string pipeline (`benchstat/steer_snapshot_build_reuse.txt`). Allocations per op are higher on the interned path because interning has fixed per-build setup the string path skips. The trade is more, smaller allocations for fewer bytes and less wall-clock.

## Compatibility

`v1.x` is feature-frozen at `v1.0.3` on `master`, critical bug fixes only. `v2.x` lives at `github.com/soumitradev/barkov/v2`. On the same corpus, the v2 interned pipeline runs about 4.8x faster than v1 with about 4x less memory and ~57x fewer allocations per op (`benchstat/v1_vs_v2.txt`). `MIGRATION.md` has before/after snippets for every API change.
