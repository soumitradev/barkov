# Barkov

A markov chain text generator. Heavily inspired by https://github.com/jsvine/markovify, but much faster, and with a few more bells and whistles. Gives way more control to the user while keeping the simplicity of the API.

## Installation

Requires Go 1.25 or newer.

```bash
go get github.com/soumitradev/barkov/v2
```

This README describes the API on this branch. For local development against a checkout containing changes that haven't been tagged yet:

```bash
go mod edit -replace github.com/soumitradev/barkov/v2=../barkov
```

## Usage

Barkov doesn't implement a tokenizer, you pick your own tokenizer and pass barkov a slice of tokens. Each inner slice is one message or sentence; training doesn't link the end of one to the start of the next.

Import `github.com/soumitradev/barkov/v2/text`. These snippets assume `corpus` is your `[][]string` and `ctx` is a `context.Context`:

```go
gen, err := text.New(corpus)
if err != nil {
    return err
}
sentence, err := gen.Sentence(ctx)
```

`Sentence` returns a string joined with spaces. `Generate` returns the `[]string` tokens so you can join them however you want. Check the generation error too: a corpus can build successfully without having an acceptable continuation for a particular seed.

`text.New` builds a variable-order markov chain and steers generation away from verbatim corpus runs. It also rejects outputs that exactly reproduce a complete corpus message. The rest of this README is for when you need more control. There's a complete runnable program in [examples/text](examples/text/main.go).

Options, if you need them:

```go
gen, err := text.New(corpus,
    text.Order(4),               // context scale, 2-8 (default 4)
    text.Timeout(5*time.Second), // budget for the whole Generate call
    text.MaxTokens(30),          // includes seed tokens; longer attempts are rejected
    text.Bidirectional(),        // also build a reverse chain for SeedMiddle
)
if err != nil {
    return err
}
out, err := gen.Generate(ctx, text.Seed([]string{"the", "quick"}))
```

`Seed` starts from the supplied words. `SeedMiddle` grows around them, using the reverse chain as well:

```go
out, err := gen.Generate(ctx,
    text.SeedMiddle([]string{"brown", "fox"}),
    text.WithMaxTokens(20),
    text.WithTimeout(2*time.Second),
    text.WithRetries(32),
    text.WithThreads(2),
)
```

Seed lookup tries each word exactly, then lowercased. Unknown words are dropped; if none remain, generation is unseeded. `SeedMiddle` requires `Bidirectional()` at construction. It tries both directions and can use one successful half if the other fails validation.

The construction options and their defaults:

| Option | Default | Meaning |
| --- | --- | --- |
| `Order(n)` | `4` | Context scale `2..8`; the longest context is `min(n+2, 8)`, so `Order(4)` looks back up to six tokens |
| `MinOrder(n)` | `1` | Shortest allowed context in tokens, `1..min(Order+2, 8)`; generation fails rather than falling below it |
| `MinSupport(n)` | `1` | Minimum observations required at every allowed order, `1..65535`; independent of context length |
| `AntiVerbatim(n)` | Longest context | Steering rejects continuations completing a corpus n-gram; positive widths are clamped to `2..min(Order+2, 8)`. `0` disables steering and whole-message rejection |
| `MaxTokens(n)` | `0` | Maximum output length, including the seed; `0` means unlimited |
| `Timeout(d)` | `10*time.Second` | Budget across all retries in one call; `0` adds no timeout, but the caller's context still applies |
| `Retries(n)` | `64` | Maximum attempts; must be at least one |
| `Threads(n)` | `0` | Workers per attempt; `0` and `1` are sequential |
| `Bidirectional()` | Off | Build a reverse chain for middle-out generation |
| `RNG(r)` | Package RNG | Supply a `*rand.Rand` from `math/rand/v2` for reproducible sequential generation |

`WithMaxTokens`, `WithTimeout`, `WithRetries`, and `WithThreads` override those defaults for one call. Negative lengths, timeouts, or worker counts return errors, as do zero or negative retry budgets. Length limits reject an attempt; they don't truncate the result. Parallelism multiplies attempts, so `Retries(32)` with `Threads(2)` can start up to 64 forward walks, or up to twice that with both directions.

Without an RNG override, generators support concurrent `Generate` and `Sentence` calls. A fixed RNG must not be shared between concurrent calls, and `Threads` greater than one is rejected with it. Middle-out generation uses that RNG sequentially too. `gen.SetRNG(r)` replaces the source; `gen.SetRNG(nil)` restores the default. Change the RNG only when generation is idle.

## Why

Markovify was slow, and it didn't give me control over tokenization or validation without overriding its classes. Plus I feel like the memory layout was very bloated and I didn't want to use Python either. So this library stays barebones about both: tokenize your text however you want, validate a sentence however you see fit. If you don't want the chain struct either, fine, there's a `GenerativeChain` interface to satisfy.

Not in scope: tokenization (you bring the tokens), combining models.

## The core API

The `text` package handles construction, retry budgets, and string conversion. Use the core package for custom token types, validators, or direct chain access.

Strings, no extras:

```go
chain := barkov.InitChain(4).BuildCompressed(corpus)
out, err := barkov.Gen(ctx, chain, barkov.WithMaxTokens[string](30))
```

`InitChain(4)` uses a fixed four-token context. `BuildRaw(corpus)` gives you the mutable model for inspection or pruning; `BuildCompressed(corpus)` builds the compact representation directly. To prune before compressing:

```go
raw := barkov.InitChain(4).BuildRaw(corpus)
raw.Prune(2)
if err := raw.Validate(); err != nil {
    return err
}
chain := raw.Compress()
out, err := barkov.Gen(ctx, chain)
```

Pruning can remove the begin state, so successful validation doesn't guarantee generation will succeed. The default string encoder uses `barkov.SEP` between tokens; keep it and the `BEGIN`/`END` sentinels out of your actual tokens.

The same thing with tokens interned to integer IDs first (less memory, faster builds on large corpora):

```go
vocab := interned.NewVocabulary()
encoded := vocab.InternCorpus(corpus)
chain := interned.Build(4, encoded) // fixed stateSize 1-8
ids, err := barkov.Gen(ctx, chain)
if err != nil {
    return err
}
out := vocab.DecodeTokens(ids)
```

And the engine `text` uses, exposed directly:

```go
chain := interned.BuildBackoff(interned.BackoffConfig{
    MaxOrder:    6, // longest context, 2-8; default 6
    MinOrder:    1, // shortest allowed context; default 1
    MinSupport:  1, // minimum observations, independently of context length
    MaxVerbatim: 6, // block copied six-token spans; -1 disables steering
}, encoded)
ids, err := barkov.Gen(ctx, chain)
```

`BackoffConfig{}` selects those defaults. `MinOrder` accepts `1..MaxOrder` (`0` selects `1`); the floor applies both to sparse contexts and copy-prevention fallback. Begin-of-message padding still supplies context at startup. A higher floor can leave seeds or attempts without any supported continuation. Lower-order tables remain available for copy detection and state inspection. `MinSupport` accepts `1..65535` and applies even at `MinOrder`; its default is now `1` rather than the previous hidden `3`. Higher values reject rare contexts and may exhaust the permitted orders; `MaxVerbatim` accepts `2..MaxOrder`, with `0` selecting `MaxOrder`. `Build` and `BuildBackoff` panic on out-of-range configuration. The higher-level `text.New` returns configuration errors instead. Unlike `text`, direct `BuildBackoff` doesn't install the whole-message validator or retry rejected attempts for you.

Why not always run a fixed-order chain? On real text many states have exactly one continuation, so the walk replays the corpus verbatim with a random branch every once in a while. Backoff tries the longest context with enough observations and an allowed continuation, falling back to shorter contexts where the corpus is thin or steering excludes every candidate.

For a custom comparable token type, implement `StateEncoder[T]` and pass it with the state size and sentinels to `barkov.NewChain(barkov.ChainConfig[T]{...})`. `AppendEncoder[T]` optionally lets builders and validators reuse encoding buffers. See [examples/custom](examples/custom/main.go).

### Streaming and errors

`Gen` returns `([]T, error)`. `GenIter` exposes an `iter.Seq2[T, error]`:

```go
for token, err := range barkov.GenIter(ctx, chain) {
    if err != nil {
        return err
    }
    fmt.Println(token)
}
```

Core generation makes one attempt by default. `WithSeed`, `WithMaxTokens`, and the validator options work with both calls. There is no built-in core timeout; pass a context with a deadline. Cancellation is checked between tokens, including seeds and buffered results.

`Gen` discards partial output on failure. `GenIter` can already have yielded a prefix when a later step fails, so keep that in mind when writing directly to an output stream. Whole-output validation and parallel generation buffer the candidate before yielding it.

Use `errors.Is` with `barkov.ErrSentenceFailedValidation` for rejected or over-length attempts, `barkov.ErrInsufficientSupport` for observed contexts below the support threshold, `barkov.ErrStateNotFound` for an unavailable state, and `context.Canceled` / `context.DeadlineExceeded` for cancellation. `text.Generate` retries validation, length and support failures up to its configured budget; it returns other errors immediately.

## Validators

A validator is any `func([]T) bool`. Install a sliding-window check with `barkov.WithValidator`; `false` ends that attempt with `ErrSentenceFailedValidation`. Core `Gen` doesn't automatically retry it.

Two footguns, and the options that remove them:

- `WithValidator` slides a `StateSize+2` window regardless of what your validator expects. If the validator knows its own width (`NGramSet`, `nhash.HashNGramSet`), use `WithNGramValidator` instead, and `Gen` slides exactly `v.N()` tokens. The width must be positive
- The sliding window never sees outputs shorter than itself. `WithOutputValidator` checks the complete candidate, including the seed, after the walk succeeds

```go
chain := barkov.InitChain(4).BuildCompressed(corpus)
set := barkov.NewNGramSet(corpus, 6, barkov.DefaultStringEncoder)
out, err := barkov.Gen(ctx, chain,
    barkov.WithNGramValidator(set),
    barkov.WithOutputValidator(func(tokens []string) bool {
        return len(tokens) >= 5
    }),
    barkov.WithMaxTokens[string](30),
    barkov.WithParallelism[string](4),
)
```

Here each of four workers tries a candidate and applies both validators. The first accepted candidate wins. Workers are canceled and joined before the call returns; custom chains and callbacks must return promptly for that to finish. `WithThreaded[T]()` is shorthand for `runtime.NumCPU()*8` workers; `WithParallelism[T](n)` lets you choose a smaller bound. Benchmark whether parallel attempts help your corpus and validation rules.

The stock anti-verbatim validators are `NGramSet` (stores complete encoded n-grams) and `nhash.HashNGramSet` (stores 64-bit digests). Hash collisions can reject a new n-gram; that probability grows with the set size. `nhash.New(corpus, n, encoder, hasher)` takes an explicit `hashers.Hasher`: `xxh3.XXH3{}`, `xxhash64.XXHash64{}`, or the zero-dependency `fnv.FNV{}`. See [examples/validators](examples/validators/main.go).

Backoff steering at width `r` prevents newly generated corpus spans of length `r` or longer; stricter checks on shorter spans still need a validator. Sliding checks don't independently validate an already-supplied seed; use a whole-output validator when you need that guarantee. `interned.NewMessageSet(encoded).NotWholeMessage()` supplies the whole-message rejection used by `text`.

## Escape hatches

If you don't like a util we write you can plug in your own. `StateEncoder[T]` for state keys, `hashers.Hasher` for digests, `StuckDetector` for failed-seed tracking, `SlicePool[T]` for allocation reuse, `GenerativeChain[T]` for the chain itself. Compressed and interned chains expose `barkov.RNGSettable` (`SetRNG`); interned chains also expose `barkov.FastMoverKey[K, T]` (`MoveKey`).

`gen.Chain()`, `gen.ReverseChain()`, and `gen.Vocab()` expose the models and vocabulary behind `text`. `ReverseChain()` is nil unless built with `Bidirectional()`. `gen.Contains(word)` uses the same exact/lowercase lookup as seeds. Inspect contexts without decoding integer IDs yourself:

```go
for tokens, observations := range gen.States(2) {
    fmt.Println(tokens, observations)
}
```

The count is how often that context was observed, summed across its possible next tokens. Direct backoff chains expose the same iterator with token IDs through `interned.BackoffChain`. A `*barkov.CompressedChain[T]` has `States()`, `StateTotal(key)`, and `ChoicesCumDist(key)`; the latter returns read-only slices into model storage. `gen.MessageSet()` exposes the whole-message digest set, or nil when anti-verbatim is disabled. Treat model and vocabulary accessors as read-only while generating.

## Subpackages

- `.../v2/text`: the quick start above. Start here
- `.../v2`: the core API: `Chain[T]`, `CompressedChain[T]`, `Gen`, `GenIter`, `NGramSet`, encoders
- `.../v2/interned`: `Vocabulary`, `TokenID`, `PackedEncoder`, `Build`, `BuildBackoff`, `MessageSet`, `TokenIDPool`
- `.../v2/nhash`: `HashNGramSet[T]`
- `.../v2/hashers/{xxh3,xxhash64,fnv}`: hashers for `nhash`
- `.../v2/stuck`: failure-count cache with TTL expiry on access, used by `WithStuckDetector` with parallel generation

## Performance

Benchmark data lives in [benchstat/](benchstat/). The recorded full-length novel benchmark (~630k tokens) showed the interned pipeline running about 2.9x faster and allocating about two-thirds fewer bytes than the plain string pipeline ([results](benchstat/steer_snapshot_build_reuse.txt)). Allocations per op were higher on the interned path because interning has fixed per-build setup the string path skips. The trade is more, smaller allocations for fewer bytes and less wall-clock. These are benchmark snapshots; measure your own corpus and settings.

## Compatibility

`v1.x` is feature-frozen at `v1.0.3` on `master`, critical bug fixes only. `v2.x` lives at `github.com/soumitradev/barkov/v2`. The recorded v1 comparison showed the v2 interned pipeline running about 4.8x faster, allocating about one-quarter as many bytes and making ~57x fewer allocations per op ([results](benchstat/v1_vs_v2.txt)). See [MIGRATION.md](MIGRATION.md) for v1 migration notes.
