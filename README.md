# Barkov

A markov chain text generator. Heavily inspired by https://github.com/jsvine/markovify, but much faster, and with a few more bells and whistles. Gives way more control to the user while keeping the simplicity of the API.

## Installation

Requires Go 1.25 or newer.

```bash
go get github.com/soumitradev/barkov/v2
```

## Usage

Barkov doesn't implement a tokenizer, you pick/write your own tokenizer and pass barkov a slice of tokens. Each inner slice is one message or sentence.

```go
import "github.com/soumitradev/barkov/v2/text"

// corpus is a 2D slice of tokens: [][]string
gen, err := text.New(corpus)
if err != nil {
    return err
}
tokens, err := gen.Generate(ctx)   // a new output as []string
sentence, err := gen.Sentence(ctx) // another one, joined with spaces
```

- `text.New` builds a markov chain model with some sensible defaults (variable-order, steers away from copying source text, never returns a whole message from the corpus).
- `Generate` returns the tokens as `[]string`. `Sentence` is syntax sugar for `Generate` + `strings.Join(tokens, " ")`.

The rest of this README is for when you need more control.

Options:

```go
gen, err := text.New(corpus,
    text.Order(4),               // markov model order, 2-8 (default 4)
    text.Timeout(5*time.Second), // per-call budget (default 10s)
    text.MaxTokens(30),          // includes the seed; longer attempts are rejected
    text.Threads(8),             // parallel attempts (default sequential)
)
if err != nil {
    return err
}
out, err := gen.Generate(ctx,
    text.Seed([]string{"once", "upon"}), // start with these words
    text.WithTimeout(time.Second),       // With* options override the defaults for this call only
)
```

`Order(4)` looks back up to six tokens. Seed words the corpus has never seen are dropped.

Full program: [examples/text](examples/text/main.go). Godoc has the rest of the options, including middle-out seeds.

### Less copying

If your outputs are directly copying source text:

```go
out, err := gen.Generate(ctx,
    text.WithEscape(1),      // try shorter contexts more often, so it branches sooner
    text.WithoutFragments(), // also reject short outputs lifted from inside a message
)
```

On the other hand, `WithoutCopySteering()` lets it copy chunks of the source content but still never returns a verbatim source sentence.

You can run validators if you're using some other way to modify sentences:
```go
out[2] = "cat" // swap a word
if err := gen.Validate(out); err != nil {
    // the edit made it too long, or it now copies the corpus
}
```

## Saving and loading

You can save a model to an `io.Writer`, then load it from an `io.Reader`. This avoids having to build models from corpus every time:

```go
// w is any io.Writer, e.g. an *os.File
if err := gen.Save(w); err != nil {
    return err
}

// later, r is any io.Reader
restored, err := text.Load(r)
if err != nil {
    return err
}
```

This is generally more memory efficient, and so if you build the model from corpus every time, you can call `gen.Compact()` once after you build the model to achieve the same result.

## Why

Markovify was slow, and it didn't give me control over tokenization or validation without overriding its classes. Plus I feel like the memory layout was very bloated and I didn't want to use Python either. So this library stays barebones about both: tokenize your text however you want, validate a sentence however you see fit. If you don't want the chain struct either, fine, there's a `GenerativeChain` interface to satisfy.

## The core API

`text` has the more general purpose API. You can use the core `v2` interfaces when you want a custom token type, your own validator, or direct access to the chain.

Example of using the core API:

```go
chain := barkov.InitChain(4).BuildCompressed(corpus)
out, err := barkov.Gen(ctx, chain, barkov.WithMaxTokens[string](30))
```

The same thing with tokens interned to integer IDs first (less memory, faster builds on large corpora):

```go
vocab := interned.NewVocabulary()
encoded := vocab.InternCorpus(corpus)
chain := interned.Build(4, encoded) // stateSize 1-8
ids, err := barkov.Gen(ctx, chain)
if err != nil {
    return err
}
out := vocab.DecodeTokens(ids)
```

The engine `text` uses by default is a variable-order chain that backs off to shorter contexts where the corpus doesn't have many branches and steers around verbatim reproductions while it walks.

Why not a fixed-order chain? I've noticed that on real text many states have exactly one continuation, so the output ends up being verbatim source text every now and then. By using this variable order, backoff tries the longest context with enough branches, then uses a smaller order when there aren't enough branches from that state.

```go
chain := interned.BuildBackoff(interned.BackoffConfig{}, encoded) // zero value = defaults
ids, err := barkov.Gen(ctx, chain)
```

Unlike `text`, core `Gen` makes one attempt, has no timeout besides your context, and doesn't reject whole messages. `GenIter` does the same but streams tokens as they arrive.

For a custom token type, implement `StateEncoder[T]` (and `AppendEncoder[T]` to reuse encoding buffers) and pass it to `barkov.NewChain(barkov.ChainConfig[T]{...})`. See [examples/custom](examples/custom/main.go). You can find more details about all this in `examples/`.

## Validators

A validator is any `func([]T) bool`. `Gen` calls it on a sliding window as the chain walks. `false` fails the attempt with `ErrSentenceFailedValidation`. This essentially allows you to set any arbitrary validation rule for your markov walk.

```go
noSwears := func(window []string) bool { return !slices.Contains(window, "heck") }
set := barkov.NewNGramSet(corpus, 6, barkov.DefaultStringEncoder)

out, err := barkov.Gen(ctx, chain,
    barkov.WithValidator(noSwears), // slides a StateSize+2 window
    barkov.WithNGramValidator(set), // slides exactly 6 tokens, the set's width
    barkov.WithOutputValidator(func(out []string) bool {
        return len(out) >= 5 // runs once on the whole output
    }),
)
```

Two footguns, and the options that remove them:

- `WithValidator` by default uses a `StateSize+2` sliding window regardless of what your validator expects. If the validator knows its own width (`NGramSet`, `nhash.HashNGramSet`), use `WithNGramValidator` instead, and `Gen` slides exactly `v.N()` tokens.
- The sliding window never sees outputs shorter than itself. `WithOutputValidator` runs once on the complete output and catches those.

`NGramSet` is the stock anti-verbatim validator. `nhash.HashNGramSet` does the same job with 64-bit digests, so it uses less memory but a hash collision can occasionally reject a new n-gram. You don't need either on the backoff engine, since it already steers around copied runs.

Validators are not really for steering, it's more for validation when an error is maybe expected: e.g. profanity filters, length caps, POS patterns, etc.

More in [examples/validators](examples/validators/main.go), including picking a hasher and running attempts in parallel.

## Escape hatches

If you don't like a util we write you can always plug in your own, there's usually just a minimal interface you need to satisfy. `StateEncoder[T]` for state keys, `hashers.Hasher` for digests, `StuckDetector` for retry detection, `SlicePool[T]` for allocation reuse, `GenerativeChain[T]` for the chain itself. The concrete interned chain's extras are reachable through `barkov.RNGSettable` (`SetRNG`) and `barkov.FastMoverKey[K, T]` (`MoveKey`). Godoc has the details.

## Subpackages

- `.../v2/text`: the quick start above. Start here.
- `.../v2`: the core API: `Chain[T]`, `CompressedChain[T]`, `Gen`, `NGramSet`, encoders.
- `.../v2/interned`: `Vocabulary`, `TokenID`, `PackedEncoder`, `Build` (stateSizes 1-8), `BuildBackoff`.
- `.../v2/nhash`: `HashNGramSet[T]`.
- `.../v2/hashers/{xxh3,xxhash64,fnv}`: hashers for `nhash`.
- `.../v2/stuck`: failed-seed cache with TTL expiry on access.

## Performance

Benchmark data lives in `benchstat/`. On a full-length novel (~630k tokens), the interned pipeline runs about 2.9x faster and allocates about two-thirds fewer bytes than the plain string pipeline (`benchstat/steer_snapshot_build_reuse.txt`). Allocations per op are higher on the interned path because interning has fixed per-build setup the string path skips. The trade is more, smaller allocations for fewer bytes and less wall-clock.

## Compatibility

`v1.x` is feature-frozen at `v1.0.3`, and I really recommend moving to v2. `v2.x` is at `github.com/soumitradev/barkov/v2`. On the same corpus, the v2 interned pipeline runs about 4.8x faster than v1 with about one-quarter as many allocated bytes and ~57x fewer allocations per op (`benchstat/v1_vs_v2.txt`). See [MIGRATION.md](MIGRATION.md) for v1 migration notes.
