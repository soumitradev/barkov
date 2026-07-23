# v1 to v2 migration

Every public API change from `v1.0.3` to `v2.0.0`. The short version: generics, functional gen options, and a subpackage split.

If you already migrated to `v2.0.0-beta.6`, the [beta.6 → beta.7 section](#upgrading-within-v2-beta6--beta7) below is the only part you need; the rest is the full v1 → v2 story.

## Upgrading within v2: beta.6 → beta.7

beta.7 fixes three correctness holes reported from real-world use, collapses the interned surface, and adds a sugared `text` package. Most consumers touch only the first two items.

**`go mod tidy` now exits 0.** The `hashers/*` packages used to be separate nested modules, so every consumer's `go mod tidy` walked into packages that don't exist inside the tagged `barkov/v2` module and exited non-zero. They are now folded into the main module. No code change; just re-run `go mod tidy`.

**`text` package: the sugared entry point.** New in beta.7. Strings in, non-parroting sentences out, fastest engine and anti-verbatim wired up for you. Nothing to migrate — this is additive — but it's the recommended starting point for new code.

```go
gen, _ := text.New(corpus) // corpus [][]string
sentence, _ := gen.Sentence(ctx)
```

**Interned surface collapsed to `Build`.** The per-N types and constructors and `interned.InitChain` are gone.

Before:
```go
compressed := interned.BuildCompressedIndexed(4, encoded)
// or the concrete: interned.BuildCompressedIndexed4(encoded)
chain, vocab := interned.InitChain(4)
```
After:
```go
compressed := interned.Build(4, encoded) // stateSize 1..8, returns GenerativeChain[TokenID]
// reach concrete behavior via interface assertion:
compressed.(barkov.RNGSettable).SetRNG(r)
// InitChain's job, spelled out:
vocab := interned.NewVocabulary()
chain := barkov.NewChain(barkov.ChainConfig[interned.TokenID]{
    StateSize: 4, Sentinels: interned.DefaultSentinels(), Encoder: interned.PackedEncoder{},
})
```

**Validator width and short-output holes closed.** `WithValidator` still slides a fixed `StateSize+2` window and skips shorter outputs (now documented). Two new options fix the footguns:

Before:
```go
// silently never matches: Gen slides StateSize+2 grams, the set keys are n-sized
set := barkov.NewNGramSet(corpus, 4, enc) // n != StateSize+2
out, _ := barkov.Gen(ctx, chain, barkov.WithValidator(set.Validator()))
```
After:
```go
// width taken from the validator's N(); short/whole outputs covered too
out, _ := barkov.Gen(ctx, chain,
    barkov.WithNGramValidator(set),               // slides exactly set.N() tokens
    barkov.WithOutputValidator(wholeSentenceOK),  // fires once on the full output
)
```

**`Prune` no longer leaves dangling transitions.** Strictly a bugfix: single-pass `Prune` used to delete emptied states while leaving surviving transitions pointing at them, which faulted `Gen` with `ErrStateNotFound` at runtime. `Prune` now cascades to a fixed point. `Chain[T].Validate() error` is the new opt-in preflight that names the first dangling transition. Aggressive `minCount` can now legitimately cascade away large parts of the chain (including the begin state); check `Validate()` and begin-state presence before `Gen`.

**CompressedChain accessors.** New: `StateTotal(state)`, `States()` iterator, `ChoicesCumDist(state)`, and `barkov.DefaultStringEncoder`. See [Common queries on CompressedChain](#common-queries-on-compressedchain) for the before/after recipes that replace raw index arithmetic.

## Module path

```
- require github.com/soumitradev/barkov v1.0.3
+ require github.com/soumitradev/barkov/v2 v2.0.0
```

Every import needs `/v2` appended:

```bash
grep -rln '"github.com/soumitradev/barkov"' --include='*.go' . \
  | xargs sed -i 's|"github.com/soumitradev/barkov"|"github.com/soumitradev/barkov/v2"|g'
go mod tidy
```

The `sed` recipe is the right tool for a large monorepo with many import sites. For a single-repo migration, dropping the old requirement and re-fetching the new module path is often cleaner — the compiler then points at each remaining import one at a time instead of relying on a blanket rewrite:

```bash
go mod edit -droprequire github.com/soumitradev/barkov
go get github.com/soumitradev/barkov/v2@v2.0.0-beta.6
```

**beta.7 note:** if you're migrating from `v2.0.0-beta.6`, upgrade straight to `v2.0.0-beta.7` — it fixes friction #1 from the beta.6 field report: `hashers/xxh3` and `hashers/xxhash64` used to be separate nested Go modules, so any consumer's `go mod tidy` walked into packages that don't exist inside the tagged `barkov/v2` module and exited non-zero. As of beta.7 those hashers are folded into the main module and `go mod tidy` exits 0 for consumers who only import the root `barkov` package.

## Chain types become generic

`Chain` was a concrete type over `string`. It's now `Chain[T comparable]`, and the old single-type entry point `InitChain(n)` is a convenience wrapper that returns `*Chain[string]`.

v1:
```go
chain := barkov.InitChain(4).Build(corpus).Compress()
```

v2 (Tier 1, unchanged at the call site):
```go
chain := barkov.InitChain(4).Build(corpus).Compress()
// or, skipping the Build→Compress round trip:
chain := barkov.InitChain(4).BuildCompressed(corpus)
```

For non-string chains, use `NewChain`:
```go
chain := barkov.NewChain(barkov.ChainConfig[int64]{
    StateSize: 4,
    Sentinels: barkov.Sentinels[int64]{Begin: -1, End: -2},
    Encoder:   myEncoder{},
})
```

## GenerativeChain gains exported methods

`GenerativeChain` used to require unexported methods (`move`, `getStateSize`, `getMaxOverlap`), which made external implementations impossible. All four required methods are now exported, and `Sentinels` / `Encoder` are added so the interface is self-describing.

v1:
```go
type GenerativeChain interface {
    getMaxOverlap() int
    getStateSize() int
    move(state State) (string, error)
}
```

v2:
```go
type GenerativeChain[T comparable] interface {
    StateSize() int
    Sentinels() Sentinels[T]
    Encoder() StateEncoder[T]
    Move(state string) (T, error)
}
```

`MaxOverlap` is no longer on the interface. Gen computes the validator window as `StateSize()+2` directly, which is what every built-in chain returned anyway. Custom `GenerativeChain` implementations can drop the method.

## Model storage

`map[State]map[string]int` → `map[string]map[T]uint32`. The uncompressed chain still uses a nested map, but counts are `uint32` (saves 4 bytes per transition).

The compressed chain changed shape: `CompressedModel`/`CompressedChoices` are gone. `CompressedChain[T]` now stores flat arrays with an indirection:

v1:
```go
type CompressedChoices struct {
    CumDist []int
    Choices []string
}
type CompressedChain struct {
    Model     map[State]CompressedChoices
    stateSize int
}
```

v2:
```go
type ChoicesIndex struct {
    Offset uint32
    Count  uint16
}
type CompressedChain[T comparable] struct {
    Model   map[string]ChoicesIndex
    Choices []T
    CumDist []uint32
    // ...
}
```

### Common queries on CompressedChain

Custom scoring and state selection used to re-derive the same recipes with raw index arithmetic and a hand-cached encoder. beta.7 adds accessors so you do not have to reach into `Choices`/`CumDist` by hand.

Total observation weight of a state.

Before:
```go
freq := chain.CumDist[uint32(idx.Offset)+uint32(idx.Count)-1]
```
After:
```go
freq, err := chain.StateTotal(state)
```

Follower tokens and their cumulative distribution (for custom samplers).

Before:
```go
choices := chain.Choices[idx.Offset : idx.Offset+uint32(idx.Count)]
cumDist := chain.CumDist[idx.Offset : idx.Offset+uint32(idx.Count)]
```
After:
```go
choices, cumDist, err := chain.ChoicesCumDist(state) // aliases internal storage; read-only
```

Iterate every state as decoded tokens plus its total weight.

Before:
```go
encoder := barkov.SepEncoder{Sep: barkov.SEP}
for state, idx := range chain.Model {
    tokens := encoder.Decode(state)
    total := chain.CumDist[uint32(idx.Offset)+uint32(idx.Count)-1]
    // ...
}
```
After:
```go
for tokens, total := range chain.States() {
    // ...
}
```

When you do need to decode a key by hand, use `barkov.DefaultStringEncoder` instead of allocating `SepEncoder{Sep: SEP}` per call.

## One `Gen` with functional options

All six v1 `Gen*` functions collapse into one `Gen(ctx, chain, opts...)`. Timeouts move to `context.Context`.

v1:
```go
out, err := barkov.Gen(chain)
out, err := barkov.GenWithStart(chain, start)
out, err := barkov.GenPruned(chain, validator)
out, err := barkov.GenPrunedWithStart(chain, start, validator)
out, err := barkov.GenThreaded(chain, validator, 10*time.Second)
out, err := barkov.GenThreadedWithStart(chain, start, validator, 10*time.Second)
```

v2:
```go
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
defer cancel()

out, err := barkov.Gen(ctx, chain)
out, err := barkov.Gen(ctx, chain, barkov.WithSeed(start))
out, err := barkov.Gen(ctx, chain, barkov.WithValidator(validator))
out, err := barkov.Gen(ctx, chain, barkov.WithValidator(validator), barkov.WithSeed(start))
out, err := barkov.Gen(ctx, chain, barkov.WithValidator(validator), barkov.WithThreaded[string]())
out, err := barkov.Gen(ctx, chain, barkov.WithValidator(validator), barkov.WithThreaded[string](), barkov.WithSeed(start))
```

The seed type is `[]T`, not a pre-encoded `State`. Pass raw tokens; the encoder is applied internally.

Additional options: `WithParallelism(n)`, `WithStuckDetector(d)`, `WithSlicePool(p)`.

## Iterator-based generation

New in v2. Streams tokens as an `iter.Seq2[T, error]`:

```go
for tok, err := range barkov.GenIter(ctx, chain, barkov.WithValidator(v)) {
    if err != nil {
        return err
    }
    fmt.Print(tok, " ")
}
```

## Interning moved to a subpackage

v1 had `Vocabulary` and `TokenID` in the root package. v2 moves them to `github.com/soumitradev/barkov/v2/interned`.

v1:
```go
import "github.com/soumitradev/barkov"

vocab := barkov.NewVocabulary()
```

v2:
```go
import "github.com/soumitradev/barkov/v2/interned"

vocab := interned.NewVocabulary()
encoded := vocab.InternCorpus(corpus)
compressed := interned.Build(4, encoded) // stateSize 1..8
```

`interned.Build(stateSize, corpus)` returns a `GenerativeChain[TokenID]` and is the single entry point for stateSizes 1..8. The per-N types and constructors (`IndexedCompressedChain4`, `BuildCompressedIndexed4`, and the `BuildCompressedIndexed` dispatcher) are gone. If you need the concrete affordances, assert for the interface: `compressed.(barkov.RNGSettable).SetRNG(r)` for a deterministic RNG, or `compressed.(barkov.FastMoverKey[[4]interned.TokenID, interned.TokenID])` for direct `MoveKey`.

`interned.InitChain(stateSize)` (the `(*Chain[TokenID], *Vocabulary)` tuple) is also removed. Build the pieces directly:

```go
vocab := interned.NewVocabulary()
chain := barkov.NewChain(barkov.ChainConfig[interned.TokenID]{
    StateSize: 4,
    Sentinels: interned.DefaultSentinels(),
    Encoder:   interned.PackedEncoder{},
})
```

## Stuck detector moved to a subpackage

v1's `StuckCache` (if you used it) is now `stuck.Cache`:

```go
import "github.com/soumitradev/barkov/v2/stuck"

cache := stuck.NewCache()
// pass via: barkov.WithStuckDetector(cache)
```

## Validators

The old "always anti-verbatim" pattern of passing `func([]string) bool` is preserved — any function of that shape works with `WithValidator`.

v2 adds two built-in validator builders:

- `barkov.NGramSet[T]` — string-keyed, in-core, zero external deps.
- `nhash.HashNGramSet[T]` — hash-keyed (uint64), ~50% less memory. Plug in `hashers/xxh3`, `hashers/xxhash64`, or `hashers/fnv`.

```go
import (
    "github.com/soumitradev/barkov/v2/hashers/xxh3"
    "github.com/soumitradev/barkov/v2/nhash"
)

set := nhash.New(corpus, 6, barkov.SepEncoder{Sep: barkov.SEP}, xxh3.XXH3{})
validator := set.Validator()
```

## Encoder abstraction

State encoding was hardcoded in v1 (`strings.Join(state, SEP)`). v2 extracts it behind an interface:

```go
type StateEncoder[T comparable] interface {
    Encode(tokens []T) string
    Decode(state string) []T
}

// Optional fast path:
type AppendEncoder[T comparable] interface {
    AppendEncoded(dst []byte, tokens []T) []byte
}
```

The default for string chains is `SepEncoder{Sep: SEP}` and matches the old behaviour byte for byte. The v1 package-level `ConstructState` / `DeconstructState` helpers are gone; use `SepEncoder{Sep: SEP}.Encode(tokens)` / `.Decode(state)` instead (identical output).

## Sentinels are configurable

v1 used package constants `BEGIN` / `END` hardcoded into every chain. v2 carries them on the chain:

```go
type Sentinels[T comparable] struct {
    Begin T
    End   T
}
```

For string chains `InitChain` picks sensible defaults (`"</BEGIN/>"`, `"</END/>"`). For `NewChain[T]` you supply your own values that can't collide with real tokens.

## Removed

- `barkov.GenWithStart`, `GenPruned`, `GenPrunedWithStart`, `GenThreaded`, `GenThreadedWithStart` — collapsed into `Gen(ctx, chain, opts...)`.
- `Result` struct — unused.
- `State` type alias — it was `string`; v2 uses `string` directly.
- `Chain[T].Build` — was a backcompat alias for `BuildRaw`. Call `BuildRaw` directly.
- `ConstructState` / `DeconstructState` — replaced by `SepEncoder{Sep: SEP}.Encode` / `.Decode`.
- `interned.IndexedCompressedChain*` per-N types, `BuildCompressedIndexed*` constructors, and `interned.InitChain` — collapsed into `interned.Build(stateSize, corpus)`, which returns a `GenerativeChain[TokenID]`. Reach concrete behavior through `barkov.RNGSettable` / `barkov.FastMoverKey` assertions.
