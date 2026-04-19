# v1 → v2 migration

Every public API change from `v1.0.3` to `v2.0.0`. The short version: generics, functional gen options, and a subpackage split.

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
compressed := interned.BuildCompressedIndexed(4, corpus) // stateSize 2..8
```

`BuildCompressedIndexed` now takes `(stateSize, corpus)` and returns a `GenerativeChain[TokenID]`. If you called `interned.BuildCompressedIndexed(encoded)` in an earlier v2 beta, update to `interned.BuildCompressedIndexed(4, encoded)`, or call `BuildCompressedIndexed4(encoded)` directly if you need the concrete `*IndexedCompressedChain4` type (e.g. to reach `SetRNG` or `MoveKey`). The `IndexedCompressedChain` unsuffixed type alias is gone — use `IndexedCompressedChain4`.

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
- `interned.IndexedCompressedChain` type alias — use `IndexedCompressedChain4` for the concrete N=4 type, or take the interface from `BuildCompressedIndexed(stateSize, corpus)`.
