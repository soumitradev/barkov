# barkov examples

Runnable programs: the quick start, one per layer of the API, then two
focused on the parts worth understanding.

| Dir | Shows |
| --- | --- |
| `text/` | `text.New(corpus)` and `gen.Sentence(ctx)`. Start here. |
| `simple/` | `InitChain(2).BuildCompressed(corpus)` + `Gen`, plus streaming tokens with `GenIter` and a `WithSeed`. String chain, no validator, no interning. |
| `optimized/` | Interned tokens + `PackedEncoder` + generic `Chain.BuildCompressed` + `nhash` xxh3 validator + `WithThreaded`. Works for any stateSize; a comment at the bottom of the file shows the faster `interned.Build(stateSize, encoded)` path for stateSizes 1-8. |
| `custom/` | `barkov.NewChain(ChainConfig[int64]{...})` with a user-supplied `StateEncoder[int64]` / `AppendEncoder[int64]`. No strings. |
| `backoff/` | The engine behind `text`: `interned.BuildBackoff` with its MaxOrder / MinSupport / MaxVerbatim knobs, deterministic generation via `SetRNG`, and a side-by-side of steering on vs off. |
| `validators/` | The three validator shapes: a hand-rolled `func([]T) bool` with `WithValidator`, `NGramSet` with width-aware `WithNGramValidator`, and a whole-output rule with `WithOutputValidator`. Plus what retry loops look like. |

Each program embeds its own toy corpus and prints a few sample generations.

```bash
go run ./examples/text
go run ./examples/simple
go run ./examples/optimized
go run ./examples/custom
go run ./examples/backoff
go run ./examples/validators
```

Toy corpora generate verbatim by construction, so the anti-verbatim
machinery rejects a lot in these programs. The validators example leans
into it (expect "all attempts rejected" from the `NGramSet` run) and the
optimized example calls it out when it happens. The text example still
finds non-verbatim sentences on the same kind of corpus because its
backoff engine steers around verbatim continuations during the walk
instead of rejecting whole sentences after the fact.
