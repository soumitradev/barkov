# barkov examples

Three runnable programs, one per usage tier.

| Dir | Tier | Shows |
| --- | --- | --- |
| `simple/` | 1. Drop-in | `InitChain(2).BuildCompressed(corpus)` + `Gen` — string chain, no validator, no interning. |
| `optimized/` | 2. Tuned | `interned.InitChain(n)` + `chain.BuildCompressed(encoded)` + `nhash.New(..., xxh3.XXH3{})` validator + `WithThreaded`. Works for any stateSize. A callout at the bottom shows the faster `BuildCompressedIndexedN` specialisation for stateSizes 2–8. |
| `custom/` | 3. Custom | `barkov.NewChain(ChainConfig[int64]{...})` with a user-supplied `StateEncoder[int64]` / `AppendEncoder[int64]`. No strings. |

Each program embeds its own toy corpus and prints 5 sample generations.

```bash
go run ./examples/simple
go run ./examples/optimized
go run ./examples/custom
```

The optimized example runs twice: once without a validator (always succeeds) and once with the anti-verbatim xxh3 validator. On a toy corpus the validator rejects nearly every output because tiny Markov corpora generate verbatim by construction — this is expected and the example calls it out.
