// Tier 2: the same corpus as examples/simple, but running every lever
// the library exposes for performance.
//
//   - Tokens interned through a Vocabulary (one uint32 per token instead
//     of per-message string headers).
//   - PackedEncoder for state keys (4-byte little-endian per token
//     instead of SEP-delimited string concatenation).
//   - nhash.HashNGramSet validator, keyed on xxh3 hashes instead of
//     full n-gram strings.
//   - WithThreaded to fan out generation attempts and return the first
//     that clears the validator.
//
// stateSize stays configurable — pick whatever matches your use case.
// (If yours happens to be 4, see the callout at the bottom of this
// file for an extra specialisation.)
package main

import (
	"context"
	"fmt"
	"strings"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/hashers/xxh3"
	"github.com/soumitradev/barkov/v2/interned"
	"github.com/soumitradev/barkov/v2/nhash"
)

func main() {
	corpus := [][]string{
		strings.Fields("the quick brown fox jumps over the lazy dog"),
		strings.Fields("the quick brown fox jumps again and again today"),
		strings.Fields("the lazy dog sleeps all day in the warm sun"),
		strings.Fields("the quick fox runs fast and smart across the field"),
		strings.Fields("a quick brown fox is faster than a lazy dog today"),
		strings.Fields("the fox and the dog are good friends in the forest"),
		strings.Fields("the brown dog barks loudly at the quick fox nearby"),
		strings.Fields("a lazy fox sleeps under the warm sun all day long"),
	}

	chain, vocab := interned.InitChain(2)
	encoded := vocab.InternCorpus(corpus)
	compressed := chain.BuildCompressed(encoded)

	fmt.Println("Tier 2 — interned + generic CompressedChain, no validator:")
	for i := range 5 {
		out, err := barkov.Gen(context.Background(), compressed)
		if err != nil {
			fmt.Printf("%d) [error] %v\n", i+1, err)
			continue
		}
		fmt.Printf("%d) %s\n", i+1, strings.Join(vocab.DecodeTokens(out), " "))
	}

	// xxh3-hashed anti-verbatim validator. On a real corpus this rejects
	// generations that reproduce a corpus n-gram verbatim. On a toy corpus
	// most outputs are verbatim by construction, so the validator rejects
	// nearly every attempt — that's expected. WithThreaded retries.
	validator := nhash.New(
		encoded,
		compressed.MaxOverlap(),
		interned.PackedEncoder{},
		xxh3.XXH3{},
	).Validator()

	fmt.Println("\nTier 2 — same chain, with xxh3-hashed anti-verbatim validator:")
	for i := range 5 {
		out, err := barkov.Gen(
			context.Background(),
			compressed,
			barkov.WithValidator(validator),
			barkov.WithThreaded[interned.TokenID](),
		)
		if err != nil {
			fmt.Printf("%d) [validator rejected] %v\n", i+1, err)
			continue
		}
		fmt.Printf("%d) %s\n", i+1, strings.Join(vocab.DecodeTokens(out), " "))
	}
}

// -----------------------------------------------------------------------
// For stateSizes 2–8, swap the build line for
// interned.BuildCompressedIndexedN(encoded) (N matching your stateSize,
// e.g. BuildCompressedIndexed5). BuildCompressedIndexed without a
// suffix is shorthand for N=4. On large corpora these indexed chains
// run roughly 1.7x faster and use about 35% less memory than the
// generic build path (see benchstat/pipeline_simple_vs_maxopt.txt).
// The returned chain satisfies GenerativeChain[TokenID] so everything
// downstream keeps working.
//
//	indexed := interned.BuildCompressedIndexed(encoded) // N=4
//	// indexed := interned.BuildCompressedIndexed5(encoded) // N=5
//	barkov.Gen(ctx, indexed, barkov.WithValidator(validator))
// -----------------------------------------------------------------------
