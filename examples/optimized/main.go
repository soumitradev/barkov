// Tier 2: interned uint32 tokens, PackedEncoder keys, an xxh3-hashed
// anti-verbatim validator, and WithThreaded generation.
//
// For stateSizes 1-8, swap BuildCompressed for interned.Build(stateSize, encoded)
// to use fixed-width array keys directly.
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

	vocab := interned.NewVocabulary()
	chain := barkov.NewChain(barkov.ChainConfig[interned.TokenID]{
		StateSize: 2,
		Sentinels: interned.DefaultSentinels(),
		Encoder:   interned.PackedEncoder{},
	})
	encoded := vocab.InternCorpus(corpus)
	compressed := chain.BuildCompressed(encoded)

	fmt.Println("Tier 2: interned + generic CompressedChain, no validator:")
	for i := range 5 {
		out, err := barkov.Gen(context.Background(), compressed)
		if err != nil {
			fmt.Printf("%d) [error] %v\n", i+1, err)
			continue
		}
		fmt.Printf("%d) %s\n", i+1, strings.Join(vocab.DecodeTokens(out), " "))
	}

	// On a toy corpus most outputs are verbatim, so high rejection is expected.
	// WithThreaded fans out attempts and returns the first that passes.
	validator := nhash.New(
		encoded,
		compressed.StateSize()+2,
		interned.PackedEncoder{},
		xxh3.XXH3{},
	).Validator()

	fmt.Println("\nTier 2: same chain, with xxh3-hashed anti-verbatim validator:")
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
