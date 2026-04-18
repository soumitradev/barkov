// Tier 2: interned tokens + hashed validator. Same surface as the
// simple example, but the corpus goes through a Vocabulary (one uint32
// per token, no per-token string overhead), the state map is the
// pointer-free IndexedCompressedChain, and the validator uses xxh3
// hashes instead of full n-gram strings.
//
// stateSize is fixed at 4 because IndexedCompressedChain is a
// stateSize=4 specialisation. For other state sizes use
// interned.InitChain(n) + chain.BuildCompressed(encoded).
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
		strings.Fields("the quick brown fox jumps over the lazy dog today"),
		strings.Fields("the quick brown fox jumps again and runs away fast"),
		strings.Fields("the lazy dog sleeps all day under the warm sun"),
		strings.Fields("the quick fox runs fast and smart across the field"),
		strings.Fields("a quick brown fox is faster than a lazy dog today"),
		strings.Fields("the fox and the dog are good friends in the forest"),
		strings.Fields("the brown dog barks loudly at the quick fox nearby"),
		strings.Fields("a lazy fox sleeps under the warm sun all day long"),
		strings.Fields("a brown dog runs fast across the quiet field at dawn"),
		strings.Fields("the sleepy fox watches the lazy dog under the tree"),
	}

	vocab := interned.NewVocabulary()
	encoded := vocab.InternCorpus(corpus)

	compressed := interned.BuildCompressedIndexed(encoded)

	fmt.Println("Tier 2 — interned + IndexedCompressedChain, no validator:")
	for i := range 5 {
		out, err := barkov.Gen(context.Background(), compressed)
		if err != nil {
			fmt.Printf("%d) [error] %v\n", i+1, err)
			continue
		}
		fmt.Printf("%d) %s\n", i+1, strings.Join(vocab.DecodeTokens(out), " "))
	}

	// Wire up an anti-verbatim validator backed by xxh3 hashes. On a real
	// corpus this rejects generations that reproduce a corpus n-gram
	// verbatim. On a toy corpus like this one, most outputs are verbatim
	// by mathematical necessity, so the validator rejects nearly every
	// attempt — that's expected. WithThreaded fans out and retries.
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
