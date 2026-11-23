// Validators: WithValidator, WithNGramValidator, and WithOutputValidator.
// Gen stops an attempt on rejection and leaves retries to the caller.
// (For anti-verbatim on backoff chains, examples/backoff steers natively.)
package main

import (
	"context"
	"fmt"
	"strings"

	barkov "github.com/soumitradev/barkov/v2"
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

	chain := barkov.InitChain(2).BuildCompressed(corpus)
	ctx := context.Background()

	// Retry until validation passes (WithThreaded fans out attempts instead).
	genRetry := func(tries int, opts ...barkov.GenOption[string]) ([]string, int) {
		for r := 0; r < tries; r++ {
			out, err := barkov.Gen(ctx, chain, opts...)
			if err == nil {
				return out, r
			}
		}
		return nil, tries
	}

	// 1. WithValidator slides a StateSize+2 window (4 tokens here) over the walk.
	noBarks := func(gram []string) bool {
		for _, tok := range gram {
			if tok == "barks" {
				return false
			}
		}
		return true
	}
	fmt.Println("Hand-rolled validator (bans 'barks'):")
	for i := range 3 {
		out, rejects := genRetry(20, barkov.WithValidator(noBarks))
		if out == nil {
			fmt.Printf("%d) [all %d attempts rejected]\n", i+1, rejects)
			continue
		}
		fmt.Printf("%d) %s (rejected %d first)\n", i+1, strings.Join(out, " "), rejects)
	}

	// 2. WithNGramValidator slides exactly grams.N() tokens instead of
	// StateSize+2. Expect many rejections on a tiny corpus.
	grams := barkov.NewNGramSet(corpus, 3, barkov.DefaultStringEncoder)
	fmt.Println("\nNGramSet(3) anti-verbatim, width taken from the set:")
	for i := range 3 {
		out, rejects := genRetry(50, barkov.WithNGramValidator(grams))
		if out == nil {
			fmt.Printf("%d) [all %d attempts rejected]\n", i+1, rejects)
			continue
		}
		fmt.Printf("%d) %s (rejected %d first)\n", i+1, strings.Join(out, " "), rejects)
	}

	// 3. WithOutputValidator runs once on the finished token slice.
	longEnough := func(out []string) bool { return len(out) >= 5 }
	fmt.Println("\nWhole-output validator (at least 5 tokens):")
	for i := range 3 {
		out, rejects := genRetry(20, barkov.WithOutputValidator(longEnough))
		if out == nil {
			fmt.Printf("%d) [all %d attempts rejected]\n", i+1, rejects)
			continue
		}
		fmt.Printf("%d) %s (rejected %d first)\n", i+1, strings.Join(out, " "), rejects)
	}

	fmt.Println("\nAll those rejections are the toy corpus talking: tiny corpora generate")
	fmt.Println("verbatim by construction, so generate-reject-retry starves. That starvation")
	fmt.Println("is the whole reason the backoff engine (examples/backoff) exists.")
}
