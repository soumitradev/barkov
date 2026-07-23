// Validators: the three ways to reject output, and when each applies.
// A validator is any func([]T) bool; Gen calls it as the chain walks
// and retries on rejection.
//
// If you came here for anti-verbatim checking only: you don't need any
// of this on the backoff engine (examples/backoff), it steers around
// verbatim continuations by itself. Validators are for everything else,
// and for fixed-order chains.
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

	// Rejection ends an attempt with ErrSentenceFailedValidation. Retry
	// loops are on you (text.Retries is the sugared version); WithThreaded
	// fans attempts out instead.
	genRetry := func(tries int, opts ...barkov.GenOption[string]) ([]string, int) {
		for r := 0; r < tries; r++ {
			out, err := barkov.Gen(ctx, chain, opts...)
			if err == nil {
				return out, r
			}
		}
		return nil, tries
	}

	// 1. A hand-rolled validator with WithValidator. Gen slides a
	// StateSize+2 window (4 tokens here) over the walk. This one bans a
	// word.
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

	// 2. NGramSet with WithNGramValidator. The set declares its own
	// width through N(), so Gen slides exactly that many tokens. (With
	// plain WithValidator you'd silently get StateSize+2 instead, and an
	// n-sized set would never match.) On a toy corpus anti-verbatim
	// rejects nearly everything; expect a high rejection count.
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

	// 3. WithOutputValidator runs once on the finished output, so it
	// catches things a sliding window can't: whole-message reproductions,
	// too-short outputs, global rules.
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
