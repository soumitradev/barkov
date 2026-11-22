// Backoff engine example: interned.BuildBackoff config, deterministic
// SetRNG seeding, and steered vs unsteered output.
package main

import (
	"context"
	"fmt"
	"math/rand/v2"
	"strings"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/interned"
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
	encoded := vocab.InternCorpus(corpus)

	// Low thresholds for the toy corpus; real corpora can use the zero-value
	// defaults (MaxOrder 6, MinOrder 1, MinSupport 1, MaxVerbatim 6).
	chain := interned.BuildBackoff(interned.BackoffConfig{
		MaxOrder:    4,
		MinSupport:  1,
		MaxVerbatim: 3,
	}, encoded)

	fmt.Println("Backoff with steering at window 3:")
	r := 0
	for i := range 3 {
		// Retry past steering dead-ends on this small corpus.
		out, err := barkov.Gen(context.Background(), chain)
		for tries := 0; err != nil && tries < 20; tries++ {
			r++
			out, err = barkov.Gen(context.Background(), chain)
		}
		if err != nil {
			fmt.Printf("%d) [no non-verbatim sentence found] %v\n", i+1, err)
			continue
		}
		fmt.Printf("%d) %s\n", i+1, strings.Join(vocab.DecodeTokens(out), " "))
	}
	fmt.Printf("(took %d retries to produce the 3 sentences above)\n", r)

	// SetRNG makes the generation and retry sequence deterministic.
	genSeeded := func(seed uint64) string {
		for attempt := uint64(0); ; attempt++ {
			chain.(barkov.RNGSettable).SetRNG(rand.New(rand.NewPCG(seed, attempt)))
			if out, err := barkov.Gen(context.Background(), chain); err == nil {
				return strings.Join(vocab.DecodeTokens(out), " ")
			}
		}
	}
	s1, s2 := genSeeded(42), genSeeded(42)
	fmt.Printf("\nSame RNG seed twice:\na) %s\nb) %s\nidentical: %v\n", s1, s2, s1 == s2)

	// MaxVerbatim -1 disables copy steering.
	plain := interned.BuildBackoff(interned.BackoffConfig{
		MaxOrder:    4,
		MinSupport:  1,
		MaxVerbatim: -1,
	}, encoded)
	fmt.Println("\nSame corpus, steering disabled:")
	for i := range 3 {
		out, err := barkov.Gen(context.Background(), plain)
		if err != nil {
			fmt.Printf("%d) [error] %v\n", i+1, err)
			continue
		}
		fmt.Printf("%d) %s\n", i+1, strings.Join(vocab.DecodeTokens(out), " "))
	}
}
