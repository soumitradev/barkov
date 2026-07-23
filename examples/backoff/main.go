// The backoff engine up close: interned.BuildBackoff with its three
// knobs, deterministic output via SetRNG, and what steering actually
// does to the output. This is the engine text.New wires up for you;
// build it yourself when you want to tune it.
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

	// The knobs, set low so a toy corpus can still generate. On real
	// corpora the zero-value config is what you want: MaxOrder 6,
	// MinSupport 3, MaxVerbatim 6.
	//
	//   MaxOrder:    longest context tried (2-8)
	//   MinSupport:  observations a context needs to be trusted
	//   MaxVerbatim: longest verbatim corpus run allowed out (-1 disables)
	chain := interned.BuildBackoff(interned.BackoffConfig{
		MaxOrder:    4,
		MinSupport:  1,
		MaxVerbatim: 3,
	}, encoded)

	fmt.Println("Backoff with steering at window 3:")
	r := 0
	for i := range 3 {
		// Steering dead-ends are common on a toy corpus (text.New retries
		// under the hood for exactly this reason), so retry a few times.
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

	// SetRNG makes generation deterministic: same seed, same sentence.
	// Handy for tests and demos. On this toy corpus steering dead-ends
	// some seeds, so retry like text.New does; the retry sequence itself
	// is identical for a given seed.
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

	// Steering off (MaxVerbatim -1): the same chain happily replays
	// corpus spans. Compare with the steered output above.
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
