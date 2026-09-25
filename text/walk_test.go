package text_test

import (
	"context"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/soumitradev/barkov/v2/text"
)

func TestSeedEndingRetriesAfterARejectedCompletion(t *testing.T) {
	// The seed always ends its message, so forward generation stops at it.
	// Walking back, "x a seed" repeats a message (rejected) while
	// "y a seed" is new (reachable only by escaping to the shorter [a]
	// context); one rejection must not end seed endings.
	corpus := [][]string{{"x", "a", "seed"}, {"y", "a", "z"}}
	g, err := text.New(corpus, text.Order(2), text.AntiVerbatim(4), text.Bidirectional())
	if err != nil {
		t.Fatal(err)
	}
	for i := range 200 {
		g.SetRNG(rand.New(rand.NewPCG(uint64(i), 3)))
		out, err := g.Generate(context.Background(), text.SeedMiddle([]string{"seed"}), text.WithSeedEnding(), text.WithEscape(1), text.WithRetries(64))
		if err != nil || !slices.Equal(out, []string{"y", "a", "seed"}) {
			t.Fatalf("run %d: %q, %v; want the new seed-ending line", i, out, err)
		}
	}
}

func TestSpliceValidationRejectsRunsCopiedAcrossTheSeed(t *testing.T) {
	// "a seed" and "seed b" are each fine, but joined they copy the corpus
	// 3-gram "a seed b" around the seed, where neither half's steering looks.
	corpus := [][]string{{"a", "seed", "b", "c"}, {"x", "b"}}
	g, err := text.New(corpus, text.Order(2), text.AntiVerbatim(3), text.Bidirectional())
	if err != nil {
		t.Fatal(err)
	}
	copied := func(out []string) bool {
		joined := " " + strings.Join(out, " ") + " "
		for _, m := range corpus {
			for i := 0; i+3 <= len(m); i++ {
				if strings.Contains(joined, " "+strings.Join(m[i:i+3], " ")+" ") {
					return true
				}
			}
		}
		return false
	}
	sawUnchecked := false
	for i := range 300 {
		g.SetRNG(rand.New(rand.NewPCG(uint64(i), 5)))
		if out, err := g.Generate(context.Background(), text.SeedMiddle([]string{"seed"})); err == nil && copied(out) {
			sawUnchecked = true
		}
		g.SetRNG(rand.New(rand.NewPCG(uint64(i), 5)))
		if out, err := g.Generate(context.Background(), text.SeedMiddle([]string{"seed"}), text.WithSpliceValidation()); err == nil && copied(out) {
			t.Fatalf("validated splice copied a corpus 3-gram: %q", out)
		}
	}
	if !sawUnchecked {
		t.Fatal("fixture never produced a copied run across the seed")
	}
}
