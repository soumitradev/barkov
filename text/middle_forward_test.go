package text_test

import (
	"context"
	"errors"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/interned"
	"github.com/soumitradev/barkov/v2/text"
)

func TestMiddleOutRetriesBackwardOnlyResultForward(t *testing.T) {
	for _, tc := range []struct {
		name      string
		corpus    [][]string
		maxTokens int
		empty     bool
	}{
		{"forward ends at seed", [][]string{{"before", "seed"}, {"before", "seed", "after"}}, 3, true},
		{"forward exceeds limit", [][]string{{"before", "seed", "long", "path"}, {"before", "seed", "after"}}, 2, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g, err := text.New(tc.corpus, text.AntiVerbatim(0), text.Bidirectional(), text.MaxTokens(tc.maxTokens))
			if err != nil {
				t.Fatal(err)
			}
			id, _ := g.Vocab().Lookup("seed")
			// Select a reproducible first draw that cannot continue forward.
			// The reverse chain can always produce "before seed" in this corpus.
			found := false
			for seed := uint64(0); seed < 64; seed++ {
				g.SetRNG(rand.New(rand.NewPCG(seed, 1)))
				first, err := barkov.Gen(context.Background(), g.Chain(), barkov.WithSeed([]interned.TokenID{id}), barkov.WithMaxTokens[interned.TokenID](tc.maxTokens))
				if tc.empty {
					if err != nil || len(first) != 1 {
						continue
					}
				} else if !errors.Is(err, barkov.ErrSentenceFailedValidation) {
					continue
				}
				found = true
				g.SetRNG(rand.New(rand.NewPCG(seed, 1)))
				out, err := g.Generate(context.Background(), text.SeedMiddle([]string{"seed"}), text.WithRetries(1))
				if out != nil || !errors.Is(err, barkov.ErrSentenceFailedValidation) {
					t.Fatalf("returned backward-only text or exceeded one-attempt budget: %v %v", out, err)
				}
				g.SetRNG(rand.New(rand.NewPCG(seed, 1)))
				out, err = g.Generate(context.Background(), text.SeedMiddle([]string{"seed"}), text.WithRetries(32))
				if err != nil || !slices.Equal(out, []string{"seed", "after"}) {
					t.Fatalf("remaining attempts did not switch to forward-only: %v %v", out, err)
				}
				break
			}
			if !found {
				t.Fatal("fixture did not exercise backward-only generation")
			}
		})
	}
}

func TestMiddleOutRejectsTerminalSeed(t *testing.T) {
	g, err := text.New([][]string{{"before", "seed"}}, text.AntiVerbatim(0), text.Bidirectional())
	if err != nil {
		t.Fatal(err)
	}
	for _, threads := range []int{0, 2} {
		for _, retries := range []int{1, 4} {
			out, err := g.Generate(context.Background(), text.SeedMiddle([]string{"seed"}), text.WithRetries(retries), text.WithThreads(threads))
			if out != nil || !errors.Is(err, barkov.ErrSentenceFailedValidation) || !strings.Contains(err.Error(), "forward continuation") {
				t.Fatalf("terminal seed produced backward-only text or an opaque error: %v %v", out, err)
			}
		}
	}
}

func TestMiddleOutForwardFallbackKeepsCopyPrevention(t *testing.T) {
	corpus := [][]string{{"before", "seed", "long", "path"}, {"before", "seed", "after"}, {"seed", "after"}, {"seed", "long", "path"}}
	g, err := text.New(corpus, text.Bidirectional(), text.MaxTokens(2), text.RNG(rand.New(rand.NewPCG(1, 2))))
	if err != nil {
		t.Fatal(err)
	}
	// The long forward path exceeds the cap; the short one copies a whole
	// message. The valid backward half must not conceal either rejection.
	out, err := g.Generate(context.Background(), text.SeedMiddle([]string{"seed"}), text.WithRetries(8))
	if out != nil || !errors.Is(err, barkov.ErrSentenceFailedValidation) {
		t.Fatalf("fallback bypassed copy prevention or token limit: %v %v", out, err)
	}
}
