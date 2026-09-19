package text_test

import (
	"context"
	"errors"
	"math/rand/v2"
	"testing"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/text"
)

func TestTextSupportDefaultAndExplicitThreshold(t *testing.T) {
	corpus := [][]string{{"hello", "alice", "welcome"}, {"hello", "bob", "welcome"}, {"hello", "carol", "welcome"}}
	for _, support := range []int{0, 1, 3} {
		opts := []text.Option{text.MinOrder(3), text.AntiVerbatim(0), text.Retries(2), text.Bidirectional()}
		if support > 0 {
			opts = append(opts, text.MinSupport(support))
		}
		g, err := text.New(corpus, opts...)
		if err != nil {
			t.Fatal(err)
		}
		for range 20 {
			for _, seed := range [][]string{nil, {"hello", "alice"}} {
				out, err := g.Generate(context.Background(), text.Seed(seed))
				if support == 3 {
					if !errors.Is(err, barkov.ErrInsufficientSupport) || errors.Is(err, barkov.ErrStateNotFound) {
						t.Fatalf("support rejected state misreported: %v %v", out, err)
					}
				} else if err != nil || len(out) != 3 {
					t.Fatalf("default support broke rare context: %v %v", out, err)
				}
			}
		}
		// Reverse seeded generation sees the same threshold: a rare complete
		// seed is observed, but neither its forward nor reverse context qualifies.
		_, err = g.Generate(context.Background(), text.SeedMiddle([]string{"hello", "alice", "welcome"}))
		if support == 3 && !errors.Is(err, barkov.ErrInsufficientSupport) {
			t.Fatalf("reverse support differs: %v", err)
		}
	}
	for _, support := range []int{0, -1, 65536} {
		if _, err := text.New(corpus, text.MinSupport(support)); err == nil {
			t.Fatalf("accepted invalid support %d", support)
		}
	}
}

func TestTextRetriesSupportRejectedCandidates(t *testing.T) {
	corpus := [][]string{{"rare", "path", "fails"}, {"common", "path", "works"}, {"common", "path", "works"}, {"common", "path", "works"}}
	g, err := text.New(corpus, text.MinOrder(3), text.MinSupport(3), text.AntiVerbatim(0))
	if err != nil {
		t.Fatal(err)
	}
	// Find a deterministic RNG that first selects the rare branch. Reusing
	// that stream with more attempts must allow a later supported branch.
	found := false
	for seed := uint64(0); seed < 50; seed++ {
		g.SetRNG(rand.New(rand.NewPCG(seed, 1)))
		if _, err := g.Generate(context.Background(), text.WithRetries(1)); !errors.Is(err, barkov.ErrInsufficientSupport) {
			continue
		}
		found = true
		g.SetRNG(rand.New(rand.NewPCG(seed, 1)))
		out, err := g.Generate(context.Background(), text.WithRetries(32))
		if err != nil || len(out) != 3 || out[0] != "common" {
			t.Fatalf("support rejection was not retried: %v %v", out, err)
		}
		break
	}
	if !found {
		t.Fatal("fixture did not exercise a support rejection")
	}
}
