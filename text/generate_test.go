package text_test

import (
	"context"
	"math/rand/v2"
	"slices"
	"testing"
	"time"

	"github.com/soumitradev/barkov/v2/text"
)

func TestGenerateRejectsInvalidOverrides(t *testing.T) {
	g, err := text.New(smallCorpus, text.AntiVerbatim(0))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		option text.GenOption
	}{
		{"retries zero", text.WithRetries(0)}, {"retries negative", text.WithRetries(-1)},
		{"threads negative", text.WithThreads(-1)}, {"tokens negative", text.WithMaxTokens(-1)},
		{"timeout negative", text.WithTimeout(-time.Second)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := g.Generate(context.Background(), tc.option)
			if err == nil || out != nil {
				t.Fatalf("invalid option produced output=%v err=%v", out, err)
			}
		})
	}
}

func TestDeterministicBidirectionalGeneration(t *testing.T) {
	g, err := text.New(smallCorpus, text.Order(2), text.AntiVerbatim(0), text.Bidirectional(), text.RNG(rand.New(rand.NewPCG(1, 2))))
	if err != nil {
		t.Fatal(err)
	}
	var previous []string
	for range 30 {
		g.SetRNG(rand.New(rand.NewPCG(1, 2)))
		got, err := g.Generate(context.Background(), text.SeedMiddle([]string{"fox"}))
		if err != nil {
			t.Fatal(err)
		}
		if previous != nil && !slices.Equal(got, previous) {
			t.Fatalf("same RNG produced different middle-out output: %v vs %v", got, previous)
		}
		previous = got
	}
}

func TestFixedRNGRejectsParallelWorkers(t *testing.T) {
	if _, err := text.New(smallCorpus, text.RNG(rand.New(rand.NewPCG(1, 2))), text.Threads(2)); err == nil {
		t.Fatal("constructor accepted shared RNG with workers")
	}
	g, err := text.New(smallCorpus, text.AntiVerbatim(0))
	if err != nil {
		t.Fatal(err)
	}
	g.SetRNG(rand.New(rand.NewPCG(1, 2)))
	if _, err := g.Generate(context.Background(), text.WithThreads(2)); err == nil {
		t.Fatal("per-call override accepted shared RNG with workers")
	}
	g.SetRNG(nil)
	if _, err := g.Generate(context.Background(), text.WithThreads(2)); err != nil {
		t.Fatalf("clearing RNG did not restore threaded generation: %v", err)
	}
}
