package barkov_test

import (
	"context"
	"errors"
	"slices"
	"testing"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/interned"
)

type genericContextChain struct {
	barkov.GenerativeChain[interned.TokenID]
}

func TestSeedContextIsHistoryNotOutput(t *testing.T) {
	vocab := interned.NewVocabulary()
	chain := interned.BuildBackoff(interned.BackoffConfig{MaxOrder: 3, MinOrder: 2, MaxVerbatim: -1}, vocab.InternCorpus([][]string{{"before", "seed", "after"}}))
	before, _ := vocab.Lookup("before")
	seed, _ := vocab.Lookup("seed")
	history := []interned.TokenID{before, seed}
	for _, c := range []barkov.GenerativeChain[interned.TokenID]{chain, genericContextChain{chain}} {
		for _, threads := range []int{0, 2} {
			out, err := barkov.Gen(context.Background(), c, barkov.WithSeed([]interned.TokenID{seed}), barkov.WithInitialState(history), barkov.WithMaxTokens[interned.TokenID](2), barkov.WithParallelism[interned.TokenID](threads))
			if err != nil || !slices.Equal(vocab.DecodeTokens(out), []string{"seed", "after"}) {
				t.Fatalf("history emitted or counted toward cap: %v %v", out, err)
			}
			if !slices.Equal(history, []interned.TokenID{before, seed}) {
				t.Fatal("history was mutated")
			}
		}
		if _, err := barkov.Gen(context.Background(), c, barkov.WithSeed([]interned.TokenID{seed})); !errors.Is(err, barkov.ErrStateNotFound) {
			t.Fatalf("ordinary sentence-start seed changed: %v", err)
		}
	}
}
