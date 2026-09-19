package interned

import (
	"context"
	"errors"
	"testing"

	barkov "github.com/soumitradev/barkov/v2"
)

func TestMinimumOrderStopsSparseAndSteeredFallback(t *testing.T) {
	corpus := [][]string{{"p", "a", "b", "c"}, {"q", "z", "b", "d"}, {"q", "z", "b", "d"}, {"q", "z", "b", "d"}}
	v := NewVocabulary()
	encoded := v.InternCorpus(corpus)
	window := []TokenID{mustLookup(t, v, "p"), mustLookup(t, v, "a"), mustLookup(t, v, "b")}
	for _, tc := range []struct {
		name              string
		support, verbatim int
		want              error
	}{
		{"steering", 1, 3, barkov.ErrSentenceFailedValidation},
		{"thin evidence", 3, -1, barkov.ErrInsufficientSupport},
	} {
		t.Run(tc.name, func(t *testing.T) {
			permissive := BuildBackoff(BackoffConfig{MaxOrder: 3, MinSupport: tc.support, MaxVerbatim: tc.verbatim}, encoded)
			strict := BuildBackoff(BackoffConfig{MaxOrder: 3, MinOrder: 2, MinSupport: tc.support, MaxVerbatim: tc.verbatim}, encoded)
			state := permissive.Encoder().Encode(window)
			if _, err := permissive.Move(state); err != nil {
				t.Fatalf("default floor should allow fallback: %v", err)
			}
			if _, err := strict.Move(state); !errors.Is(err, tc.want) {
				t.Fatalf("floor bypassed: %v", err)
			}
			// Fast generation uses MoveKey; it must enforce the same floor.
			if _, err := strict.(barkov.FastMoverKey[[3]TokenID, TokenID]).MoveKey([3]TokenID(window)); !errors.Is(err, tc.want) {
				t.Fatalf("fast path bypassed floor: %v", err)
			}
		})
	}
	strict := BuildBackoff(BackoffConfig{MaxOrder: 3, MinOrder: 2, MinSupport: 1, MaxVerbatim: -1}, encoded)
	missing := []TokenID{mustLookup(t, v, "q"), mustLookup(t, v, "p"), mustLookup(t, v, "b")}
	if _, err := strict.Move(strict.Encoder().Encode(missing)); !errors.Is(err, barkov.ErrStateNotFound) {
		t.Fatalf("missing bigram fell back: %v", err)
	}
}

func TestMinimumOrderPreservesPaddedBeginningsAndStateInspection(t *testing.T) {
	v := NewVocabulary()
	encoded := v.InternCorpus([][]string{{"one", "two", "three"}})
	chain := BuildBackoff(BackoffConfig{MaxOrder: 3, MinOrder: 3, MinSupport: 1, MaxVerbatim: -1}, encoded)
	out, err := barkov.Gen(context.Background(), chain)
	if err != nil || len(out) != 3 {
		t.Fatalf("padded initial context failed: %v %v", out, err)
	}
	states := 0
	for range chain.(BackoffChain).States(1) {
		states++
	}
	if states == 0 {
		t.Fatal("lower tables needed for steering/inspection were removed")
	}
}

func TestMinimumOrderValidation(t *testing.T) {
	for _, floor := range []int{-1, 4} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("accepted floor %d for max 3", floor)
				}
			}()
			BuildBackoff(BackoffConfig{MaxOrder: 3, MinOrder: floor}, nil)
		}()
	}
}
