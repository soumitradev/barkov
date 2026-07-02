package interned

import (
	"context"
	"errors"
	"math/rand/v2"
	"testing"

	barkov "github.com/soumitradev/barkov/v2"
)

// repeat returns n copies of msg (sharing the same backing slice; the
// builders only read).
func repeat(msg []string, n int) [][]string {
	out := make([][]string, n)
	for i := range out {
		out[i] = msg
	}
	return out
}

// TestBackoffSelection pins the core backoff rule: an order-3 context
// with Obs < MinSupport is skipped in favor of the order-2 context's
// wider distribution.
//
// Corpus: "x y z a" x2 gives [x y z] Obs=2 with sole follower "a";
// "p y z b" x5 gives [y z] Obs=7 with followers {a:2, b:5}. Under
// MinSupport=3 a walk at window [x y z] must sample from the order-2
// distribution (so "b" appears); under MinSupport=1 the order-3 context
// wins and only "a" is possible.
func TestBackoffSelection(t *testing.T) {
	corpus := append(
		repeat([]string{"x", "y", "z", "a"}, 2),
		repeat([]string{"p", "y", "z", "b"}, 5)...,
	)
	vocab := NewVocabulary()
	encoded := vocab.InternCorpus(corpus)
	window := []TokenID{
		mustLookup(t, vocab, "x"), mustLookup(t, vocab, "y"), mustLookup(t, vocab, "z"),
	}
	tokA := mustLookup(t, vocab, "a")
	tokB := mustLookup(t, vocab, "b")

	draw := func(minSupport int) map[TokenID]int {
		chain := BuildBackoff(BackoffConfig{MaxOrder: 3, MinSupport: minSupport, MaxVerbatim: -1}, encoded)
		chain.(barkov.RNGSettable).SetRNG(rand.New(rand.NewPCG(7, 7)))
		core := chain.(*backoffChain3)
		counts := make(map[TokenID]int)
		for range 200 {
			tok, err := core.moveTail(window)
			if err != nil {
				t.Fatalf("moveTail: %v", err)
			}
			counts[tok]++
		}
		return counts
	}

	// MinSupport=3: [x y z] (Obs=2) is skipped; the order-2 [y z]
	// distribution {a:2, b:5} is sampled, so both followers appear.
	backedOff := draw(3)
	if backedOff[tokA] == 0 || backedOff[tokB] == 0 {
		t.Errorf("MinSupport=3 should sample the order-2 distribution {a,b}, got %v", backedOff)
	}

	// MinSupport=1: [x y z] has enough support; "a" is its only follower.
	direct := draw(1)
	if direct[tokB] != 0 || direct[tokA] != 200 {
		t.Errorf("MinSupport=1 should always emit 'a' from the order-3 context, got %v", direct)
	}
}

func mustLookup(t *testing.T, v *Vocabulary, w string) TokenID {
	t.Helper()
	id, ok := v.Lookup(w)
	if !ok {
		t.Fatalf("token %q not in vocabulary", w)
	}
	return id
}

// TestBackoffReplayNeverErrors walks every corpus position with the
// exact context that precedes it; every window must resolve at some
// order (mid-walk states can never miss all orders).
func TestBackoffReplayNeverErrors(t *testing.T) {
	corpus := [][]string{
		{"the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"},
		{"a", "quick", "brown", "fox", "is", "faster", "than", "a", "lazy", "dog"},
		{"the", "lazy", "dog", "sleeps", "all", "day"},
	}
	vocab := NewVocabulary()
	encoded := vocab.InternCorpus(corpus)
	const maxOrder = 4
	chain := BuildBackoff(BackoffConfig{MaxOrder: maxOrder, MaxVerbatim: -1}, encoded)
	core := chain.(*backoffChain4)

	window := make([]TokenID, maxOrder)
	for _, msg := range encoded {
		for i := range window {
			window[i] = BeginTokenID
		}
		for _, tok := range append(msg, EndTokenID) {
			if _, err := core.moveTail(window); err != nil {
				t.Fatalf("moveTail errored mid-replay at window %v: %v", window, err)
			}
			if tok == EndTokenID {
				break
			}
			copy(window, window[1:])
			window[maxOrder-1] = tok
		}
	}
}

// TestBackoffOutOfVocabSeed pins the error contract: a window whose last
// token was never interned misses even the order-1 table.
func TestBackoffOutOfVocabSeed(t *testing.T) {
	corpus := [][]string{{"a", "b", "c"}}
	vocab := NewVocabulary()
	encoded := vocab.InternCorpus(corpus)
	chain := BuildBackoff(BackoffConfig{MaxOrder: 3, MaxVerbatim: -1}, encoded)
	core := chain.(*backoffChain3)

	_, err := core.moveTail([]TokenID{99999, 99999, 99999})
	if !errors.Is(err, barkov.ErrStateNotFound) {
		t.Errorf("out-of-vocab window should return ErrStateNotFound, got %v", err)
	}
}

// TestBackoffEndToEnd runs the whole pipeline: BuildBackoff with defaults
// (steering disabled until I.4) through unmodified Gen.
func TestBackoffEndToEnd(t *testing.T) {
	corpus := [][]string{
		{"the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"},
		{"the", "quick", "brown", "fox", "runs", "away", "fast"},
		{"a", "lazy", "dog", "sleeps", "all", "day", "long"},
		{"the", "brown", "dog", "barks", "at", "the", "fox"},
	}
	vocab := NewVocabulary()
	encoded := vocab.InternCorpus(corpus)
	chain := BuildBackoff(BackoffConfig{MaxVerbatim: -1}, encoded)

	for range 20 {
		out, err := barkov.Gen(context.Background(), chain)
		if err != nil {
			t.Fatalf("Gen: %v", err)
		}
		if len(out) == 0 {
			t.Fatal("empty output")
		}
		for _, tok := range out {
			if tok == BeginTokenID || tok == EndTokenID {
				t.Fatalf("sentinel leaked into output: %v", out)
			}
		}
	}
}

// TestBackoffConfigValidation pins the panic contract on invalid configs
// and the zero-value defaults.
func TestBackoffConfigValidation(t *testing.T) {
	valid := BackoffConfig{}.withDefaults()
	if valid.MaxOrder != 6 || valid.MinSupport != 3 || valid.MaxVerbatim != 6 {
		t.Errorf("zero-value defaults = %+v, want MaxOrder 6, MinSupport 3, MaxVerbatim 6", valid)
	}

	for _, bad := range []BackoffConfig{
		{MaxOrder: 1},
		{MaxOrder: 9},
		{MinSupport: -1},
		{MinSupport: 65536},
		{MaxVerbatim: 1},
		{MaxVerbatim: 7}, // > default MaxOrder 6
		{MaxVerbatim: -2},
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("config %+v should panic", bad)
				}
			}()
			bad.withDefaults()
		}()
	}
}
