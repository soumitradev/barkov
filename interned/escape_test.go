package interned

import (
	"errors"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	barkov "github.com/soumitradev/barkov/v2"
)

func repeatCorpus(parts ...any) [][]string {
	var corpus [][]string
	for i := 0; i < len(parts); i += 2 {
		for range parts[i+1].(int) {
			corpus = append(corpus, strings.Fields(parts[i].(string)))
		}
	}
	return corpus
}

// drawNext samples the step after window n times and returns frequencies.
func drawNext(t *testing.T, chain BackoffChain, v *Vocabulary, window []string, n int) map[string]float64 {
	t.Helper()
	ids := make([]TokenID, len(window))
	for i, w := range window {
		if w == "^" {
			ids[i] = BeginTokenID
		} else {
			ids[i] = mustLookup(t, v, w)
		}
	}
	chain.SetRNG(rand.New(rand.NewPCG(1, 2)))
	state := chain.Encoder().Encode(ids)
	freq := map[string]float64{}
	for range n {
		tok, err := chain.Move(state)
		if err != nil {
			t.Fatal(err)
		}
		name := "END"
		if tok != EndTokenID {
			name = v.Token(tok)
		}
		freq[name] += 1 / float64(n)
	}
	return freq
}

func TestEscapeSamplesTheSteeredMixtureExactly(t *testing.T) {
	// After "q a b": orders 3 and 2 see x×9 and END×1, order 1 sees x×9,
	// END×1 and y×3. Steering forbids x, which completes a corpus 3-gram.
	v := NewVocabulary()
	encoded := v.InternCorpus(repeatCorpus("q a b x", 9, "q a b", 1, "z b y", 3))
	chain := BuildBackoff(BackoffConfig{MaxOrder: 3, MinOrder: 1, MaxVerbatim: 3}, encoded).(BackoffChain).WithEscape(1)
	e := 2.0 / 12 // T/(N+T) at orders 3 and 2
	a3, a2, a1 := 1-e, e*(1-e), e*e
	end, y := a3*0.1+a2*0.1+a1/13, a1*3/13
	want := y / (end + y) // 0.0606; renormalising each order separately gives 0.0208
	got := drawNext(t, chain, v, []string{"q", "a", "b"}, 200000)
	if math.Abs(got["y"]-want) > 0.004 || got["x"] != 0 {
		t.Fatalf("P(y) = %.4f, want %.4f (and never x: %.4f)", got["y"], want, got["x"])
	}
}

func TestEscapeFailsWhenNoContinuationIsAllowed(t *testing.T) {
	v := NewVocabulary()
	encoded := v.InternCorpus(repeatCorpus("q a b x", 2))
	chain := BuildBackoff(BackoffConfig{MaxOrder: 3, MinOrder: 1, MaxVerbatim: 3}, encoded).(BackoffChain).WithEscape(1)
	ids := []TokenID{mustLookup(t, v, "q"), mustLookup(t, v, "a"), mustLookup(t, v, "b")}
	if _, err := chain.Move(chain.Encoder().Encode(ids)); !errors.Is(err, barkov.ErrSentenceFailedValidation) {
		t.Fatalf("every continuation forbidden: %v, want ErrSentenceFailedValidation", err)
	}
}

func TestEscapeCountsStartPaddingOnce(t *testing.T) {
	// At a message start "a" continues with b; elsewhere a also precedes d.
	// Escaping once from the start context reaches [a] with probability 1/2,
	// which then picks d half the time. Escaping through each padded copy of
	// the start context would halve that again.
	v := NewVocabulary()
	encoded := v.InternCorpus(repeatCorpus("a b", 1, "c a d", 1))
	chain := BuildBackoff(BackoffConfig{MaxOrder: 3, MinOrder: 1, MaxVerbatim: -1}, encoded).(BackoffChain).WithEscape(1)
	got := drawNext(t, chain, v, []string{"^", "^", "a"}, 100000)
	if math.Abs(got["d"]-0.25) > 0.01 {
		t.Fatalf("P(d) after a message-start a = %.3f, want 0.25", got["d"])
	}
}

func TestEscapeViewKeepsLongestContextByDefault(t *testing.T) {
	v := NewVocabulary()
	encoded := v.InternCorpus(repeatCorpus("a b", 1, "c a d", 1))
	chain := BuildBackoff(BackoffConfig{MaxOrder: 3, MinOrder: 1, MaxVerbatim: -1}, encoded).(BackoffChain)
	if got := drawNext(t, chain.WithEscape(0), v, []string{"^", "^", "a"}, 2000); got["b"] < 0.9999 {
		t.Fatalf("escape 0 should always use the start context: %v", got)
	}
}

func TestEscapeFindsARareButAllowedContinuation(t *testing.T) {
	// x is forbidden and nearly all the mass; y is allowed but reachable
	// only by escaping twice (probability ~1e-9). Conditioning on steering
	// must still return y rather than give up.
	v := NewVocabulary()
	encoded := v.InternCorpus(repeatCorpus("q a b x", 1000, "z b y", 1))
	chain := BuildBackoff(BackoffConfig{MaxOrder: 3, MinOrder: 1, MaxVerbatim: 3}, encoded).(BackoffChain).WithEscape(1)
	if got := drawNext(t, chain, v, []string{"q", "a", "b"}, 200); got["y"] < 0.9999 {
		t.Fatalf("rare allowed continuation: %v, want always y", got)
	}
}
