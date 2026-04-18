package barkov

import (
	"errors"
	"fmt"
	"testing"
)

func TestChainBuildRaw(t *testing.T) {
	cfg := ChainConfig[string]{
		StateSize: 2,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Encoder:   SepEncoder{Sep: SEP},
	}
	chain := NewChain(cfg)

	corpus := [][]string{
		{"the", "quick", "brown", "fox"},
		{"the", "lazy", "dog"},
	}
	chain.BuildRaw(corpus)

	// Verify model has expected states
	beginState := SepEncoder{Sep: SEP}.Encode([]string{BEGIN, BEGIN})
	if _, ok := chain.Model[beginState]; !ok {
		t.Errorf("expected begin state %q in model", beginState)
	}

	// "the" should follow the begin state twice (both sentences start with "the")
	if chain.Model[beginState]["the"] != 2 {
		t.Errorf("expected 'the' count of 2 after begin state, got %d", chain.Model[beginState]["the"])
	}

	// "the quick" should lead to "brown"
	theQuickState := SepEncoder{Sep: SEP}.Encode([]string{"the", "quick"})
	if chain.Model[theQuickState]["brown"] != 1 {
		t.Errorf("expected 'brown' after 'the quick'")
	}
}

func TestChainMove(t *testing.T) {
	cfg := ChainConfig[string]{
		StateSize: 2,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Encoder:   SepEncoder{Sep: SEP},
	}
	chain := NewChain(cfg)
	corpus := [][]string{{"a", "b", "c"}}
	chain.BuildRaw(corpus)

	// Valid state should return a token
	state := SepEncoder{Sep: SEP}.Encode([]string{"a", "b"})
	tok, err := chain.Move(state)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tok != "c" {
		t.Errorf("expected 'c', got %q", tok)
	}

	// Invalid state should return wrapped ErrStateNotFound
	_, err = chain.Move("nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent state")
	}
	if !errors.Is(err, ErrStateNotFound) {
		t.Errorf("expected ErrStateNotFound, got %v", err)
	}
	// Error message should contain the state
	if !containsString(err.Error(), "nonexistent") {
		t.Errorf("error message should contain the state, got: %s", err.Error())
	}
}

func TestChainMoveTokens(t *testing.T) {
	cfg := ChainConfig[string]{
		StateSize: 2,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Encoder:   SepEncoder{Sep: SEP},
	}
	chain := NewChain(cfg)
	corpus := [][]string{{"a", "b", "c"}}
	chain.BuildRaw(corpus)

	// MoveTokens should be equivalent to Move(Encode(...))
	tok, err := chain.MoveTokens([]string{"a", "b"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tok != "c" {
		t.Errorf("expected 'c', got %q", tok)
	}
}

func TestChainWithIntTokens(t *testing.T) {
	// Test that the generic chain works with non-string token types
	cfg := ChainConfig[int]{
		StateSize: 2,
		Sentinels: Sentinels[int]{Begin: -1, End: -2},
		Encoder:   intEncoder{},
	}
	chain := NewChain(cfg)

	corpus := [][]int{
		{1, 2, 3},
		{1, 2, 4},
	}
	chain.BuildRaw(corpus)

	// Verify model was built
	if len(chain.Model) == 0 {
		t.Error("expected non-empty model")
	}
}

// intEncoder implements StateEncoder[int] for testing
type intEncoder struct{}

func (intEncoder) Encode(tokens []int) string {
	// Simple encoding: space-separated integers
	if len(tokens) == 0 {
		return ""
	}
	result := make([]byte, 0, len(tokens)*4)
	for i, tok := range tokens {
		if i > 0 {
			result = append(result, ' ')
		}
		result = append(result, []byte(fmt.Sprintf("%d", tok))...)
	}
	return string(result)
}

func (intEncoder) Decode(state string) []int {
	if state == "" {
		return nil
	}
	var result []int
	for _, part := range splitString(state, ' ') {
		var n int
		fmt.Sscanf(part, "%d", &n)
		result = append(result, n)
	}
	return result
}

// TestBuildCompressedEquivalence asserts that the direct BuildCompressed
// path produces structurally equivalent output to the legacy
// BuildRaw().Compress() pipeline. Order within a state group is
// unspecified, so Choices/CumDist are compared as multisets of
// (token, count) pairs derived from the cumulative deltas.
func TestBuildCompressedEquivalence(t *testing.T) {
	cases := []struct {
		name   string
		corpus [][]string
	}{
		{"empty", [][]string{}},
		{"single_long", [][]string{{"the", "quick", "brown", "fox", "jumps", "over"}}},
		{"shorter_than_state", [][]string{{"a"}, {"b", "c"}, {"d", "e", "f", "g"}}},
		{"duplicates", [][]string{
			{"a", "b", "c", "d", "e"},
			{"a", "b", "c", "d", "e"},
			{"a", "b", "c", "d", "e"},
		}},
		{"every_ngram_unique", [][]string{
			{"alpha", "beta", "gamma", "delta", "epsilon"},
			{"zeta", "eta", "theta", "iota", "kappa"},
		}},
		{"every_ngram_same", [][]string{
			{"x", "x", "x", "x", "x", "x", "x", "x"},
		}},
		{"mixed", [][]string{
			{"the", "cat", "sat", "on", "the", "mat"},
			{"the", "dog", "sat", "on", "the", "log"},
			{"a", "b"},
			{"the", "cat", "ran", "away"},
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := InitChain(4).Build(tc.corpus).Compress()
			b := InitChain(4).BuildCompressed(tc.corpus)

			if len(a.Model) != len(b.Model) {
				t.Fatalf("Model size mismatch: Build+Compress=%d, BuildCompressed=%d",
					len(a.Model), len(b.Model))
			}

			for state, aIdx := range a.Model {
				bIdx, ok := b.Model[state]
				if !ok {
					t.Errorf("state %q missing from BuildCompressed model", state)
					continue
				}
				if aIdx.Count != bIdx.Count {
					t.Errorf("state %q: Count mismatch a=%d b=%d", state, aIdx.Count, bIdx.Count)
					continue
				}

				aCounts := extractCounts(a.Choices, a.CumDist, aIdx)
				bCounts := extractCounts(b.Choices, b.CumDist, bIdx)
				if len(aCounts) != len(bCounts) {
					t.Errorf("state %q: distinct-follow count differs a=%d b=%d",
						state, len(aCounts), len(bCounts))
					continue
				}
				for tok, ac := range aCounts {
					if bc := bCounts[tok]; bc != ac {
						t.Errorf("state %q token %q: count mismatch a=%d b=%d",
							state, tok, ac, bc)
					}
				}
			}
		})
	}
}

func extractCounts(choices []string, cumDist []uint32, idx ChoicesIndex) map[string]uint32 {
	out := make(map[string]uint32, idx.Count)
	var prev uint32
	for i := uint32(0); i < uint32(idx.Count); i++ {
		off := idx.Offset + i
		out[choices[off]] = cumDist[off] - prev
		prev = cumDist[off]
	}
	return out
}

func containsString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func splitString(s string, sep byte) []string {
	var result []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == sep {
			result = append(result, s[start:i])
			start = i + 1
		}
	}
	result = append(result, s[start:])
	return result
}
