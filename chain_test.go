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
