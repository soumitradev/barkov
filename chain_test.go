package barkov

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

func TestChainBuildRaw(t *testing.T) {
	cfg := ChainConfig[string]{
		StateSize: 2,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Key:       joinKey,
	}
	chain := NewChain(cfg)

	corpus := [][]string{
		{"the", "quick", "brown", "fox"},
		{"the", "lazy", "dog"},
	}
	chain.BuildRaw(corpus)

	// Verify model has expected states
	beginState := joinKey([]string{BEGIN, BEGIN})
	if _, ok := chain.Model[beginState]; !ok {
		t.Errorf("expected begin state %q in model", beginState)
	}

	// "the" should follow the begin state twice (both sentences start with "the")
	if chain.Model[beginState]["the"] != 2 {
		t.Errorf("expected 'the' count of 2 after begin state, got %d", chain.Model[beginState]["the"])
	}

	// "the quick" should lead to "brown"
	theQuickState := joinKey([]string{"the", "quick"})
	if chain.Model[theQuickState]["brown"] != 1 {
		t.Errorf("expected 'brown' after 'the quick'")
	}
}

func TestChainMove(t *testing.T) {
	cfg := ChainConfig[string]{
		StateSize: 2,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Key:       joinKey,
	}
	chain := NewChain(cfg)
	corpus := [][]string{{"a", "b", "c"}}
	chain.BuildRaw(corpus)

	// Valid state should return a token
	state := []string{"a", "b"}
	tok, err := chain.Move(state)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tok != "c" {
		t.Errorf("expected 'c', got %q", tok)
	}

	// Invalid state should return wrapped ErrStateNotFound
	_, err = chain.Move([]string{"nonexistent"})
	if err == nil {
		t.Fatal("expected error for nonexistent state")
	}
	if !errors.Is(err, ErrStateNotFound) {
		t.Errorf("expected ErrStateNotFound, got %v", err)
	}
	// Error message should contain the state
	if !strings.Contains(err.Error(), "nonexistent") {
		t.Errorf("error message should contain the state, got: %s", err.Error())
	}
}

func TestChainWithIntTokens(t *testing.T) {
	// Test that the generic chain works with non-string token types
	cfg := ChainConfig[int]{
		StateSize: 2,
		Sentinels: Sentinels[int]{Begin: -1, End: -2},
		Key:       intKey,
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

// intKey formats states like "[1 2]", which is distinct for distinct int slices
func intKey(state []int) string {
	return fmt.Sprint(state)
}

func TestJoinKeyDistinguishesStates(t *testing.T) {
	states := [][]string{
		{"a", "b"},
		{"ab", ""},
		{"", "ab"},
		{BEGIN, "a"},
		{BEGIN, BEGIN},
	}
	seen := make(map[string][]string)
	for _, state := range states {
		key := joinKey(state)
		if prev, ok := seen[key]; ok {
			t.Errorf("%v and %v share key %q", prev, state, key)
		}
		seen[key] = state
	}
}
