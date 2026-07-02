package barkov

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"
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
			a := InitChain(4).BuildRaw(tc.corpus).Compress()
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

func TestPrune(t *testing.T) {
	// "a b c common x" repeats so the "common" branch keeps a >=2 follower
	// and survives the cascade; the single "rare" sentence is the low-count
	// tail that Prune(2) drops. (Cascade behavior is covered separately by
	// TestPruneCascadeFrictionRepro / TestPruneCascadePublicCorpus.)
	corpus := [][]string{
		{"a", "b", "c", "common", "x"},
		{"a", "b", "c", "common", "x"},
		{"a", "b", "c", "common", "x"},
		{"a", "b", "c", "rare", "q"},
	}
	chain := InitChain(3).BuildRaw(corpus)

	stateABC := SepEncoder{Sep: SEP}.Encode([]string{"a", "b", "c"})
	if chain.Model[stateABC]["common"] != 3 || chain.Model[stateABC]["rare"] != 1 {
		t.Fatalf("unexpected pre-prune counts: %v", chain.Model[stateABC])
	}

	chain.Prune(2)

	if _, ok := chain.Model[stateABC]["rare"]; ok {
		t.Errorf("rare transition should be pruned")
	}
	if chain.Model[stateABC]["common"] != 3 {
		t.Errorf("common transition should survive, got %d", chain.Model[stateABC]["common"])
	}

	stateRareQ := SepEncoder{Sep: SEP}.Encode([]string{"b", "c", "rare"})
	if _, ok := chain.Model[stateRareQ]; ok {
		t.Errorf("state with all-pruned transitions should be removed")
	}

	// The surviving common branch leaves no dangling transitions.
	if err := chain.Validate(); err != nil {
		t.Errorf("pruned chain should validate cleanly, got: %v", err)
	}
}

func TestPruneNoop(t *testing.T) {
	corpus := [][]string{{"a", "b", "c", "d"}}
	chain := InitChain(2).BuildRaw(corpus)
	sizeBefore := len(chain.Model)
	chain.Prune(1)
	if len(chain.Model) != sizeBefore {
		t.Errorf("Prune(1) should be a no-op, size changed %d -> %d", sizeBefore, len(chain.Model))
	}
}

func TestChainValidate(t *testing.T) {
	corpus := [][]string{
		{"a", "b", "c", "d"},
		{"a", "b", "c", "e"},
	}
	chain := InitChain(2).BuildRaw(corpus)

	// A freshly built chain has no dangling transitions.
	if err := chain.Validate(); err != nil {
		t.Fatalf("freshly built chain should validate, got: %v", err)
	}

	// Inject a dangling transition by hand: delete a successor state that a
	// surviving transition points at, bypassing Prune's cascade.
	enc := SepEncoder{Sep: SEP}
	ab := enc.Encode([]string{"a", "b"})
	if _, ok := chain.Model[ab]["c"]; !ok {
		t.Fatalf("test precondition: expected (a b)->c transition")
	}
	delete(chain.Model, enc.Encode([]string{"b", "c"})) // successor of (a b) on "c"

	err := chain.Validate()
	if err == nil {
		t.Fatal("expected Validate to report the dangling transition")
	}
	if !errors.Is(err, ErrStateNotFound) {
		t.Errorf("dangling error should wrap ErrStateNotFound, got: %v", err)
	}
	if !containsString(err.Error(), ab) {
		t.Errorf("error should name the offending state, got: %s", err.Error())
	}
}

// TestPruneCascadeFrictionRepro pins friction #3's exact shape: a
// high-count transition whose successor state has only singleton
// followers. Single-pass Prune deletes the successor and leaves the
// transition dangling; the fixed-point cascade removes it instead.
func TestPruneCascadeFrictionRepro(t *testing.T) {
	// (a b)->c is count 5, but its successor (b c) fans out to five
	// singleton followers, all pruned by Prune(2). A separate high-count
	// backbone (z y x w) keeps the begin state generable.
	corpus := [][]string{
		{"a", "b", "c", "d1"},
		{"a", "b", "c", "d2"},
		{"a", "b", "c", "d3"},
		{"a", "b", "c", "d4"},
		{"a", "b", "c", "d5"},
		{"z", "y", "x", "w"},
		{"z", "y", "x", "w"},
		{"z", "y", "x", "w"},
	}
	chain := InitChain(2).BuildRaw(corpus)
	enc := SepEncoder{Sep: SEP}

	ab := enc.Encode([]string{"a", "b"})
	if chain.Model[ab]["c"] != 5 {
		t.Fatalf("precondition: expected (a b)->c count 5, got %d", chain.Model[ab]["c"])
	}

	chain.Prune(2)

	// The cascade must remove the orphaned (a b)->c transition and the
	// emptied (a b) state rather than leaving it dangling.
	if _, ok := chain.Model[ab]; ok {
		t.Errorf("cascade should have removed the orphaned (a b) state")
	}
	if err := chain.Validate(); err != nil {
		t.Errorf("pruned chain should have no dangling transitions, got: %v", err)
	}

	// The begin state survives via the backbone branch.
	begin := enc.Encode([]string{BEGIN, BEGIN})
	if _, ok := chain.Model[begin]; !ok {
		t.Fatal("begin state should survive on the backbone branch")
	}

	// Walking the pruned chain must never fault with ErrStateNotFound.
	compressed := chain.Compress()
	for i := 0; i < 50; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_, err := Gen(ctx, compressed)
		cancel()
		if errors.Is(err, ErrStateNotFound) {
			t.Fatalf("Gen faulted with ErrStateNotFound on a pruned chain: %v", err)
		}
	}
}

// TestPruneCascadePublicCorpus is the property test: on the public corpus,
// after Prune(k) the chain has no dangling transitions (checked two ways:
// Validate and an independent re-walk) and Gen never faults mid-walk.
func TestPruneCascadePublicCorpus(t *testing.T) {
	enc := SepEncoder{Sep: SEP}
	beginKey := enc.Encode([]string{BEGIN, BEGIN, BEGIN, BEGIN})

	for _, minCount := range []uint32{2, 5} {
		t.Run(fmt.Sprintf("minCount=%d", minCount), func(t *testing.T) {
			chain := InitChain(4).BuildRaw(testCorpus)
			chain.Prune(minCount)

			if err := chain.Validate(); err != nil {
				t.Fatalf("Validate reported a dangling transition after Prune(%d): %v", minCount, err)
			}

			// Independent re-walk, not routed through chain.successorKey.
			for state, choices := range chain.Model {
				toks := enc.Decode(state)
				for token := range choices {
					if token == END {
						continue
					}
					succ := enc.Encode(append(append([]string{}, toks[1:]...), token))
					if _, ok := chain.Model[succ]; !ok {
						t.Fatalf("dangling transition from %q on %q -> %q", state, token, succ)
					}
				}
			}

			if _, ok := chain.Model[beginKey]; !ok {
				t.Fatalf("begin state pruned away at minCount=%d; smoke loop needs a generable chain", minCount)
			}
			compressed := chain.Compress()
			for i := 0; i < 50; i++ {
				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
				_, err := Gen(ctx, compressed)
				cancel()
				if errors.Is(err, ErrStateNotFound) {
					t.Fatalf("Gen faulted with ErrStateNotFound after Prune(%d): %v", minCount, err)
				}
			}
		})
	}
}
