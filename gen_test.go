package barkov

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func TestGenSingleThreaded(t *testing.T) {
	cfg := ChainConfig[string]{
		StateSize: 2,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Encoder:   SepEncoder{Sep: SEP},
	}
	chain := NewChain(cfg)
	corpus := [][]string{
		{"a", "b", "c"},
		{"a", "b", "d"},
		{"a", "b", "e"},
	}
	chain.BuildRaw(corpus)

	ctx := context.Background()
	result, err := Gen(ctx, chain)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result) == 0 {
		t.Error("expected non-empty result")
	}

	// Result should start with "a" (only option after BEGIN BEGIN)
	if result[0] != "a" {
		t.Errorf("expected first token 'a', got %q", result[0])
	}
}

func TestGenWithSeed(t *testing.T) {
	cfg := ChainConfig[string]{
		StateSize: 2,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Encoder:   SepEncoder{Sep: SEP},
	}
	chain := NewChain(cfg)
	corpus := [][]string{
		{"a", "b", "c"},
	}
	chain.BuildRaw(corpus)

	ctx := context.Background()
	result, err := Gen(ctx, chain, WithSeed([]string{"a", "b"}))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should be ["a", "b", "c"] - seed is yielded then "c" follows
	if len(result) < 3 {
		t.Fatalf("expected at least 3 tokens, got %d: %v", len(result), result)
	}
	if result[0] != "a" || result[1] != "b" || result[2] != "c" {
		t.Errorf("expected [a b c], got %v", result)
	}
}

func TestGenWithValidator(t *testing.T) {
	cfg := ChainConfig[string]{
		StateSize: 2,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Encoder:   SepEncoder{Sep: SEP},
	}
	chain := NewChain(cfg)
	corpus := [][]string{
		{"a", "b", "c", "d"},
	}
	chain.BuildRaw(corpus)

	// Validator that rejects any n-gram containing "c"
	validator := func(gram []string) bool {
		for _, tok := range gram {
			if tok == "c" {
				return false
			}
		}
		return true
	}

	ctx := context.Background()
	_, err := Gen(ctx, chain, WithValidator(validator))
	if err == nil {
		t.Fatal("expected validation error")
	}
	if !errors.Is(err, ErrSentenceFailedValidation) {
		t.Errorf("expected ErrSentenceFailedValidation, got %v", err)
	}
}

func TestGenContextCancellation(t *testing.T) {
	cfg := ChainConfig[string]{
		StateSize: 2,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Encoder:   SepEncoder{Sep: SEP},
	}
	chain := NewChain(cfg)
	corpus := [][]string{
		{"a", "b", "c", "d", "e", "f", "g"},
	}
	chain.BuildRaw(corpus)

	// Cancel the context immediately
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel before starting

	_, err := Gen(ctx, chain)
	if err == nil {
		t.Fatal("expected context error")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func TestGenIterEarlyBreak(t *testing.T) {
	cfg := ChainConfig[string]{
		StateSize: 2,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Encoder:   SepEncoder{Sep: SEP},
	}
	chain := NewChain(cfg)
	corpus := [][]string{
		{"a", "b", "c", "d", "e"},
	}
	chain.BuildRaw(corpus)

	ctx := context.Background()
	count := 0
	for tok, err := range GenIter(ctx, chain) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		count++
		_ = tok
		if count >= 2 {
			break // Early exit
		}
	}

	if count != 2 {
		t.Errorf("expected to iterate 2 times, got %d", count)
	}
}

func TestGenThreaded(t *testing.T) {
	cfg := ChainConfig[string]{
		StateSize: 2,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Encoder:   SepEncoder{Sep: SEP},
	}
	chain := NewChain(cfg)
	corpus := [][]string{
		{"the", "quick", "brown", "fox"},
		{"the", "lazy", "dog"},
		{"a", "b", "c"},
	}
	chain.BuildRaw(corpus)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	result, err := Gen(ctx, chain, WithThreaded[string]())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result) == 0 {
		t.Error("expected non-empty result")
	}
}

func TestGenStateNotFound(t *testing.T) {
	cfg := ChainConfig[string]{
		StateSize: 2,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Encoder:   SepEncoder{Sep: SEP},
	}
	chain := NewChain(cfg)
	// Empty corpus
	chain.BuildRaw([][]string{})

	ctx := context.Background()
	_, err := Gen(ctx, chain)
	if err == nil {
		t.Fatal("expected error for empty model")
	}
	if !errors.Is(err, ErrStateNotFound) {
		t.Errorf("expected ErrStateNotFound, got %v", err)
	}
}

func TestGenWithIntTokens(t *testing.T) {
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

	ctx := context.Background()
	result, err := Gen(ctx, chain)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result) == 0 {
		t.Error("expected non-empty result")
	}

	// First token should be 1 (only option after [-1, -1])
	if result[0] != 1 {
		t.Errorf("expected first token 1, got %d", result[0])
	}
}

// TestGenValidatorSkippedOnShortOutput pins the documented behavior: the
// per-window validator's window is fixed at StateSize+2, so an output
// shorter than that never accumulates enough history to trigger even one
// call. Here StateSize=4 and the (deterministic, single-path) corpus
// produces exactly a 4-token output, which is short of the 6-token window.
func TestGenValidatorSkippedOnShortOutput(t *testing.T) {
	cfg := ChainConfig[string]{
		StateSize: 4,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Encoder:   SepEncoder{Sep: SEP},
	}
	chain := NewChain(cfg)
	chain.BuildRaw([][]string{{"a", "b", "c", "d"}})

	var calls int
	validator := func(gram []string) bool {
		calls++
		return true
	}

	ctx := context.Background()
	result, err := Gen(ctx, chain, WithValidator(validator))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 4 {
		t.Fatalf("expected 4-token output, got %d: %v", len(result), result)
	}
	if calls != 0 {
		t.Errorf("expected per-window validator never called for output shorter than StateSize+2, got %d calls", calls)
	}
}

// TestGenOutputValidator covers WithOutputValidator's contract: called
// exactly once with the complete output, rejection aborts the attempt
// with ErrSentenceFailedValidation, and on the threaded path a rejected
// candidate is a retryable worker failure rather than a call-ending error.
func TestGenOutputValidator(t *testing.T) {
	newChain := func() GenerativeChain[string] {
		cfg := ChainConfig[string]{
			StateSize: 2,
			Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
			Encoder:   SepEncoder{Sep: SEP},
		}
		chain := NewChain(cfg)
		chain.BuildRaw([][]string{{"a", "b", "c"}})
		return chain
	}

	t.Run("CalledOnceWithFullOutput", func(t *testing.T) {
		chain := newChain()
		var calls int
		var seen []string
		outputValidator := func(tokens []string) bool {
			calls++
			seen = append([]string(nil), tokens...)
			return true
		}

		ctx := context.Background()
		result, err := Gen(ctx, chain, WithOutputValidator(outputValidator))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if calls != 1 {
			t.Fatalf("expected output validator called exactly once, got %d", calls)
		}
		if len(seen) != len(result) {
			t.Fatalf("output validator saw %v, Gen returned %v", seen, result)
		}
		for i := range result {
			if seen[i] != result[i] {
				t.Fatalf("output validator saw %v, Gen returned %v", seen, result)
			}
		}
	})

	t.Run("RejectionAbortsWithError", func(t *testing.T) {
		chain := newChain()
		ctx := context.Background()
		_, err := Gen(ctx, chain, WithOutputValidator(func([]string) bool { return false }))
		if err == nil {
			t.Fatal("expected validation error")
		}
		if !errors.Is(err, ErrSentenceFailedValidation) {
			t.Errorf("expected ErrSentenceFailedValidation, got %v", err)
		}
	})

	t.Run("ThreadedRetriesOnRejection", func(t *testing.T) {
		chain := newChain()

		// The chain is deterministic (single-path), so with parallelism=4
		// exactly 4 workers each call outputValidator exactly once. Reject
		// the first three (by arrival order) and accept the fourth: since
		// atomic.AddInt32 hands out 1..4 in call order and cancellation
		// only follows a genuine success, the call that receives 4 is
		// necessarily the last of the four to occur, so no worker is
		// short-circuited before it gets to validate. This proves other
		// workers keep retrying instead of the whole call failing on the
		// first rejection.
		var calls int32
		outputValidator := func(tokens []string) bool {
			return atomic.AddInt32(&calls, 1) > 3
		}

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		result, err := Gen(ctx, chain, WithParallelism[string](4), WithOutputValidator(outputValidator))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(result) != 3 || result[0] != "a" || result[1] != "b" || result[2] != "c" {
			t.Errorf("expected [a b c], got %v", result)
		}
		if got := atomic.LoadInt32(&calls); got != 4 {
			t.Errorf("expected exactly 4 output validator calls (one per worker), got %d", got)
		}
	})
}

// TestWithNGramValidator covers WithNGramValidator's width propagation:
// an NGramSet built with n=4 must actually be probed with 4-token
// windows (not the StateSize+2=6 default, which would never trigger on
// this corpus's 5-token output — the exact silent no-op the old
// hand-wired WithValidator + NGramSet combination suffered from). The
// second subtest pins that the width survives the whole-config copy into
// each threaded worker.
func TestWithNGramValidator(t *testing.T) {
	corpus := [][]string{{"a", "b", "c", "d", "e"}}
	newChain := func() GenerativeChain[string] {
		cfg := ChainConfig[string]{
			StateSize: 4,
			Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
			Encoder:   SepEncoder{Sep: SEP},
		}
		chain := NewChain(cfg)
		chain.BuildRaw(corpus)
		return chain
	}
	// n=4 grams present in the corpus: [a b c d] and [b c d e]. The
	// chain's deterministic walk (a -> b -> c -> d -> e) produces both.
	ngrams := NewNGramSet(corpus, 4, SepEncoder{Sep: SEP})

	t.Run("MatchesConfiguredWidth", func(t *testing.T) {
		chain := newChain()
		ctx := context.Background()
		_, err := Gen(ctx, chain, WithNGramValidator(ngrams))
		if err == nil {
			t.Fatal("expected validation error: n=4 window should match the verbatim 4-gram in the corpus")
		}
		if !errors.Is(err, ErrSentenceFailedValidation) {
			t.Errorf("expected ErrSentenceFailedValidation, got %v", err)
		}
	})

	t.Run("WidthPropagatesThroughThreadedPath", func(t *testing.T) {
		chain := newChain()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_, err := Gen(ctx, chain, WithNGramValidator(ngrams), WithParallelism[string](4))
		if err == nil {
			t.Fatal("expected validation error: validatorWidth should propagate into each worker's config")
		}
		if !errors.Is(err, ErrSentenceFailedValidation) {
			t.Errorf("expected ErrSentenceFailedValidation, got %v", err)
		}
	})
}
