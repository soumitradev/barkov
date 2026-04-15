package barkov

import (
	"context"
	"errors"
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
