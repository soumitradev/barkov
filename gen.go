package barkov

import (
	"context"
	"fmt"
	"iter"
)

// GenIter streams one generation attempt, stopping at the first error. ctx is
// checked between tokens, including seeds and buffered results, and parallel
// workers finish before the iterator returns.
func GenIter[T comparable](
	ctx context.Context,
	chain GenerativeChain[T],
	opts ...GenOption[T],
) iter.Seq2[T, error] {
	cfg := &genConfig[T]{}
	for _, opt := range opts {
		opt(cfg)
	}
	if err := cfg.validate(); err != nil {
		return errorIter[T](err)
	}
	if cfg.parallelism > 1 {
		// Workers check outputs themselves so a fast worker's rejected output
		// doesn't win the race and fail the call while others could succeed
		return genIterThreaded(ctx, chain, cfg)
	}
	inner := genIterSingle(ctx, chain, cfg)
	if cfg.outputValidator == nil {
		return inner
	}
	return func(yield func(T, error) bool) {
		out, err := collectAttempt(inner, cfg.outputValidator)
		if err != nil {
			var zero T
			yield(zero, err)
			return
		}
		replay(ctx, out, yield)
	}
}

// Gen collects GenIter into a slice. A call without options on a FastMoverKey
// chain skips the iterator for a plain loop with the same output.
func Gen[T comparable](
	ctx context.Context,
	chain GenerativeChain[T],
	opts ...GenOption[T],
) ([]T, error) {
	if len(opts) == 0 && ctx.Done() == nil {
		if w := keyWalkerFor(chain); w != nil {
			return w.collect(chain)
		}
	}
	cfg := &genConfig[T]{}
	for _, opt := range opts {
		opt(cfg)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}

	if cfg.parallelism > 1 {
		return collect(genIterThreaded(ctx, chain, cfg))
	}
	return collectAttempt(genIterSingle(ctx, chain, cfg), cfg.outputValidator)
}

func collectAttempt[T any](attempt iter.Seq2[T, error], valid func([]T) bool) ([]T, error) {
	out, err := collect(attempt)
	if err != nil {
		return nil, err
	}
	if valid != nil && !valid(out) {
		return nil, fmt.Errorf("barkov: output failed validation: %w", ErrSentenceFailedValidation)
	}
	return out, nil
}

// collect returns nil for an empty sequence
func collect[T any](seq iter.Seq2[T, error]) ([]T, error) {
	out := make([]T, 0, 32)
	for tok, err := range seq {
		if err != nil {
			return nil, err
		}
		out = append(out, tok)
	}
	if len(out) == 0 {
		return nil, nil
	}
	return out, nil
}

func replay[T any](ctx context.Context, tokens []T, yield func(T, error) bool) {
	for _, tok := range tokens {
		if err := ctx.Err(); err != nil {
			var zero T
			yield(zero, err)
			return
		}
		if !yield(tok, nil) {
			return
		}
	}
}
