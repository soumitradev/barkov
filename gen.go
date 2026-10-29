package barkov

import (
	"context"
	"fmt"
	"iter"
)

// GenIter streams tokens from a generation attempt as an iter.Seq2.
// The second return is an error; on the first non-nil error the iterator
// stops yielding. ctx cancellation is checked between tokens, including seeds
// and buffered results. Parallel workers finish before the iterator returns.
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
		// genIterThreaded applies cfg.outputValidator inside each worker
		// rather than decorating here: the
		// first worker to finish would otherwise win the race and then
		// fail the whole call instead of letting other workers retry.
		return genIterThreaded(ctx, chain, cfg)
	}
	inner := genIterSingle(ctx, chain, cfg)
	if cfg.outputValidator != nil {
		return validateOutputIter(ctx, inner, cfg.outputValidator)
	}
	return inner
}

// validateOutputIter decorates inner with a whole-output check, without
// touching the hot loops in genIterSingle/genIterSingleFast. It buffers
// inner's entire sequence (seed tokens plus generated, sentinels
// excluded, exactly what Gen would return), then either yields
// ErrSentenceFailedValidation or replays the buffered tokens. This makes
// generation non-streaming by definition: nothing is yielded until inner
// completes.
func validateOutputIter[T comparable](ctx context.Context, inner iter.Seq2[T, error], v func([]T) bool) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		var buf []T
		for tok, err := range inner {
			if err != nil {
				var zero T
				yield(zero, err)
				return
			}
			buf = append(buf, tok)
		}
		if !v(buf) {
			var zero T
			yield(zero, fmt.Errorf("barkov: output failed validation: %w", ErrSentenceFailedValidation))
			return
		}
		for _, tok := range buf {
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
}

// Gen collects the iterator into a slice.
//
// For the common unconfigured call (no seed, validators, threading, or
// stuck detection) on a chain that implements FastMoverKey, Gen skips the
// iterator machinery entirely and runs the walk in a plain loop: same
// MoveKey sequence, same RNG draws, same output, without the per-token
// yield call or the per-call config/closure allocations.
func Gen[T comparable](
	ctx context.Context,
	chain GenerativeChain[T],
	opts ...GenOption[T],
) ([]T, error) {
	if len(opts) == 0 && ctx.Done() == nil {
		switch chain.StateSize() {
		case 1:
			if fm, ok := any(chain).(FastMoverKey[[1]T, T]); ok {
				return genFastCollect[T, [1]T](chain, fm)
			}
		case 2:
			if fm, ok := any(chain).(FastMoverKey[[2]T, T]); ok {
				return genFastCollect[T, [2]T](chain, fm)
			}
		case 3:
			if fm, ok := any(chain).(FastMoverKey[[3]T, T]); ok {
				return genFastCollect[T, [3]T](chain, fm)
			}
		case 4:
			if fm, ok := any(chain).(FastMoverKey[[4]T, T]); ok {
				return genFastCollect[T, [4]T](chain, fm)
			}
		case 5:
			if fm, ok := any(chain).(FastMoverKey[[5]T, T]); ok {
				return genFastCollect[T, [5]T](chain, fm)
			}
		case 6:
			if fm, ok := any(chain).(FastMoverKey[[6]T, T]); ok {
				return genFastCollect[T, [6]T](chain, fm)
			}
		case 7:
			if fm, ok := any(chain).(FastMoverKey[[7]T, T]); ok {
				return genFastCollect[T, [7]T](chain, fm)
			}
		case 8:
			if fm, ok := any(chain).(FastMoverKey[[8]T, T]); ok {
				return genFastCollect[T, [8]T](chain, fm)
			}
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
		out := make([]T, 0, 32)
		for tok, err := range genIterThreaded(ctx, chain, cfg) {
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

	return collectAttempt(ctx, chain, cfg)
}

func collectAttempt[T comparable](ctx context.Context, chain GenerativeChain[T], cfg *genConfig[T]) ([]T, error) {
	out := make([]T, 0, 32)
	for tok, err := range genIterSingle(ctx, chain, cfg) {
		if err != nil {
			return nil, err
		}
		out = append(out, tok)
	}
	if cfg.outputValidator != nil && !cfg.outputValidator(out) {
		return nil, fmt.Errorf("barkov: output failed validation: %w", ErrSentenceFailedValidation)
	}
	if len(out) == 0 {
		return nil, nil
	}
	return out, nil
}
