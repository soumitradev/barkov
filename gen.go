package barkov

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"sync"
)

// Stream tokens from a generation attempt as an iter.Seq2
// On first non-nil error the iterator stops yielding
// ctx cancellation is checked between tokens
func GenIter[T comparable](
	ctx context.Context,
	chain Transitioner[T],
	opts ...GenOption[T],
) iter.Seq2[T, error] {
	cfg := &genConfig[T]{}
	for _, opt := range opts {
		opt(cfg)
	}
	if cfg.parallelism > 1 {
		return genIterThreaded(ctx, chain, cfg)
	}
	return genIterSingle(ctx, chain, cfg)
}

// Gen collects a generated sequence into a slice.
func Gen[T comparable](
	ctx context.Context,
	chain Transitioner[T],
	opts ...GenOption[T],
) ([]T, error) {
	if len(opts) == 0 {
		return genSingle(ctx, chain, genConfig[T]{})
	}
	var cfg genConfig[T]
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.parallelism <= 1 {
		return genSingle(ctx, chain, cfg)
	}
	out := make([]T, 0, 32)
	for tok, err := range genIterThreaded(ctx, chain, &cfg) {
		if err != nil {
			return nil, err
		}
		out = append(out, tok)
	}
	return out, nil
}

func initState[T comparable](stateSize int, begin T, seed []T) []T {
	state := make([]T, 0, stateSize)
	for i := 0; i < stateSize-len(seed); i++ {
		state = append(state, begin)
	}
	if len(seed) <= stateSize {
		state = append(state, seed...)
	} else {
		state = append(state, seed[len(seed)-stateSize:]...)
	}
	return state
}

func genSingle[T comparable](
	ctx context.Context,
	chain Transitioner[T],
	cfg genConfig[T],
) ([]T, error) {
	sentinels := chain.Sentinels()
	state := initState(chain.StateSize(), sentinels.Begin, cfg.seed)

	out := make([]T, 0, 32)
	for _, tok := range cfg.seed {
		if tok != sentinels.Begin && tok != sentinels.End {
			out = append(out, tok)
		}
	}

	done := ctx.Done()
	for {
		if done != nil {
			select {
			case <-done:
				return nil, ctx.Err()
			default:
			}
		}

		next, err := chain.Move(state)
		if err != nil {
			return nil, err
		}
		if next == sentinels.End {
			return out, nil
		}
		out = append(out, next)
		if cfg.validator != nil && !cfg.validator(out) {
			return nil, fmt.Errorf("barkov: n-gram failed validation: %w", ErrSentenceFailedValidation)
		}
		copy(state, state[1:])
		state[len(state)-1] = next
	}
}

func genIterSingle[T comparable](
	ctx context.Context,
	chain Transitioner[T],
	cfg *genConfig[T],
) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		sentinels := chain.Sentinels()
		state := initState(chain.StateSize(), sentinels.Begin, cfg.seed)

		var history []T
		if cfg.validator != nil {
			history = make([]T, 0, 64)
		}

		for _, tok := range cfg.seed {
			if tok != sentinels.Begin && tok != sentinels.End {
				if cfg.validator != nil {
					history = append(history, tok)
				}
				if !yield(tok, nil) {
					return
				}
			}
		}

		done := ctx.Done()
		for {
			if done != nil {
				select {
				case <-done:
					var zero T
					yield(zero, ctx.Err())
					return
				default:
				}
			}

			next, err := chain.Move(state)
			if err != nil {
				var zero T
				yield(zero, err)
				return
			}
			if next == sentinels.End {
				return
			}
			if cfg.validator != nil {
				history = append(history, next)
				if !cfg.validator(history) {
					var zero T
					yield(zero, fmt.Errorf("barkov: n-gram failed validation: %w", ErrSentenceFailedValidation))
					return
				}
			}
			if !yield(next, nil) {
				return
			}
			copy(state, state[1:])
			state[len(state)-1] = next
		}
	}
}

// genIterThreaded fans out workers to generate in parallel
// First worker to complete successfully wins
func genIterThreaded[T comparable](
	ctx context.Context,
	chain Transitioner[T],
	cfg *genConfig[T],
) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		type result struct {
			tokens []T
			err    error
		}

		// Create a cancellable context for workers
		workerCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		resultCh := make(chan result, cfg.parallelism)
		var wg sync.WaitGroup

		// Launch workers
		for range cfg.parallelism {
			wg.Go(func() {

				workerCfg := genConfig[T]{
					seed:      cfg.seed,
					validator: cfg.validator,
				}
				tokens, err := genSingle(workerCtx, chain, workerCfg)
				if err != nil {
					select {
					case resultCh <- result{err: err}:
					case <-workerCtx.Done():
					}
					return
				}

				select {
				case resultCh <- result{tokens: tokens}:
				case <-workerCtx.Done():
				}
			})
		}

		// Close result channel when all workers are done
		go func() {
			wg.Wait()
			close(resultCh)
		}()

		// Collect results, return first success
		var lastErr error
		successCount := 0
		failCount := 0

		for res := range resultCh {
			if res.err != nil {
				lastErr = res.err
				failCount++
				// If it's a non-retryable error, cancel other workers
				if errors.Is(res.err, ErrStateNotFound) {
					cancel()
				}
				continue
			}

			// Success! Cancel other workers and replay tokens
			successCount++
			cancel()

			for _, tok := range res.tokens {
				if !yield(tok, nil) {
					return
				}
			}
			return
		}

		// All workers failed
		if lastErr != nil {
			var zero T
			yield(zero, lastErr)
		}
	}
}
