package barkov

import (
	"context"
	"fmt"
	"iter"
	"sync"
)

// GenIterGeneric streams tokens from a generation attempt as an iter.Seq2.
// The second return is an error; on the first non-nil error the iterator
// stops yielding. ctx cancellation is checked between tokens.
//
// Will be renamed to GenIter in Phase C.
func GenIterGeneric[T comparable](
	ctx context.Context,
	chain GenericGenerativeChain[T],
	opts ...GenericGenOption[T],
) iter.Seq2[T, error] {
	cfg := &genericGenConfig[T]{}
	for _, opt := range opts {
		opt(cfg)
	}
	if cfg.parallelism > 1 {
		return genIterThreaded(ctx, chain, cfg)
	}
	return genIterSingle(ctx, chain, cfg)
}

// GenGeneric collects the iterator into a slice.
//
// Will be renamed to Gen in Phase C.
func GenGeneric[T comparable](
	ctx context.Context,
	chain GenericGenerativeChain[T],
	opts ...GenericGenOption[T],
) ([]T, error) {
	var out []T
	for tok, err := range GenIterGeneric(ctx, chain, opts...) {
		if err != nil {
			return nil, err
		}
		out = append(out, tok)
	}
	return out, nil
}

func genIterSingle[T comparable](
	ctx context.Context,
	chain GenericGenerativeChain[T],
	cfg *genericGenConfig[T],
) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		stateSize := chain.StateSize()
		sentinels := chain.Sentinels()
		encoder := chain.Encoder()
		maxOverlap := chain.MaxOverlap()

		// Build initial state, padding with Begin tokens if needed
		state := make([]T, 0, stateSize)
		for i := 0; i < stateSize-len(cfg.seed); i++ {
			state = append(state, sentinels.Begin)
		}
		if len(cfg.seed) <= stateSize {
			state = append(state, cfg.seed...)
		} else {
			// Seed is longer than state size, use the last stateSize tokens
			state = append(state, cfg.seed[len(cfg.seed)-stateSize:]...)
		}

		// Track history for validator
		history := make([]T, 0, 64)

		// Yield seed tokens (without sentinels) up front
		for _, tok := range cfg.seed {
			if tok != sentinels.Begin && tok != sentinels.End {
				history = append(history, tok)
				if !yield(tok, nil) {
					return
				}
			}
		}

		for {
			select {
			case <-ctx.Done():
				var zero T
				yield(zero, ctx.Err())
				return
			default:
			}

			next, err := chain.Move(encoder.Encode(state))
			if err != nil {
				var zero T
				yield(zero, err)
				return
			}
			if next == sentinels.End {
				return
			}

			history = append(history, next)
			if cfg.validator != nil && len(history) >= maxOverlap {
				gram := history[len(history)-maxOverlap:]
				if !cfg.validator(gram) {
					var zero T
					yield(zero, fmt.Errorf("barkov: n-gram failed validation: %w", ErrSentenceFailedValidation))
					return
				}
			}
			if !yield(next, nil) {
				return
			}
			// Shift state window
			copy(state, state[1:])
			state[len(state)-1] = next
		}
	}
}

// genIterThreaded fans out workers to generate in parallel.
// The first worker to complete successfully wins; its tokens are
// replayed through the iterator.
func genIterThreaded[T comparable](
	ctx context.Context,
	chain GenericGenerativeChain[T],
	cfg *genericGenConfig[T],
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
			wg.Add(1)
			go func() {
				defer wg.Done()

				// Build a single-threaded config for this worker
				workerCfg := &genericGenConfig[T]{
					seed:      cfg.seed,
					validator: cfg.validator,
					pool:      cfg.pool,
					// parallelism = 0 for single-threaded
				}

				var tokens []T
				for tok, err := range genIterSingle(workerCtx, chain, workerCfg) {
					if err != nil {
						// Check if it's a validation failure (retryable) vs state not found (not retryable)
						select {
						case resultCh <- result{err: err}:
						case <-workerCtx.Done():
						}
						return
					}
					tokens = append(tokens, tok)
				}

				select {
				case resultCh <- result{tokens: tokens}:
				case <-workerCtx.Done():
				}
			}()
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
				if res.err == ErrStateNotFound {
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
