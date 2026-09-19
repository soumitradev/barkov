package barkov

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"sync"
)

// genIterThreaded fans out workers to generate in parallel.
// The first worker to complete successfully wins; its tokens are
// replayed through the iterator.
func genIterThreaded[T comparable](
	ctx context.Context,
	chain GenerativeChain[T],
	cfg *genConfig[T],
) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		type result struct {
			tokens []T
			err    error
		}

		encoder := chain.Encoder()
		seedKey := encoder.Encode(cfg.seed)

		// Short-circuit if this seed is known-stuck
		if cfg.stuckCache != nil && cfg.stuckCache.IsStuck(seedKey) {
			var zero T
			yield(zero, fmt.Errorf("barkov: seed is stuck: %w", ErrStateNotFound))
			return
		}

		// Create a cancellable context for workers
		workerCtx, cancel := context.WithCancel(ctx)

		resultCh := make(chan result, cfg.parallelism)
		var wg sync.WaitGroup
		defer func() { cancel(); wg.Wait() }()

		// Launch workers
		for range cfg.parallelism {
			wg.Add(1)
			go func() {
				defer wg.Done()

				// Build a single-threaded config for this worker by
				// copying the whole struct and zeroing parallelism, so
				// validatorWidth, outputValidator, and any future field
				// can't be silently dropped by hand-copying fields.
				workerCfg := *cfg
				workerCfg.parallelism = 0

				tokens, err := collectAttempt(workerCtx, chain, &workerCfg)
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
			}()
		}

		// Close result channel when all workers are done
		go func() {
			wg.Wait()
			close(resultCh)
		}()

		// Collect results, return first success
		var lastErr error

		for res := range resultCh {
			if res.err != nil {
				canceled := errors.Is(res.err, context.Canceled) || errors.Is(res.err, context.DeadlineExceeded)
				if lastErr == nil || !canceled {
					lastErr = res.err
				}
				if cfg.stuckCache != nil && !canceled {
					cfg.stuckCache.RecordFailure(seedKey)
				}
				// Non-retryable: cancel remaining workers
				if errors.Is(res.err, ErrStateNotFound) {
					cancel()
				}
				continue
			}

			// Success
			if cfg.stuckCache != nil {
				cfg.stuckCache.RecordSuccess(seedKey)
			}
			cancel()
			wg.Wait()

			for _, tok := range res.tokens {
				if err := ctx.Err(); err != nil {
					var zero T
					yield(zero, err)
					return
				}
				if !yield(tok, nil) {
					return
				}
			}
			return
		}

		// Caller cancellation must not become an empty success if workers drop their results.
		if ctx.Err() != nil {
			lastErr = ctx.Err()
		}
		// All workers failed
		if lastErr != nil {
			var zero T
			yield(zero, lastErr)
		}
	}
}
