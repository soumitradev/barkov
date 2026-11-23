package barkov

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"strconv"
	"sync"
)

// genIterThreaded races cfg.parallelism attempts and replays the first success
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

		var seedKey string
		if cfg.stuckCache != nil {
			seedKey = stuckKey(chain.Encoder(), cfg)
		}

		if cfg.stuckCache != nil && cfg.stuckCache.IsStuck(seedKey) {
			var zero T
			yield(zero, fmt.Errorf("barkov: seed is stuck: %w", ErrStateNotFound))
			return
		}

		workerCtx, cancel := context.WithCancel(ctx)

		resultCh := make(chan result, cfg.parallelism)
		var wg sync.WaitGroup
		defer func() { cancel(); wg.Wait() }()

		for range cfg.parallelism {
			wg.Add(1)
			go func() {
				defer wg.Done()
				tokens, err := collectAttempt(genIterSingle(workerCtx, chain, cfg), cfg.outputValidator)
				// Each worker sends once into a buffer sized for all of them, so this never blocks
				resultCh <- result{tokens: tokens, err: err}
			}()
		}

		go func() {
			wg.Wait()
			close(resultCh)
		}()

		var lastErr error
		for res := range resultCh {
			if res.err != nil {
				canceled := errors.Is(res.err, context.Canceled) || errors.Is(res.err, context.DeadlineExceeded)
				if lastErr == nil || !canceled {
					lastErr = res.err
				}
				// A missing state fails every worker the same way
				if errors.Is(res.err, ErrStateNotFound) {
					cancel()
				}
				continue
			}

			if cfg.stuckCache != nil {
				cfg.stuckCache.RecordSuccess(seedKey)
			}
			cancel()
			wg.Wait()

			replay(ctx, res.tokens, yield)
			return
		}

		var zero T
		if ctx.Err() != nil {
			yield(zero, ctx.Err())
			return
		}
		// One failed call is one failure, however many workers it ran
		if cfg.stuckCache != nil {
			cfg.stuckCache.RecordFailure(seedKey)
		}
		yield(zero, lastErr)
	}
}

// stuckKey covers both the seed, which the walk must emit, and the initial
// state it starts from. The seed's length keeps different splits apart.
func stuckKey[T comparable](encoder StateEncoder[T], cfg *genConfig[T]) string {
	seed := encoder.Encode(cfg.seed)
	return strconv.Itoa(len(seed)) + ":" + seed + encoder.Encode(cfg.initialState)
}
