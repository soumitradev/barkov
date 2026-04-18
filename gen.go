package barkov

import (
	"context"
	"fmt"
	"iter"
	"sync"
	"unsafe"
)

// GenIter streams tokens from a generation attempt as an iter.Seq2.
// The second return is an error; on the first non-nil error the iterator
// stops yielding. ctx cancellation is checked between tokens.
func GenIter[T comparable](
	ctx context.Context,
	chain GenerativeChain[T],
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

// Gen collects the iterator into a slice.
func Gen[T comparable](
	ctx context.Context,
	chain GenerativeChain[T],
	opts ...GenOption[T],
) ([]T, error) {
	var out []T
	for tok, err := range GenIter(ctx, chain, opts...) {
		if err != nil {
			return nil, err
		}
		out = append(out, tok)
	}
	return out, nil
}

func genIterSingle[T comparable](
	ctx context.Context,
	chain GenerativeChain[T],
	cfg *genConfig[T],
) iter.Seq2[T, error] {
	// Fast path for chains that implement FastMoverKey[[N]T, T] for their
	// stateSize N ∈ 2..8: bypass encoder.Encode and Move(string) entirely
	// by handing the raw N-token state array to MoveKey directly.
	// Eliminates one string allocation per generated token. Each N needs
	// its own assertion because Go can't parameterize an interface type
	// by an int const.
	switch chain.StateSize() {
	case 2:
		if fm, ok := any(chain).(FastMoverKey[[2]T, T]); ok {
			return genIterSingleFast[T, [2]T](ctx, chain, fm, cfg)
		}
	case 3:
		if fm, ok := any(chain).(FastMoverKey[[3]T, T]); ok {
			return genIterSingleFast[T, [3]T](ctx, chain, fm, cfg)
		}
	case 4:
		if fm, ok := any(chain).(FastMoverKey[[4]T, T]); ok {
			return genIterSingleFast[T, [4]T](ctx, chain, fm, cfg)
		}
	case 5:
		if fm, ok := any(chain).(FastMoverKey[[5]T, T]); ok {
			return genIterSingleFast[T, [5]T](ctx, chain, fm, cfg)
		}
	case 6:
		if fm, ok := any(chain).(FastMoverKey[[6]T, T]); ok {
			return genIterSingleFast[T, [6]T](ctx, chain, fm, cfg)
		}
	case 7:
		if fm, ok := any(chain).(FastMoverKey[[7]T, T]); ok {
			return genIterSingleFast[T, [7]T](ctx, chain, fm, cfg)
		}
	case 8:
		if fm, ok := any(chain).(FastMoverKey[[8]T, T]); ok {
			return genIterSingleFast[T, [8]T](ctx, chain, fm, cfg)
		}
	}
	return func(yield func(T, error) bool) {
		stateSize := chain.StateSize()
		sentinels := chain.Sentinels()
		encoder := chain.Encoder()
		maxOverlap := chain.MaxOverlap()
		// Hoisted: if the encoder supports AppendEncoder, we build the state
		// key each step into a stack scratch buffer instead of allocating a
		// fresh string. The resulting string is only passed to chain.Move,
		// which does a read-only map lookup, so aliasing the scratch bytes
		// via unsafe.String is safe.
		appendEnc, hasAppend := any(encoder).(AppendEncoder[T])
		var scratchBuf [256]byte

		// Allocate state window and history, reusing pool slices when available.
		// history is only read by validator, so skip allocating it otherwise.
		needHistory := cfg.validator != nil
		var state, history []T
		if cfg.pool != nil {
			sp := cfg.pool.GetState()
			state = (*sp)[:0]
			if needHistory {
				hp := cfg.pool.GetGenerated()
				defer func() {
					cfg.pool.PutState(sp)
					cfg.pool.PutGenerated(hp)
				}()
				history = (*hp)[:0]
			} else {
				defer cfg.pool.PutState(sp)
			}
		} else {
			state = make([]T, 0, stateSize)
			if needHistory {
				history = make([]T, 0, 64)
			}
		}

		// Build initial state, padding with Begin tokens if needed
		for i := 0; i < stateSize-len(cfg.seed); i++ {
			state = append(state, sentinels.Begin)
		}
		if len(cfg.seed) <= stateSize {
			state = append(state, cfg.seed...)
		} else {
			// Seed is longer than state size, use the last stateSize tokens
			state = append(state, cfg.seed[len(cfg.seed)-stateSize:]...)
		}

		// Yield seed tokens (without sentinels) up front
		for _, tok := range cfg.seed {
			if tok != sentinels.Begin && tok != sentinels.End {
				if needHistory {
					history = append(history, tok)
				}
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

			var stateKey string
			if hasAppend {
				scratch := appendEnc.AppendEncoded(scratchBuf[:0], state)
				stateKey = unsafe.String(unsafe.SliceData(scratch), len(scratch))
			} else {
				stateKey = encoder.Encode(state)
			}
			next, err := chain.Move(stateKey)
			if err != nil {
				var zero T
				yield(zero, err)
				return
			}
			if next == sentinels.End {
				return
			}

			if needHistory {
				history = append(history, next)
				if len(history) >= maxOverlap {
					gram := history[len(history)-maxOverlap:]
					if !cfg.validator(gram) {
						var zero T
						yield(zero, fmt.Errorf("barkov: n-gram failed validation: %w", ErrSentenceFailedValidation))
						return
					}
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

// genIterSingleFast is the fast path for chains that implement
// FastMoverKey[K, T] with K = [N]T. State lives on the stack as a fixed
// [8]T buffer sliced to stateSize; each step reinterprets that slice as
// a K value (safe because K is always [stateSize]T by construction at
// the dispatch site) and hands it to fm.MoveKey directly. No encoder
// invocation, no string allocation.
func genIterSingleFast[T comparable, K comparable](
	ctx context.Context,
	chain GenerativeChain[T],
	fm FastMoverKey[K, T],
	cfg *genConfig[T],
) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		sentinels := chain.Sentinels()
		maxOverlap := chain.MaxOverlap()
		stateSize := chain.StateSize()

		// history is only consumed by validator; skip it entirely otherwise.
		needHistory := cfg.validator != nil
		var history []T
		if needHistory {
			if cfg.pool != nil {
				hp := cfg.pool.GetGenerated()
				defer cfg.pool.PutGenerated(hp)
				history = (*hp)[:0]
			} else {
				history = make([]T, 0, 64)
			}
		}

		// stateBuf backs the state slice on the stack. 8 is the upper
		// bound on supported stateSize for FastMoverKey; dispatch in
		// genIterSingle only selects this path for N ∈ 2..8.
		var stateBuf [8]T
		state := stateBuf[:stateSize]
		for i := range state {
			state[i] = sentinels.Begin
		}
		seed := cfg.seed
		if len(seed) > stateSize {
			seed = seed[len(seed)-stateSize:]
		}
		copy(state[stateSize-len(seed):], seed)

		for _, tok := range cfg.seed {
			if tok != sentinels.Begin && tok != sentinels.End {
				if needHistory {
					history = append(history, tok)
				}
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

			// &state[0] points at stateBuf[0]; K is [stateSize]T by
			// construction, so reading K bytes from there produces the
			// intended array value.
			next, err := fm.MoveKey(*(*K)(unsafe.Pointer(&state[0])))
			if err != nil {
				var zero T
				yield(zero, err)
				return
			}
			if next == sentinels.End {
				return
			}

			if needHistory {
				history = append(history, next)
				if len(history) >= maxOverlap {
					gram := history[len(history)-maxOverlap:]
					if !cfg.validator(gram) {
						var zero T
						yield(zero, fmt.Errorf("barkov: n-gram failed validation: %w", ErrSentenceFailedValidation))
						return
					}
				}
			}
			if !yield(next, nil) {
				return
			}
			copy(state, state[1:])
			state[stateSize-1] = next
		}
	}
}

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
		defer cancel()

		resultCh := make(chan result, cfg.parallelism)
		var wg sync.WaitGroup

		// Launch workers
		for range cfg.parallelism {
			wg.Add(1)
			go func() {
				defer wg.Done()

				// Build a single-threaded config for this worker
				workerCfg := &genConfig[T]{
					seed:      cfg.seed,
					validator: cfg.validator,
					pool:      cfg.pool,
					// parallelism = 0 for single-threaded
				}

				var tokens []T
				for tok, err := range genIterSingle(workerCtx, chain, workerCfg) {
					if err != nil {
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

		for res := range resultCh {
			if res.err != nil {
				lastErr = res.err
				if cfg.stuckCache != nil {
					cfg.stuckCache.RecordFailure(seedKey)
				}
				// Non-retryable: cancel remaining workers
				if res.err == ErrStateNotFound {
					cancel()
				}
				continue
			}

			// Success
			if cfg.stuckCache != nil {
				cfg.stuckCache.RecordSuccess(seedKey)
			}
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
