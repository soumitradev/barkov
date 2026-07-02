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
		// genIterThreaded applies cfg.outputValidator inside each worker
		// (see the worker loop below) rather than decorating here: the
		// first worker to finish would otherwise win the race and then
		// fail the whole call instead of letting other workers retry.
		return genIterThreaded(ctx, chain, cfg)
	}
	inner := genIterSingle(ctx, chain, cfg)
	if cfg.outputValidator != nil {
		return validateOutputIter(inner, cfg.outputValidator)
	}
	return inner
}

// validateOutputIter decorates inner with a whole-output check, without
// touching the hot loops in genIterSingle/genIterSingleFast. It buffers
// inner's entire sequence (seed tokens plus generated, sentinels
// excluded — exactly what Gen would return), then either yields
// ErrSentenceFailedValidation or replays the buffered tokens. This makes
// generation non-streaming by definition: nothing is yielded until inner
// completes.
func validateOutputIter[T comparable](inner iter.Seq2[T, error], v func([]T) bool) iter.Seq2[T, error] {
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
			if !yield(tok, nil) {
				return
			}
		}
	}
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

// fillInitialState writes [begin×pad, seed-tail...] into state. The
// seed tail is the last len(state) tokens of seed (or all of seed if
// shorter). state must already have its final length.
func fillInitialState[T comparable](state []T, seed []T, begin T) {
	for i := range state {
		state[i] = begin
	}
	if len(seed) > len(state) {
		seed = seed[len(seed)-len(state):]
	}
	copy(state[len(state)-len(seed):], seed)
}

// emitSeed yields every non-sentinel token from cfg.seed through yield,
// appending to history along the way when history tracking is active.
// Returns the updated history plus true if the iterator should keep
// going, or the possibly-modified history and false if the consumer
// stopped yielding early.
func emitSeed[T comparable](
	seed []T,
	sentinels Sentinels[T],
	history []T,
	needHistory bool,
	yield func(T, error) bool,
) ([]T, bool) {
	for _, tok := range seed {
		if tok == sentinels.Begin || tok == sentinels.End {
			continue
		}
		if needHistory {
			history = append(history, tok)
		}
		if !yield(tok, nil) {
			return history, false
		}
	}
	return history, true
}

func genIterSingle[T comparable](
	ctx context.Context,
	chain GenerativeChain[T],
	cfg *genConfig[T],
) iter.Seq2[T, error] {
	// Fast path for chains that implement FastMoverKey[[N]T, T] for their
	// stateSize N ∈ 1..8: bypass encoder.Encode and Move(string) entirely
	// by handing the raw N-token state array to MoveKey directly.
	// Eliminates one string allocation per generated token. Each N needs
	// its own assertion because Go can't parameterize an interface type
	// by an int const.
	switch chain.StateSize() {
	case 1:
		if fm, ok := any(chain).(FastMoverKey[[1]T, T]); ok {
			return genIterSingleFast[T, [1]T](ctx, chain, fm, cfg)
		}
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
		maxOverlap := cfg.validatorWidth
		if maxOverlap == 0 {
			maxOverlap = stateSize + 2
		}
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
			state = (*sp)[:stateSize]
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
			state = make([]T, stateSize)
			if needHistory {
				history = make([]T, 0, 64)
			}
		}

		fillInitialState(state, cfg.seed, sentinels.Begin)

		var keepGoing bool
		history, keepGoing = emitSeed(cfg.seed, sentinels, history, needHistory, yield)
		if !keepGoing {
			return
		}

		// See genIterSingleFast for the rationale on hoisting ctx.Done().
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
		stateSize := chain.StateSize()
		maxOverlap := cfg.validatorWidth
		if maxOverlap == 0 {
			maxOverlap = stateSize + 2
		}

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
		// genIterSingle only selects this path for N ∈ 1..8.
		var stateBuf [8]T
		state := stateBuf[:stateSize]
		fillInitialState(state, cfg.seed, sentinels.Begin)

		var keepGoing bool
		history, keepGoing = emitSeed(cfg.seed, sentinels, history, needHistory, yield)
		if !keepGoing {
			return
		}

		// Hoist ctx.Done() out of the per-token loop. context.Background()
		// (and any context with no deadline/cancellation) returns a nil
		// channel from Done(); receiving from nil blocks forever, so the
		// select's default arm always wins — but the select call itself
		// still runs every token. When done == nil we skip it entirely,
		// which is the common case (and every gen benchmark). When a real
		// cancellable context is in play, done != nil and we keep the
		// per-token select for responsiveness.
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

				// Build a single-threaded config for this worker by
				// copying the whole struct and zeroing parallelism, so
				// validatorWidth, outputValidator, and any future field
				// can't be silently dropped by hand-copying fields.
				workerCfg := *cfg
				workerCfg.parallelism = 0

				var tokens []T
				for tok, err := range genIterSingle(workerCtx, chain, &workerCfg) {
					if err != nil {
						select {
						case resultCh <- result{err: err}:
						case <-workerCtx.Done():
						}
						return
					}
					tokens = append(tokens, tok)
				}

				// Run the output validator here, inside the worker, so a
				// rejected candidate counts as a worker failure and other
				// workers keep trying instead of the whole call failing.
				if cfg.outputValidator != nil && !cfg.outputValidator(tokens) {
					select {
					case resultCh <- result{err: fmt.Errorf("barkov: output failed validation: %w", ErrSentenceFailedValidation)}:
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
