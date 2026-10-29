package barkov

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"unsafe"
)

// genFastCollect is Gen's plain-config fast path: the state window lives
// in a stack buffer, one MoveKey per token, output pre-sized for a
// typical sentence. Mirrors genIterSingleFast with an empty seed and no
// validator; reaching it requires len(opts)==0, so no option can be
// silently dropped.
func genFastCollect[T comparable, K comparable](
	chain GenerativeChain[T],
	fm FastMoverKey[K, T],
) ([]T, error) {
	sentinels := chain.Sentinels()
	stateSize := chain.StateSize()
	var stateBuf [8]T
	state := stateBuf[:stateSize]
	for i := range state {
		state[i] = sentinels.Begin
	}

	out := make([]T, 0, 32)
	for {
		next, err := fm.MoveKey(*(*K)(unsafe.Pointer(&state[0])))
		if err != nil {
			return nil, err
		}
		if next == sentinels.End {
			if len(out) == 0 {
				return nil, nil // match the iterator path's nil-on-empty
			}
			return out, nil
		}
		out = append(out, next)
		copy(state, state[1:])
		state[stateSize-1] = next
	}
}

// fillInitialState uses an explicit state when supplied, otherwise the seed.
// Only seed tokens are emitted, independently of the history used by the walk.
func fillInitialState[T comparable](state, initial, seed []T, begin T) {
	if initial != nil {
		seed = initial
	}
	for i := range state {
		state[i] = begin
	}
	if len(seed) > len(state) {
		seed = seed[len(seed)-len(state):]
	}
	copy(state[len(state)-len(seed):], seed)
}

// emitSeed validates the complete seed before yielding and checks cancellation
// between seed tokens just as the walk does between generated tokens.
func emitSeed[T comparable](ctx context.Context, cfg *genConfig[T], sentinels Sentinels[T], history []T, width int, yield func(T, error) bool) ([]T, int, bool) {
	var zero T
	if err := ctx.Err(); err != nil {
		yield(zero, err)
		return history, 0, false
	}
	emitted := 0
	for _, tok := range cfg.seed {
		if tok != sentinels.Begin && tok != sentinels.End {
			emitted++
		}
	}
	if cfg.maxTokens > 0 && emitted > cfg.maxTokens {
		yield(zero, tokenLimitError(cfg.maxTokens))
		return history, 0, false
	}
	done := ctx.Done()
	for _, tok := range cfg.seed {
		if done != nil {
			select {
			case <-done:
				yield(zero, ctx.Err())
				return history, emitted, false
			default:
			}
		}
		if tok == sentinels.Begin || tok == sentinels.End {
			continue
		}
		if cfg.validator != nil {
			history = appendHistory(history, tok, width)
		}
		if !yield(tok, nil) {
			return history, emitted, false
		}
	}
	return history, emitted, true
}

// Keep only the history needed by the next validation window, compacting when
// its backing buffer fills. Long streams never retain their complete history.
func appendHistory[T any](history []T, token T, width int) []T {
	if len(history) == cap(history) {
		keep := min(len(history), width-1)
		copy(history, history[len(history)-keep:])
		history = history[:keep]
	}
	return append(history, token)
}

func tokenLimitError(limit int) error {
	return fmt.Errorf("barkov: output exceeded maximum tokens (%d): %w", limit, ErrSentenceFailedValidation)
}

func errorIter[T any](err error) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) { var zero T; yield(zero, err) }
}

func genIterSingle[T comparable](
	ctx context.Context,
	chain GenerativeChain[T],
	cfg *genConfig[T],
) iter.Seq2[T, error] {
	if chain.StateSize() < 1 {
		return errorIter[T](fmt.Errorf("barkov: state size must be positive"))
	}

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
		// Built-in map lookups can borrow the scratch key. Custom chains
		// receive owned strings because Move may retain its argument.
		appendEnc, hasAppend := any(encoder).(AppendEncoder[T])
		switch chain.(type) {
		case *Chain[T], *CompressedChain[T]:
		default:
			hasAppend = false
		}

		var scratchBuf [256]byte

		// Allocate state window and history, reusing pool slices when available.
		// history is only read by validator, so skip allocating it otherwise.
		needHistory := cfg.validator != nil
		var state, history []T
		if cfg.pool != nil {
			sp := cfg.pool.GetState()
			state = slices.Grow((*sp)[:0], stateSize)[:stateSize]
			defer func() { *sp = state[:0]; cfg.pool.PutState(sp) }()
			if needHistory {
				hp := cfg.pool.GetGenerated()
				history = (*hp)[:0]
				defer func() { *hp = history[:0]; cfg.pool.PutGenerated(hp) }()
			}
		} else {
			state = make([]T, stateSize)
			if needHistory {
				history = make([]T, 0, max(64, maxOverlap))
			}
		}

		fillInitialState(state, cfg.initialState, cfg.seed, sentinels.Begin)

		var keepGoing bool
		var emitted int
		history, emitted, keepGoing = emitSeed(ctx, cfg, sentinels, history, maxOverlap, yield)
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

			emitted++
			if cfg.maxTokens > 0 && emitted > cfg.maxTokens {
				var zero T
				yield(zero, tokenLimitError(cfg.maxTokens))
				return
			}

			if needHistory {
				history = appendHistory(history, next, maxOverlap)
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
				defer func() { *hp = history[:0]; cfg.pool.PutGenerated(hp) }()
				history = (*hp)[:0]
			} else {
				history = make([]T, 0, max(64, maxOverlap))
			}
		}

		// stateBuf backs the state slice on the stack. 8 is the upper
		// bound on supported stateSize for FastMoverKey; dispatch in
		// genIterSingle only selects this path for N ∈ 1..8.
		var stateBuf [8]T
		state := stateBuf[:stateSize]
		fillInitialState(state, cfg.initialState, cfg.seed, sentinels.Begin)

		var keepGoing bool
		var emitted int
		history, emitted, keepGoing = emitSeed(ctx, cfg, sentinels, history, maxOverlap, yield)
		if !keepGoing {
			return
		}

		// Hoist ctx.Done() out of the per-token loop. context.Background()
		// (and any context with no deadline/cancellation) returns a nil
		// channel from Done(); receiving from nil blocks forever, so the
		// select's default arm always wins, but the select call itself
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

			emitted++
			if cfg.maxTokens > 0 && emitted > cfg.maxTokens {
				var zero T
				yield(zero, tokenLimitError(cfg.maxTokens))
				return
			}

			if needHistory {
				history = appendHistory(history, next, maxOverlap)
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
