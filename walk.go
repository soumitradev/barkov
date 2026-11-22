package barkov

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"unsafe"
)

// keyWalker runs walks through a chain's FastMoverKey, skipping state encoding
type keyWalker[T comparable] interface {
	collect(chain GenerativeChain[T]) ([]T, error)
	stream(ctx context.Context, chain GenerativeChain[T], cfg *genConfig[T]) iter.Seq2[T, error]
}

// arrayKeyWalker is zero sized so picking one per call never allocates
type arrayKeyWalker[T, K comparable] struct{}

// Go can't parameterize an interface by an int const, hence a case per N
func keyWalkerFor[T comparable](chain GenerativeChain[T]) keyWalker[T] {
	switch chain.StateSize() {
	case 1:
		return arrayKeyWalkerFor[T, [1]T](chain)
	case 2:
		return arrayKeyWalkerFor[T, [2]T](chain)
	case 3:
		return arrayKeyWalkerFor[T, [3]T](chain)
	case 4:
		return arrayKeyWalkerFor[T, [4]T](chain)
	case 5:
		return arrayKeyWalkerFor[T, [5]T](chain)
	case 6:
		return arrayKeyWalkerFor[T, [6]T](chain)
	case 7:
		return arrayKeyWalkerFor[T, [7]T](chain)
	case 8:
		return arrayKeyWalkerFor[T, [8]T](chain)
	}
	return nil
}

func arrayKeyWalkerFor[T, K comparable](chain GenerativeChain[T]) keyWalker[T] {
	if _, ok := chain.(FastMoverKey[K, T]); !ok {
		return nil
	}
	return arrayKeyWalker[T, K]{}
}

// arrayKey reads the state window as K, which is [len(state)]T by construction
func arrayKey[K, T any](state []T) K {
	return *(*K)(unsafe.Pointer(&state[0]))
}

func (arrayKeyWalker[T, K]) collect(chain GenerativeChain[T]) ([]T, error) {
	fm := chain.(FastMoverKey[K, T])
	sentinels := chain.Sentinels()
	var stateBuf [8]T
	state := stateBuf[:chain.StateSize()]
	fillInitialState(state, nil, nil, sentinels.Begin)

	out := make([]T, 0, 32)
	guard := walkGuard[T]{limit: DefaultMaxTokens}
	for {
		next, err := fm.MoveKey(arrayKey[K](state))
		if err != nil {
			return nil, err
		}
		if next == sentinels.End {
			break
		}
		if err := guard.admit(state, next); err != nil {
			return nil, err
		}
		out = append(out, next)
		shiftState(state, next)
	}
	if len(out) == 0 {
		return nil, nil
	}
	return out, nil
}

func (arrayKeyWalker[T, K]) stream(ctx context.Context, chain GenerativeChain[T], cfg *genConfig[T]) iter.Seq2[T, error] {
	fm := chain.(FastMoverKey[K, T])
	return func(yield func(T, error) bool) {
		var stateBuf [8]T
		state := stateBuf[:chain.StateSize()]
		move := func() (T, error) { return fm.MoveKey(arrayKey[K](state)) }
		walk(ctx, cfg, chain.Sentinels(), state, move, yield)
	}
}

// keyBorrower marks chains whose Move never retains its key, so walks can pass scratch
type keyBorrower interface{ borrowsStateKey() }

func (*Chain[T]) borrowsStateKey()           {}
func (*CompressedChain[T]) borrowsStateKey() {}

func genIterSingle[T comparable](
	ctx context.Context,
	chain GenerativeChain[T],
	cfg *genConfig[T],
) iter.Seq2[T, error] {
	if chain.StateSize() < 1 {
		return errorIter[T](fmt.Errorf("barkov: state size must be positive"))
	}
	if w := keyWalkerFor(chain); w != nil {
		return w.stream(ctx, chain, cfg)
	}
	return func(yield func(T, error) bool) {
		stateSize := chain.StateSize()
		var state []T
		if cfg.pool != nil {
			sp := cfg.pool.GetState()
			state = slices.Grow((*sp)[:0], stateSize)[:stateSize]
			defer func() { *sp = state[:0]; cfg.pool.PutState(sp) }()
		} else {
			state = make([]T, stateSize)
		}

		encoder := chain.Encoder()
		move := func() (T, error) { return chain.Move(encoder.Encode(state)) }
		appendEnc, canAppend := encoder.(AppendEncoder[T])
		if _, borrows := chain.(keyBorrower); canAppend && borrows {
			scratch := make([]byte, 0, 256)
			move = func() (T, error) {
				scratch = appendEnc.AppendEncoded(scratch[:0], state)
				return chain.Move(unsafe.String(unsafe.SliceData(scratch), len(scratch)))
			}
		}
		walk(ctx, cfg, chain.Sentinels(), state, move, yield)
	}
}

// move reads state rather than taking it because an indirect call leaks its
// arguments, which would push a stack-backed state onto the heap
func walk[T comparable](
	ctx context.Context,
	cfg *genConfig[T],
	sentinels Sentinels[T],
	state []T,
	move func() (T, error),
	yield func(T, error) bool,
) {
	guard := walkGuard[T]{limit: cfg.tokenLimit(), validator: cfg.validator, width: cfg.windowWidth(len(state))}
	if guard.validator != nil {
		if cfg.pool != nil {
			hp := cfg.pool.GetGenerated()
			guard.history = (*hp)[:0]
			defer func() { *hp = guard.history[:0]; cfg.pool.PutGenerated(hp) }()
		} else {
			guard.history = make([]T, 0, max(64, guard.width))
		}
	}
	fillInitialState(state, cfg.initialState, cfg.seed, sentinels.Begin)
	if !guard.emitSeed(ctx, cfg.seed, sentinels, yield) {
		return
	}

	var zero T
	done := ctx.Done()
	for {
		if canceled(done) {
			yield(zero, ctx.Err())
			return
		}
		next, err := move()
		if err != nil {
			yield(zero, err)
			return
		}
		if next == sentinels.End {
			return
		}
		if err := guard.admit(state, next); err != nil {
			yield(zero, err)
			return
		}
		if !yield(next, nil) {
			return
		}
		shiftState(state, next)
	}
}

func canceled(done <-chan struct{}) bool {
	if done == nil {
		return false
	}
	select {
	case <-done:
		return true
	default:
		return false
	}
}

func shiftState[T any](state []T, next T) {
	copy(state, state[1:])
	state[len(state)-1] = next
}

// fillInitialState prefers an explicit initial state to the seed
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

// walkGuard stops walks that run too long, cycle or fail the n-gram validator
type walkGuard[T comparable] struct {
	limit     int // 0 for no limit
	emitted   int
	generated int
	cycles    cycleGuard[T]
	validator func([]T) bool
	width     int
	history   []T
}

// emitSeed checks the whole seed against the limit before yielding any of it
func (g *walkGuard[T]) emitSeed(ctx context.Context, seed []T, sentinels Sentinels[T], yield func(T, error) bool) bool {
	var zero T
	if err := ctx.Err(); err != nil {
		yield(zero, err)
		return false
	}
	for _, tok := range seed {
		if tok != sentinels.Begin && tok != sentinels.End {
			g.emitted++
		}
	}
	if g.limit > 0 && g.emitted > g.limit {
		yield(zero, tokenLimitError(g.limit))
		return false
	}
	done := ctx.Done()
	for _, tok := range seed {
		if canceled(done) {
			yield(zero, ctx.Err())
			return false
		}
		if tok == sentinels.Begin || tok == sentinels.End {
			continue
		}
		if g.validator != nil {
			g.history = appendHistory(g.history, tok, g.width)
		}
		if !yield(tok, nil) {
			return false
		}
	}
	return true
}

func (g *walkGuard[T]) admit(state []T, next T) error {
	g.emitted++
	if g.limit > 0 && g.emitted > g.limit {
		return tokenLimitError(g.limit)
	}
	if g.generated++; g.generated > cycleCheckAfter && g.cycles.repeats(state, next) {
		return cycleError()
	}
	if g.validator == nil {
		return nil
	}
	g.history = appendHistory(g.history, next, g.width)
	if len(g.history) >= g.width && !g.validator(g.history[len(g.history)-g.width:]) {
		return fmt.Errorf("barkov: n-gram failed validation: %w", ErrSentenceFailedValidation)
	}
	return nil
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

// Ordinary messages end before the cycle guard starts watching
const cycleCheckAfter = 32

// run[p-1] counts how many tokens in a row matched the token p before them
type cycleGuard[T comparable] struct {
	inline [8]uint8
	extra  []uint8
}

// repeats reports a cycle of period at most the state size that has run for
// three state lengths. By then the walk has left one state the same way
// several times.
func (g *cycleGuard[T]) repeats(state []T, next T) bool {
	n := len(state)
	var run []uint8
	if n <= len(g.inline) {
		run = g.inline[:n]
	} else {
		if len(g.extra) < n {
			g.extra = make([]uint8, n)
		}
		run = g.extra[:n]
	}
	limit := min(3*n, 255)
	lim := uint8(limit)
	for p := 1; p <= n; p++ {
		if state[n-p] != next {
			run[p-1] = 0
			continue
		}
		if run[p-1]++; run[p-1] >= lim {
			return true
		}
	}
	return false
}

func cycleError() error {
	return fmt.Errorf("barkov: %w: %w", ErrWalkCycle, ErrSentenceFailedValidation)
}

func tokenLimitError(limit int) error {
	return fmt.Errorf("barkov: %w (%d): %w", ErrTokenLimit, limit, ErrSentenceFailedValidation)
}

func errorIter[T any](err error) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) { var zero T; yield(zero, err) }
}
