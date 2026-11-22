package barkov

import (
	"fmt"
	"math/rand/v2"
	"sort"
)

// Compile-time check that Chain implements GenerativeChain.
var _ GenerativeChain[string] = (*Chain[string])(nil)

// Sentinels bundles the begin/end tokens for a given Token type.
// Users choose values that won't collide with real tokens in their corpus.
type Sentinels[T comparable] struct {
	Begin T
	End   T
}

// ChainConfig holds the dependencies a Chain needs.
// Required: StateSize, Sentinels, Encoder.
type ChainConfig[T comparable] struct {
	StateSize int
	Sentinels Sentinels[T]
	Encoder   StateEncoder[T]
}

// Chain is the generic markov chain type.
type Chain[T comparable] struct {
	Model     map[string]map[T]uint32
	stateSize int
	sentinels Sentinels[T]
	encoder   StateEncoder[T]

	beginChoices []T
	beginCumDist []uint32
}

// NewChain constructs a Chain from a config.
// This is the generic entry point. For the common string case,
// use InitChain(stateSize) which fills in defaults.
func NewChain[T comparable](cfg ChainConfig[T]) *Chain[T] {
	return &Chain[T]{
		Model:     make(map[string]map[T]uint32),
		stateSize: cfg.StateSize,
		sentinels: cfg.Sentinels,
		encoder:   cfg.Encoder,
	}
}

// BuildRaw constructs the chain from a pre-encoded corpus. The corpus
// is already in the target token type T, so no string conversion happens.
// This is the canonical, generic build path.
//
// When the encoder implements AppendEncoder[T], BuildRaw uses a
// zero-allocation scratch buffer and arena-interned map keys, dropping
// per-observation allocations to zero on the hot loop. Custom encoders
// without AppendEncoder fall back to the simple per-call Encode path.
func (c *Chain[T]) BuildRaw(corpus [][]T) *Chain[T] {
	beginSeq := c.beginWindow()
	maxLen, totalObs := corpusShape(corpus)
	items := make([]T, 0, c.stateSize+maxLen+1)

	// totalObs/4 undershoots actual unique states by ~3.3x on typical
	// corpora; size eagerly to skip map rehashes.
	estUnique := totalObs + 64
	if len(c.Model) == 0 {
		c.Model = make(map[string]map[T]uint32, estUnique)
	}
	ks := newKeyStream(c.encoder, estUnique*c.stateSize*8)

	for _, run := range corpus {
		items = items[:0]
		items = append(items, beginSeq...)
		items = append(items, run...)
		items = append(items, c.sentinels.End)

		for i := 0; i < len(run)+1; i++ {
			probeKey := ks.probe(items[i : i+c.stateSize])
			follow := items[i+c.stateSize]
			if inner, ok := c.Model[probeKey]; ok {
				inner[follow]++
				continue
			}
			inner := make(map[T]uint32, 2)
			c.Model[ks.intern(probeKey)] = inner
			inner[follow]++
		}
	}

	c.precomputeBeginState()
	return c
}

// beginWindow returns a state-sized window of Begin sentinels
func (c *Chain[T]) beginWindow() []T {
	w := make([]T, c.stateSize)
	for i := range w {
		w[i] = c.sentinels.Begin
	}
	return w
}

// corpusShape returns the longest run and the number of transitions the
// corpus observes, counting each run's final move to End
func corpusShape[T any](corpus [][]T) (maxLen, totalObs int) {
	for _, run := range corpus {
		maxLen = max(maxLen, len(run))
		totalObs += len(run) + 1
	}
	return maxLen, totalObs
}

func (c *Chain[T]) precomputeBeginState() {
	key := c.encoder.Encode(c.beginWindow())
	if choices, ok := c.Model[key]; ok {
		c.beginChoices, c.beginCumDist = calculateCumDist(choices)
	}
}

// Implements GenerativeChain[T].
func (c *Chain[T]) StateSize() int           { return c.stateSize }
func (c *Chain[T]) Sentinels() Sentinels[T]  { return c.sentinels }
func (c *Chain[T]) Encoder() StateEncoder[T] { return c.encoder }

// Move transitions from a state to a randomly chosen next token.
// Returns ErrStateNotFound (wrapped with the state) if the state doesn't exist.
func (c *Chain[T]) Move(state string) (T, error) {
	choices, ok := c.Model[state]
	if !ok {
		var zero T
		return zero, fmt.Errorf("barkov: state %q not in model: %w", state, ErrStateNotFound)
	}

	// Fast-path for the begin state: reuse precomputed distributions.
	if state == c.encoder.Encode(c.beginWindow()) && len(c.beginChoices) > 0 {
		return c.beginChoices[chooseToken32(c.beginCumDist)], nil
	}

	keys, cumDist := calculateCumDist(choices)
	return keys[chooseToken32(cumDist)], nil
}

// MoveTokens is a convenience wrapper that encodes the state for the caller.
// Equivalent to c.Move(c.Encoder().Encode(tokens)) but saves boilerplate.
// Not part of the GenerativeChain[T] interface.
func (c *Chain[T]) MoveTokens(tokens []T) (T, error) {
	return c.Move(c.encoder.Encode(tokens))
}

// Prune drops transitions with count < minCount, then removes any state
// whose transitions are all pruned. Intended for the BuildRaw → Prune →
// Compress flow; has no effect when minCount <= 1.
//
// On prose corpora, minCount=2 typically halves memory by removing the
// long tail of singleton transitions (often OCR noise, typos, hapax
// legomena) without materially changing generation quality.
//
// Pruning cascades: removing an emptied state can orphan transitions in
// surviving states that pointed at it, so Prune repeats until no dangling
// transition remains. That keeps Gen from faulting with ErrStateNotFound
// mid-walk. Aggressive minCount can therefore cascade away large parts of
// the chain, including the begin state, leaving a model Gen cannot start
// from (it then returns ErrStateNotFound). Validate and a begin-state
// presence check are the caller's preflight before Gen.
func (c *Chain[T]) Prune(minCount uint32) *Chain[T] {
	if minCount <= 1 {
		return c
	}
	// First pass: drop low-count transitions and any state left empty.
	for state, choices := range c.Model {
		for token, count := range choices {
			if count < minCount {
				delete(choices, token)
			}
		}
		if len(choices) == 0 {
			delete(c.Model, state)
		}
	}

	// Fixed-point cascade: deleting a state orphans transitions in
	// surviving states that pointed at it. Repeatedly drop transitions
	// whose successor state no longer exists (and any state that empties
	// as a result) until a full pass changes nothing. Transitions to the
	// End sentinel are terminal and never dangling.
	for {
		changed := false
		for state, choices := range c.Model {
			toks := c.encoder.Decode(state)
			for token := range choices {
				if token == c.sentinels.End {
					continue
				}
				if _, ok := c.Model[c.successorKey(toks, token)]; !ok {
					delete(choices, token)
					changed = true
				}
			}
			if len(choices) == 0 {
				delete(c.Model, state)
				changed = true
			}
		}
		if !changed {
			break
		}
	}

	c.precomputeBeginState()
	return c
}

// successorKey returns the encoded key of the state reached by taking
// transition token out of the state whose decoded tokens are toks. The
// successor of state [t0 t1 … t_{n-1}] on token x is the window shifted
// left by one: [t1 … t_{n-1} x]. Callers must not pass the End sentinel
// as token: transitions to End are terminal and have no successor state.
func (c *Chain[T]) successorKey(toks []T, token T) string {
	next := make([]T, len(toks))
	copy(next, toks[1:])
	next[len(next)-1] = token
	return c.encoder.Encode(next)
}

// Validate walks the model and returns a descriptive error naming the
// first transition whose successor state is absent (a dangling transition
// that would fault Gen with ErrStateNotFound mid-walk), or nil if every
// non-terminal transition resolves. The returned error wraps
// ErrStateNotFound, so callers can errors.Is it.
//
// Prune's cascade already guarantees a clean graph; Validate is opt-in
// belt-and-braces for callers who mutate Model by hand or want to assert
// the invariant before Compress. O(states × fanout).
func (c *Chain[T]) Validate() error {
	for state, choices := range c.Model {
		toks := c.encoder.Decode(state)
		for token := range choices {
			if token == c.sentinels.End {
				continue
			}
			successor := c.successorKey(toks, token)
			if _, ok := c.Model[successor]; !ok {
				return fmt.Errorf("barkov: dangling transition from state %q on token %v to missing successor %q: %w",
					state, token, successor, ErrStateNotFound)
			}
		}
	}
	return nil
}

func calculateCumDist[T comparable](next map[T]uint32) ([]T, []uint32) {
	keys := make([]T, 0, len(next))
	cumDist := make([]uint32, 0, len(next))
	var total uint32
	for k, v := range next {
		total += v
		keys = append(keys, k)
		cumDist = append(cumDist, total)
	}
	return keys, cumDist
}

func chooseToken32(cumDist []uint32) int {
	choiceNum := rand.Uint32N(cumDist[len(cumDist)-1])
	return sort.Search(len(cumDist), func(i int) bool {
		return cumDist[i] > choiceNum
	})
}

// InitChain is the zero-configuration constructor for string chains.
// This is the common entry point for most users.
func InitChain(stateSize int) *Chain[string] {
	return NewChain(ChainConfig[string]{
		StateSize: stateSize,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Encoder:   SepEncoder{Sep: SEP},
	})
}
