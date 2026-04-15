package barkov

import (
	"fmt"
	"math/rand/v2"
	"sort"
)

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

// GenericChain is the generic chain type. Once all tests pass with this,
// we'll rename it to Chain and delete the old monolithic Chain in Phase C.
type GenericChain[T comparable] struct {
	Model     map[string]map[T]uint32
	stateSize int
	sentinels Sentinels[T]
	encoder   StateEncoder[T]

	beginChoices []T
	beginCumDist []uint32
}

// NewGenericChain constructs a GenericChain from a config.
// This is the generic entry point. For the common string case,
// use InitChain(stateSize) which fills in defaults.
func NewGenericChain[T comparable](cfg ChainConfig[T]) *GenericChain[T] {
	return &GenericChain[T]{
		Model:     make(map[string]map[T]uint32),
		stateSize: cfg.StateSize,
		sentinels: cfg.Sentinels,
		encoder:   cfg.Encoder,
	}
}

// BuildRaw constructs the chain from a pre-encoded corpus. Unlike Build,
// it does not convert from strings — the corpus is already in the target
// token type T. This is the canonical, generic build path.
func (c *GenericChain[T]) BuildRaw(corpus [][]T) *GenericChain[T] {
	beginSeq := make([]T, c.stateSize)
	for i := range beginSeq {
		beginSeq[i] = c.sentinels.Begin
	}

	maxLen := 0
	for _, run := range corpus {
		if len(run) > maxLen {
			maxLen = len(run)
		}
	}
	items := make([]T, 0, c.stateSize+maxLen+1)

	for _, run := range corpus {
		items = items[:0]
		items = append(items, beginSeq...)
		items = append(items, run...)
		items = append(items, c.sentinels.End)

		for i := 0; i < len(run)+1; i++ {
			state := c.encoder.Encode(items[i : i+c.stateSize])
			follow := items[i+c.stateSize]

			if _, ok := c.Model[state]; !ok {
				c.Model[state] = make(map[T]uint32)
			}
			c.Model[state][follow]++
		}
	}

	c.precomputeBeginState()
	return c
}

func (c *GenericChain[T]) precomputeBeginState() {
	beginSeq := make([]T, c.stateSize)
	for i := range beginSeq {
		beginSeq[i] = c.sentinels.Begin
	}
	key := c.encoder.Encode(beginSeq)
	if choices, ok := c.Model[key]; ok {
		c.beginChoices, c.beginCumDist = calculateCumDistGeneric(choices)
	}
}

// Implements GenericGenerativeChain[T].
func (c *GenericChain[T]) StateSize() int           { return c.stateSize }
func (c *GenericChain[T]) MaxOverlap() int          { return c.stateSize + 2 }
func (c *GenericChain[T]) GetSentinels() Sentinels[T]  { return c.sentinels }
func (c *GenericChain[T]) Encoder() StateEncoder[T] { return c.encoder }

// Move transitions from a state to a randomly chosen next token.
// Returns ErrStateNotFound (wrapped with the state) if the state doesn't exist.
func (c *GenericChain[T]) Move(state string) (T, error) {
	choices, ok := c.Model[state]
	if !ok {
		var zero T
		return zero, fmt.Errorf("barkov: state %q not in model: %w", state, ErrStateNotFound)
	}

	// Fast-path for the begin state: reuse precomputed distributions.
	beginSeq := make([]T, c.stateSize)
	for i := range beginSeq {
		beginSeq[i] = c.sentinels.Begin
	}
	if state == c.encoder.Encode(beginSeq) && len(c.beginChoices) > 0 {
		return c.beginChoices[chooseToken32(c.beginCumDist)], nil
	}

	keys, cumDist := calculateCumDistGeneric(choices)
	return keys[chooseToken32(cumDist)], nil
}

// MoveTokens is a convenience wrapper that encodes the state for the caller.
// Equivalent to c.Move(c.Encoder().Encode(tokens)) but saves boilerplate.
// Not part of the GenericGenerativeChain[T] interface.
func (c *GenericChain[T]) MoveTokens(tokens []T) (T, error) {
	return c.Move(c.encoder.Encode(tokens))
}

func calculateCumDistGeneric[T comparable](next map[T]uint32) ([]T, []uint32) {
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
