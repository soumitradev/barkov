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

// BuildRaw constructs the chain from a pre-encoded corpus. Unlike Build,
// it does not convert from strings — the corpus is already in the target
// token type T. This is the canonical, generic build path.
func (c *Chain[T]) BuildRaw(corpus [][]T) *Chain[T] {
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

func (c *Chain[T]) precomputeBeginState() {
	beginSeq := make([]T, c.stateSize)
	for i := range beginSeq {
		beginSeq[i] = c.sentinels.Begin
	}
	key := c.encoder.Encode(beginSeq)
	if choices, ok := c.Model[key]; ok {
		c.beginChoices, c.beginCumDist = calculateCumDist(choices)
	}
}

// Implements GenerativeChain[T].
func (c *Chain[T]) StateSize() int           { return c.stateSize }
func (c *Chain[T]) MaxOverlap() int          { return c.stateSize + 2 }
func (c *Chain[T]) Sentinels() Sentinels[T] { return c.sentinels }
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
	beginSeq := make([]T, c.stateSize)
	for i := range beginSeq {
		beginSeq[i] = c.sentinels.Begin
	}
	if state == c.encoder.Encode(beginSeq) && len(c.beginChoices) > 0 {
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

// ChoicesIndex points into the flat Choices/CumDist arrays.
type ChoicesIndex struct {
	Offset uint32
	Count  uint16
}

// CompressedChain uses a struct-of-arrays layout for cache efficiency.
type CompressedChain[T comparable] struct {
	Model     map[string]ChoicesIndex
	Choices   []T
	CumDist   []uint32
	stateSize int
	sentinels Sentinels[T]
	encoder   StateEncoder[T]
}

// Compile-time check that CompressedChain implements GenerativeChain.
var _ GenerativeChain[string] = (*CompressedChain[string])(nil)

// Compress converts the chain to SoA layout for better cache performance.
func (c *Chain[T]) Compress() *CompressedChain[T] {
	totalEntries := 0
	for _, choices := range c.Model {
		totalEntries += len(choices)
	}

	cc := &CompressedChain[T]{
		stateSize: c.stateSize,
		sentinels: c.sentinels,
		encoder:   c.encoder,
		Model:     make(map[string]ChoicesIndex, len(c.Model)),
		Choices:   make([]T, 0, totalEntries),
		CumDist:   make([]uint32, 0, totalEntries),
	}

	for state, choices := range c.Model {
		offset := uint32(len(cc.Choices))
		keys, cumDist := calculateCumDist(choices)
		cc.Choices = append(cc.Choices, keys...)
		cc.CumDist = append(cc.CumDist, cumDist...)
		cc.Model[state] = ChoicesIndex{
			Offset: offset,
			Count:  uint16(len(keys)),
		}
	}
	return cc
}

// Implements GenerativeChain[T].
func (cc *CompressedChain[T]) StateSize() int           { return cc.stateSize }
func (cc *CompressedChain[T]) MaxOverlap() int          { return cc.stateSize + 2 }
func (cc *CompressedChain[T]) Sentinels() Sentinels[T]  { return cc.sentinels }
func (cc *CompressedChain[T]) Encoder() StateEncoder[T] { return cc.encoder }

// Move transitions from a state to a randomly chosen next token.
func (cc *CompressedChain[T]) Move(state string) (T, error) {
	idx, ok := cc.Model[state]
	if !ok {
		var zero T
		return zero, fmt.Errorf("barkov: state %q not in model: %w", state, ErrStateNotFound)
	}
	cumDist := cc.CumDist[idx.Offset : idx.Offset+uint32(idx.Count)]
	choices := cc.Choices[idx.Offset : idx.Offset+uint32(idx.Count)]
	return choices[chooseToken32(cumDist)], nil
}

// MoveTokens is a convenience wrapper that encodes the state for the caller.
func (cc *CompressedChain[T]) MoveTokens(tokens []T) (T, error) {
	return cc.Move(cc.encoder.Encode(tokens))
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

// Build constructs the chain from a corpus. This is a convenience wrapper
// around BuildRaw for backwards compatibility with the original API.
func (c *Chain[T]) Build(corpus [][]T) *Chain[T] {
	return c.BuildRaw(corpus)
}
