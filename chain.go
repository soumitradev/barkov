package barkov

import (
	"fmt"
	"math/rand/v2"
	"sort"
	"strings"
)

// Compile-time check that Chain implements Transitioner
var _ Transitioner[string] = (*Chain[string])(nil)

// Sentinels to signify begin or end
type Sentinels[T comparable] struct {
	Begin T
	End   T
}

// Config, for a chain (duh)
type ChainConfig[T comparable] struct {
	StateSize int
	Sentinels Sentinels[T]
	// Key turns a state into a map key. It must give distinct keys to
	// distinct states of length StateSize
	Key func(state []T) string
}

// Chain is the generic markov chain type
type Chain[T comparable] struct {
	Model     map[string]map[T]uint32
	stateSize int
	sentinels Sentinels[T]
	key       func(state []T) string

	beginChoices []T
	beginCumDist []uint32
}

// Make a chain with a config
func NewChain[T comparable](cfg ChainConfig[T]) *Chain[T] {
	return &Chain[T]{
		Model:     make(map[string]map[T]uint32),
		stateSize: cfg.StateSize,
		sentinels: cfg.Sentinels,
		key:       cfg.Key,
	}
}

// BuildRaw constructs the chain from a pre-encoded corpus
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
			state := c.key(items[i : i+c.stateSize])
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
	key := c.key(beginSeq)
	if choices, ok := c.Model[key]; ok {
		c.beginChoices, c.beginCumDist = calculateCumDist(choices)
	}
}

// Implements Transitioner[T]
func (c *Chain[T]) StateSize() int          { return c.stateSize }
func (c *Chain[T]) Sentinels() Sentinels[T] { return c.sentinels }

// Move transitions from a state to the chosen next token
// Returns ErrStateNotFound (wrapped with the state) if the state doesn't exist
func (c *Chain[T]) Move(state []T) (T, error) {
	keys, cumDist, err := c.distribution(state)
	if err != nil {
		var zero T
		return zero, err
	}
	return keys[chooseToken32(cumDist)], nil
}

// distribution returns the transition distribution P(next | state)
// The begin state is cached
func (c *Chain[T]) distribution(state []T) ([]T, []uint32, error) {
	if c.isBeginState(state) && len(c.beginChoices) > 0 {
		return c.beginChoices, c.beginCumDist, nil
	}
	choices, ok := c.Model[c.key(state)]
	if !ok {
		return nil, nil, fmt.Errorf("barkov: state %v not in model: %w", state, ErrStateNotFound)
	}
	keys, cumDist := calculateCumDist(choices)
	return keys, cumDist, nil
}

func (c *Chain[T]) isBeginState(state []T) bool {
	if len(state) != c.stateSize {
		return false
	}
	for _, tok := range state {
		if tok != c.sentinels.Begin {
			return false
		}
	}
	return true
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

// ChoicesIndex points into the flat Choices/CumDist arrays
type ChoicesIndex struct {
	Offset uint32
	Count  uint16
}

// CompressedChain uses a struct-of-arrays layout for cache performance
type CompressedChain[T comparable] struct {
	Model     map[string]ChoicesIndex
	Choices   []T
	CumDist   []uint32
	stateSize int
	sentinels Sentinels[T]
	key       func(state []T) string
}

// Compile-time check that CompressedChain implements Transitioner
var _ Transitioner[string] = (*CompressedChain[string])(nil)

// Compress converts the chain to SoA layout for better cache performance
func (c *Chain[T]) Compress() *CompressedChain[T] {
	totalEntries := 0
	for _, choices := range c.Model {
		totalEntries += len(choices)
	}

	cc := &CompressedChain[T]{
		stateSize: c.stateSize,
		sentinels: c.sentinels,
		key:       c.key,
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

// Implements Transitioner[T]
func (cc *CompressedChain[T]) StateSize() int          { return cc.stateSize }
func (cc *CompressedChain[T]) Sentinels() Sentinels[T] { return cc.sentinels }
func (cc *CompressedChain[T]) Pick(idx ChoicesIndex) T {
	if idx.Count == 1 {
		return cc.Choices[idx.Offset]
	}
	cumDist := cc.CumDist[idx.Offset : idx.Offset+uint32(idx.Count)]
	choices := cc.Choices[idx.Offset : idx.Offset+uint32(idx.Count)]
	return choices[chooseToken32(cumDist)]
}

// Move transitions from a state to a randomly chosen next token
func (cc *CompressedChain[T]) Move(state []T) (T, error) {
	idx, ok := cc.Model[cc.key(state)]
	if !ok {
		var zero T
		return zero, fmt.Errorf("barkov: state %v not in model: %w", state, ErrStateNotFound)
	}
	cumDist := cc.CumDist[idx.Offset : idx.Offset+uint32(idx.Count)]
	choices := cc.Choices[idx.Offset : idx.Offset+uint32(idx.Count)]
	return choices[chooseToken32(cumDist)], nil
}

// InitChain is syntax sugar for a default string chain
func InitChain(stateSize int) *Chain[string] {
	return NewChain(ChainConfig[string]{
		StateSize: stateSize,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Key:       joinKey,
	})
}

// joinKey is the default key for string chains. Tokens must not contain SEP,
// or two states can share a key
func joinKey(state []string) string {
	return strings.Join(state, SEP)
}

// Build constructs the chain from a corpus
// This is a convenience wrapper around BuildRaw for backwards compatibility with the original API
func (c *Chain[T]) Build(corpus [][]T) *Chain[T] {
	return c.BuildRaw(corpus)
}
