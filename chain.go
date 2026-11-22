package barkov

import (
	"fmt"
	"math/rand/v2"
	"sort"
)

var _ GenerativeChain[string] = (*Chain[string])(nil)

// Sentinels are the tokens framing each run, picked to never appear in the corpus.
type Sentinels[T comparable] struct {
	Begin T
	End   T
}

// ChainConfig holds what a Chain needs. Every field is required.
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

// NewChain builds an empty Chain. InitChain covers the common string case.
func NewChain[T comparable](cfg ChainConfig[T]) *Chain[T] {
	return &Chain[T]{
		Model:     make(map[string]map[T]uint32),
		stateSize: cfg.StateSize,
		sentinels: cfg.Sentinels,
		encoder:   cfg.Encoder,
	}
}

// BuildRaw counts every transition in corpus into the chain. Encoders that
// implement AppendEncoder build it without allocating per transition.
func (c *Chain[T]) BuildRaw(corpus [][]T) *Chain[T] {
	beginSeq := c.beginWindow()
	maxLen, totalObs := corpusShape(corpus)
	items := make([]T, 0, c.stateSize+maxLen+1)
	estUnique := totalObs + 64
	if len(c.Model) == 0 {
		c.Model = make(map[string]map[T]uint32, estUnique)
	}
	ks := newKeyStream(c.encoder, estUnique*c.stateSize*8)

	for _, run := range corpus {
		items = c.padRun(items, beginSeq, run)
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

func (c *Chain[T]) beginWindow() []T {
	w := make([]T, c.stateSize)
	for i := range w {
		w[i] = c.sentinels.Begin
	}
	return w
}

func corpusShape[T any](corpus [][]T) (maxLen, totalObs int) {
	for _, run := range corpus {
		maxLen = max(maxLen, len(run))
		totalObs += len(run) + 1
	}
	return maxLen, totalObs
}

func (c *Chain[T]) padRun(dst, begin, run []T) []T {
	dst = append(dst[:0], begin...)
	dst = append(dst, run...)
	return append(dst, c.sentinels.End)
}

func (c *Chain[T]) precomputeBeginState() {
	key := c.encoder.Encode(c.beginWindow())
	if choices, ok := c.Model[key]; ok {
		c.beginChoices, c.beginCumDist = calculateCumDist(choices)
	}
}

func (c *Chain[T]) StateSize() int           { return c.stateSize }
func (c *Chain[T]) Sentinels() Sentinels[T]  { return c.sentinels }
func (c *Chain[T]) Encoder() StateEncoder[T] { return c.encoder }

// Move samples the token after state, or returns ErrStateNotFound.
func (c *Chain[T]) Move(state string) (T, error) {
	choices, ok := c.Model[state]
	if !ok {
		var zero T
		return zero, fmt.Errorf("barkov: state %q not in model: %w", state, ErrStateNotFound)
	}
	if state == c.encoder.Encode(c.beginWindow()) && len(c.beginChoices) > 0 {
		return c.beginChoices[chooseToken32(c.beginCumDist)], nil
	}

	keys, cumDist := calculateCumDist(choices)
	return keys[chooseToken32(cumDist)], nil
}

// MoveTokens is Move for a state that isn't encoded yet.
func (c *Chain[T]) MoveTokens(tokens []T) (T, error) {
	return c.Move(c.encoder.Encode(tokens))
}

// Prune drops transitions seen fewer than minCount times, then any
// transition left leading to a missing state, so Gen never faults mid-walk.
// A high minCount can cascade away the begin state too, leaving Gen nowhere
// to start. minCount <= 1 keeps everything, and 2 typically halves memory on prose.
func (c *Chain[T]) Prune(minCount uint32) *Chain[T] {
	if minCount <= 1 {
		return c
	}
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

	// Dropping a state orphans transitions into it, so repeat until nothing changes
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

func (c *Chain[T]) successorKey(toks []T, token T) string {
	next := make([]T, len(toks))
	copy(next, toks[1:])
	next[len(next)-1] = token
	return c.encoder.Encode(next)
}

// Validate reports the first transition whose successor state is missing,
// wrapping ErrStateNotFound. Prune never leaves one, but hand edits to Model can.
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

// InitChain builds an empty string chain with the default sentinels and encoder.
func InitChain(stateSize int) *Chain[string] {
	// A constant literal converts to StateEncoder without allocating, unlike DefaultStringEncoder
	return NewChain(ChainConfig[string]{
		StateSize: stateSize,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Encoder:   SepEncoder{Sep: SEP},
	})
}
