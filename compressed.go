package barkov

import (
	"fmt"
	"iter"
	"math/rand/v2"
	"slices"
	"sort"
)

// ChoicesIndex locates a state's followers in Choices and CumDist, and Obs
// counts how often the state was observed.
type ChoicesIndex struct {
	Offset uint32
	Count  uint32
	Obs    uint32
}

// CompressedChain uses a struct-of-arrays layout for cache efficiency.
type CompressedChain[T comparable] struct {
	Model     map[string]ChoicesIndex
	Choices   []T
	CumDist   []uint32
	stateSize int
	sentinels Sentinels[T]
	encoder   StateEncoder[T]
	rng       *rand.Rand
}

// SetRNG makes Move draw from r instead of math/rand/v2, for deterministic
// tests and benchmarks. It is not safe for concurrent use.
func (cc *CompressedChain[T]) SetRNG(r *rand.Rand) { cc.rng = r }

var (
	_ GenerativeChain[string] = (*CompressedChain[string])(nil)
	_ RNGSettable             = (*CompressedChain[string])(nil)
)

// Compress packs the chain into a CompressedChain. It copies transitions as
// they are, so Validate a hand-edited Model first.
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
		var total uint32
		for k, v := range choices {
			total += v
			cc.Choices = append(cc.Choices, k)
			cc.CumDist = append(cc.CumDist, total)
		}
		cc.Model[state] = ChoicesIndex{
			Offset: offset,
			Count:  uint32(len(choices)),
			Obs:    total,
		}
	}
	return cc
}

// BuildCompressed builds a CompressedChain straight from a corpus, skipping
// the per-state maps of BuildRaw().Compress(). Order within a state is unspecified.
func (c *Chain[T]) BuildCompressed(corpus [][]T) *CompressedChain[T] {
	keys, states, follows := c.observe(corpus)
	groups := groupByState(states, len(keys))
	cc := &CompressedChain[T]{
		stateSize: c.stateSize,
		sentinels: c.sentinels,
		encoder:   c.encoder,
		Model:     make(map[string]ChoicesIndex, len(keys)),
		Choices:   make([]T, 0, len(follows)),
		CumDist:   make([]uint32, 0, len(follows)),
	}
	var dedupe followDeduper[T]
	for s, key := range keys {
		start := len(cc.Choices)
		dedupe.appendFollows(cc, follows, groups, s)
		cc.Model[key] = ChoicesIndex{
			Offset: uint32(start),
			Count:  uint32(len(cc.Choices) - start),
			Obs:    accumulate(cc.CumDist[start:]),
		}
	}
	return cc
}

func (c *Chain[T]) observe(corpus [][]T) (keys []string, states []uint32, follows []T) {
	beginSeq := c.beginWindow()
	maxLen, totalObs := corpusShape(corpus)
	estUnique := totalObs + 64
	stateIdx := make(map[string]uint32, estUnique)
	keys = make([]string, 0, estUnique)
	states = make([]uint32, 0, totalObs)
	follows = make([]T, 0, totalObs)
	items := make([]T, 0, c.stateSize+maxLen+1)
	ks := newKeyStream(c.encoder, estUnique*c.stateSize*8)

	for _, run := range corpus {
		items = c.padRun(items, beginSeq, run)
		for i := 0; i < len(run)+1; i++ {
			probeKey := ks.probe(items[i : i+c.stateSize])
			idx, ok := stateIdx[probeKey]
			if !ok {
				idx = uint32(len(keys))
				stableKey := ks.intern(probeKey)
				stateIdx[stableKey] = idx
				keys = append(keys, stableKey)
			}
			states = append(states, idx)
			follows = append(follows, items[i+c.stateSize])
		}
	}
	return keys, states, follows
}

// stateGroups threads each state's transitions into a list without sorting:
// head[s] is the latest transition out of s, or -1, and next[i] the one before i.
type stateGroups struct {
	head, next []int32
}

func groupByState(states []uint32, numStates int) stateGroups {
	g := stateGroups{head: make([]int32, numStates), next: make([]int32, len(states))}
	for s := range g.head {
		g.head[s] = -1
	}
	for i, s := range states {
		g.next[i] = g.head[s]
		g.head[s] = int32(i)
	}
	return g
}

func (g stateGroups) size(s int) int {
	n := 0
	for i := g.head[s]; i != -1; i = g.next[i] {
		n++
	}
	return n
}

// Linear scans beat hashing for groups this small and allocate nothing.
const linearDedupeMax = 16

// followDeduper reuses one counting map across wide states like begin, which
// would go quadratic under a linear scan.
type followDeduper[T comparable] struct {
	counts map[T]uint32
}

func (d *followDeduper[T]) appendFollows(cc *CompressedChain[T], follows []T, g stateGroups, s int) {
	size := g.size(s)
	if size <= linearDedupeMax {
		start := len(cc.Choices)
		for i := g.head[s]; i != -1; i = g.next[i] {
			tok := follows[i]
			if m := slices.Index(cc.Choices[start:], tok); m >= 0 {
				cc.CumDist[start+m]++
				continue
			}
			cc.Choices = append(cc.Choices, tok)
			cc.CumDist = append(cc.CumDist, 1)
		}
		return
	}

	if d.counts == nil {
		d.counts = make(map[T]uint32, size)
	} else {
		clear(d.counts)
	}
	for i := g.head[s]; i != -1; i = g.next[i] {
		d.counts[follows[i]]++
	}
	for tok, n := range d.counts {
		cc.Choices = append(cc.Choices, tok)
		cc.CumDist = append(cc.CumDist, n)
	}
}

func accumulate(counts []uint32) uint32 {
	var total uint32
	for i, n := range counts {
		total += n
		counts[i] = total
	}
	return total
}

func (cc *CompressedChain[T]) StateSize() int           { return cc.stateSize }
func (cc *CompressedChain[T]) Sentinels() Sentinels[T]  { return cc.sentinels }
func (cc *CompressedChain[T]) Encoder() StateEncoder[T] { return cc.encoder }

// Move samples the token after state, or returns ErrStateNotFound.
func (cc *CompressedChain[T]) Move(state string) (T, error) {
	idx, ok := cc.Model[state]
	if !ok {
		var zero T
		return zero, fmt.Errorf("barkov: state %q not in model: %w", state, ErrStateNotFound)
	}
	if idx.Count == 1 {
		return cc.Choices[idx.Offset], nil
	}
	cumDist := cc.CumDist[idx.Offset : idx.Offset+idx.Count]
	choices := cc.Choices[idx.Offset : idx.Offset+idx.Count]
	var choiceNum uint32
	if cc.rng != nil {
		choiceNum = cc.rng.Uint32N(cumDist[len(cumDist)-1])
	} else {
		choiceNum = rand.Uint32N(cumDist[len(cumDist)-1])
	}
	return choices[scanCumDist(cumDist, choiceNum)], nil
}

// scanCumDist returns the first index whose weight exceeds choiceNum. Most
// states have one or two followers, so it scans unless the state is wide.
func scanCumDist(cum []uint32, choiceNum uint32) int {
	if len(cum) <= 16 {
		for i, c := range cum {
			if c > choiceNum {
				return i
			}
		}
		return len(cum) - 1 // unreachable: choiceNum < cum[len-1]
	}
	return sort.Search(len(cum), func(i int) bool { return cum[i] > choiceNum })
}

// MoveTokens is Move for a state that isn't encoded yet.
func (cc *CompressedChain[T]) MoveTokens(tokens []T) (T, error) {
	return cc.Move(cc.encoder.Encode(tokens))
}

// ChoicesCumDist returns a state's followers and cumulative weights for
// custom sampling. The slices alias the chain's storage, so don't modify them.
func (cc *CompressedChain[T]) ChoicesCumDist(state string) (choices []T, cumDist []uint32, err error) {
	idx, ok := cc.Model[state]
	if !ok {
		return nil, nil, fmt.Errorf("barkov: state %q not in model: %w", state, ErrStateNotFound)
	}
	choices = cc.Choices[idx.Offset : idx.Offset+idx.Count]
	cumDist = cc.CumDist[idx.Offset : idx.Offset+idx.Count]
	return choices, cumDist, nil
}

// StateTotal returns how many times state was observed, or ErrStateNotFound.
func (cc *CompressedChain[T]) StateTotal(state string) (uint32, error) {
	idx, ok := cc.Model[state]
	if !ok {
		return 0, fmt.Errorf("barkov: state %q not in model: %w", state, ErrStateNotFound)
	}
	return idx.Obs, nil
}

// States yields each state's decoded tokens and observation count. Each
// slice is fresh and safe to keep. Iterate Model to skip decoding.
func (cc *CompressedChain[T]) States() iter.Seq2[[]T, uint32] {
	return func(yield func([]T, uint32) bool) {
		for state, idx := range cc.Model {
			if !yield(cc.encoder.Decode(state), idx.Obs) {
				return
			}
		}
	}
}
