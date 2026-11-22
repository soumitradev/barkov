package barkov

import (
	"fmt"
	"iter"
	"math/rand/v2"
	"slices"
	"sort"
)

// ChoicesIndex points into the flat Choices/CumDist arrays.
type ChoicesIndex struct {
	Offset uint32
	Count  uint32
	// Obs is the total number of corpus observations of this state.
	Obs uint32
}

// CompressedChain uses a struct-of-arrays layout for cache efficiency.
type CompressedChain[T comparable] struct {
	Model     map[string]ChoicesIndex
	Choices   []T
	CumDist   []uint32
	stateSize int
	sentinels Sentinels[T]
	encoder   StateEncoder[T]
	rng       *rand.Rand // nil = use package math/rand/v2
}

// SetRNG overrides the random source used by Move. Intended for
// deterministic benchmarks and tests; not safe for concurrent use.
func (cc *CompressedChain[T]) SetRNG(r *rand.Rand) { cc.rng = r }

// Compile-time checks that CompressedChain implements GenerativeChain and
// exposes its RNG override through RNGSettable.
var (
	_ GenerativeChain[string] = (*CompressedChain[string])(nil)
	_ RNGSettable             = (*CompressedChain[string])(nil)
)

// Compress converts the chain to SoA layout for better cache performance.
//
// Compress copies transitions verbatim and does not check for dangling
// successors; if the source Chain was mutated by hand (bypassing Prune's
// cascade), call Validate first to catch orphaned transitions at build
// time rather than as an ErrStateNotFound from Gen.
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

// observe streams every transition in corpus, returning the distinct state
// keys and, per transition, the dense index of its state and its follower
func (c *Chain[T]) observe(corpus [][]T) (keys []string, states []uint32, follows []T) {
	beginSeq := c.beginWindow()
	maxLen, totalObs := corpusShape(corpus)

	// Fanout is barely above 1 on prose, so nearly every transition is a new state
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

// stateGroups links each state's transitions into a list without sorting a
// copy: head[s] is the latest transition out of state s, or -1, and next[i]
// is the one before i out of the same state
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

// Linear scans beat hashing for groups this small and allocate nothing
const linearDedupeMax = 16

// followDeduper appends one state's distinct followers to a chain with their
// raw counts. Wide states like begin share one counting map so they stay
// linear in group size instead of quadratic.
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

// accumulate turns counts into a cumulative distribution in place and
// returns their total
func accumulate(counts []uint32) uint32 {
	var total uint32
	for i, n := range counts {
		total += n
		counts[i] = total
	}
	return total
}

// Implements GenerativeChain[T].
func (cc *CompressedChain[T]) StateSize() int           { return cc.stateSize }
func (cc *CompressedChain[T]) Sentinels() Sentinels[T]  { return cc.sentinels }
func (cc *CompressedChain[T]) Encoder() StateEncoder[T] { return cc.encoder }

// Move transitions from a state to a randomly chosen next token.
func (cc *CompressedChain[T]) Move(state string) (T, error) {
	idx, ok := cc.Model[state]
	if !ok {
		var zero T
		return zero, fmt.Errorf("barkov: state %q not in model: %w", state, ErrStateNotFound)
	}
	// Deterministic-choice fast path: when a state has exactly one follower,
	// we skip RNG, slice construction, and sort.Search.
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

// scanCumDist returns the first index whose cumulative weight exceeds
// choiceNum. State fanout averages ~1.1-1.8 on prose, so a linear scan
// wins for small groups; begin-state fanouts in the thousands flip the
// crossover to binary search.
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

// MoveTokens is a convenience wrapper that encodes the state for the caller.
func (cc *CompressedChain[T]) MoveTokens(tokens []T) (T, error) {
	return cc.Move(cc.encoder.Encode(tokens))
}

// ChoicesCumDist returns the choices and cumulative-distribution slices
// backing a state. The returned slices alias internal storage and must
// not be modified. Intended for callers that want to implement their
// own sampling (temperature scaling, repetition penalty, etc.) instead
// of the uniform weighted draw that Move performs.
//
// Returns ErrStateNotFound if the state isn't in the model.
func (cc *CompressedChain[T]) ChoicesCumDist(state string) (choices []T, cumDist []uint32, err error) {
	idx, ok := cc.Model[state]
	if !ok {
		return nil, nil, fmt.Errorf("barkov: state %q not in model: %w", state, ErrStateNotFound)
	}
	choices = cc.Choices[idx.Offset : idx.Offset+idx.Count]
	cumDist = cc.CumDist[idx.Offset : idx.Offset+idx.Count]
	return choices, cumDist, nil
}

// StateTotal returns the total observation weight of state (the sum of
// all its transition counts), or ErrStateNotFound.
func (cc *CompressedChain[T]) StateTotal(state string) (uint32, error) {
	idx, ok := cc.Model[state]
	if !ok {
		return 0, fmt.Errorf("barkov: state %q not in model: %w", state, ErrStateNotFound)
	}
	return idx.Obs, nil
}

// States iterates all states as decoded token slices with their total
// weights. The yielded slice is freshly allocated each iteration (via the
// chain's own encoder) and safe to retain, so callers doing custom
// scoring never cache an encoder or decode keys by hand. Perf-sensitive
// callers can iterate Model directly.
func (cc *CompressedChain[T]) States() iter.Seq2[[]T, uint32] {
	return func(yield func([]T, uint32) bool) {
		for state, idx := range cc.Model {
			if !yield(cc.encoder.Decode(state), idx.Obs) {
				return
			}
		}
	}
}
