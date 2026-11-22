package barkov

import (
	"fmt"
	"iter"
	"math/rand/v2"
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

// BuildCompressed builds a CompressedChain directly from a corpus, skipping
// the map-of-maps intermediate that Build+Compress produces. It eliminates
// the per-state map[T]uint32 allocations and the subsequent Compress
// iteration over them.
//
// Algorithm:
//  1. Stream observations into two parallel slices: a dense stateIdx per
//     observation and the follow token. State keys are interned into an
//     arena; unique states get sequential uint32 indices.
//  2. Counting-sort by stateIdx to group observations per state. Uses a
//     linked-list-style placement (head + next arrays) that needs O(obs)
//     aux memory instead of a full sorted buffer.
//  3. For each state group, linear-scan dedupe follow tokens directly into
//     cc.Choices/cc.CumDist for small groups, or fall back to a reusable
//     counting map for large groups. Then convert counts to cumulative in
//     place.
//
// Result is identical in behaviour to BuildRaw().Compress() except that the
// order of entries within a state group is unspecified (Compress also has
// unspecified order due to Go's random map iteration).
func (c *Chain[T]) BuildCompressed(corpus [][]T) *CompressedChain[T] {
	beginSeq := c.beginWindow()
	maxLen, totalObs := corpusShape(corpus)

	// Observed on the public corpus: numStates ≈ 0.82 * totalObs (fanout
	// barely >1), so the old estimate of totalObs/4 undershot by ~3.3x,
	// forcing the map through several rehashes and the slice through
	// multiple doublings. Using totalObs as the upper bound wastes ~20% in
	// slack but eliminates all growth cost.
	estUnique := totalObs + 64
	stateIdx := make(map[string]uint32, estUnique)
	stateKeys := make([]string, 0, estUnique)
	items := make([]T, 0, c.stateSize+maxLen+1)

	pendingState := make([]uint32, 0, totalObs)
	pendingTok := make([]T, 0, totalObs)

	ks := newKeyStream(c.encoder, estUnique*c.stateSize*8)

	for _, run := range corpus {
		items = items[:0]
		items = append(items, beginSeq...)
		items = append(items, run...)
		items = append(items, c.sentinels.End)

		for i := 0; i < len(run)+1; i++ {
			probeKey := ks.probe(items[i : i+c.stateSize])
			follow := items[i+c.stateSize]
			idx, ok := stateIdx[probeKey]
			if !ok {
				idx = uint32(len(stateKeys))
				stableKey := ks.intern(probeKey)
				stateIdx[stableKey] = idx
				stateKeys = append(stateKeys, stableKey)
			}
			pendingState = append(pendingState, idx)
			pendingTok = append(pendingTok, follow)
		}
	}

	numStates := len(stateKeys)
	// Per-state linked list of observation indices. head[s] is the latest
	// observation index for state s (or -1); next[i] is the prior obs idx
	// sharing the same state.
	head := make([]int32, numStates)
	for i := range head {
		head[i] = -1
	}
	next := make([]int32, len(pendingState))
	for i, s := range pendingState {
		next[i] = head[s]
		head[s] = int32(i)
	}

	cc := &CompressedChain[T]{
		stateSize: c.stateSize,
		sentinels: c.sentinels,
		encoder:   c.encoder,
		Model:     make(map[string]ChoicesIndex, numStates),
		Choices:   make([]T, 0, totalObs),
		CumDist:   make([]uint32, 0, totalObs),
	}

	// Walk each state's linked list; dedupe follows into cc.Choices/CumDist.
	// Small groups (<=16) dedupe via linear scan (beats map hashing and
	// allocates nothing); large groups use a reusable counting map whose
	// bucket storage is cleared and reused across states, amortising cost
	// at O(group_size) without the O(group²) blowup that linear scan would
	// suffer on pathological fan-out states like begin.
	const linearThreshold = 16
	var countBuf map[T]uint32
	for s := range numStates {
		offset := uint32(len(cc.Choices))
		groupStart := len(cc.Choices)

		groupSize := 0
		for i := head[s]; i != -1; i = next[i] {
			groupSize++
		}

		if groupSize <= linearThreshold {
			for i := head[s]; i != -1; i = next[i] {
				tok := pendingTok[i]
				found := false
				for m := groupStart; m < len(cc.Choices); m++ {
					if cc.Choices[m] == tok {
						cc.CumDist[m]++
						found = true
						break
					}
				}
				if !found {
					cc.Choices = append(cc.Choices, tok)
					cc.CumDist = append(cc.CumDist, 1)
				}
			}
		} else {
			if countBuf == nil {
				countBuf = make(map[T]uint32, groupSize)
			} else {
				clear(countBuf)
			}
			for i := head[s]; i != -1; i = next[i] {
				countBuf[pendingTok[i]]++
			}
			for tok, cnt := range countBuf {
				cc.Choices = append(cc.Choices, tok)
				cc.CumDist = append(cc.CumDist, cnt)
			}
		}

		// Convert raw counts to cumulative distribution.
		var total uint32
		for m := groupStart; m < len(cc.Choices); m++ {
			total += cc.CumDist[m]
			cc.CumDist[m] = total
		}
		cc.Model[stateKeys[s]] = ChoicesIndex{
			Offset: offset,
			Count:  uint32(len(cc.Choices) - groupStart),
			Obs:    total,
		}
	}
	return cc
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
