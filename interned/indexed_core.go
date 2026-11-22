package interned

import (
	"fmt"
	"slices"
	"unsafe"

	barkov "github.com/soumitradev/barkov/v2"
)

// indexedCore is a fixed-order chain keyed by K = [N]TokenID. Keeping K
// pointer-free lets the GC skip the state table's entries entirely.
type indexedCore[K comparable] struct {
	random
	Model     *stateMap[K, barkov.ChoicesIndex]
	Choices   []TokenID
	CumDist   []uint32
	sentinels barkov.Sentinels[TokenID]
	encoder   PackedEncoder
	stateSize int
}

func (c *indexedCore[K]) StateSize() int                        { return c.stateSize }
func (c *indexedCore[K]) Sentinels() barkov.Sentinels[TokenID]  { return c.sentinels }
func (c *indexedCore[K]) Encoder() barkov.StateEncoder[TokenID] { return c.encoder }

// pickFollow resolves a ChoicesIndex to a TokenID
func (c *indexedCore[K]) pickFollow(idx barkov.ChoicesIndex) TokenID {
	// Fanout-1 states keep their only follower in Offset, which saves a
	// Choices load on the most common lookup
	if idx.Count == 1 {
		return TokenID(idx.Offset)
	}
	cumDist := c.CumDist[idx.Offset : idx.Offset+idx.Count]
	choices := c.Choices[idx.Offset : idx.Offset+idx.Count]
	return choices[scanCumDist(cumDist, c.uint32n(cumDist[len(cumDist)-1]))]
}

func (c *indexedCore[K]) MoveKey(key K) (TokenID, error) {
	idx, ok := c.Model.Get(key)
	if !ok {
		return 0, fmt.Errorf("barkov: state %v not in model: %w", key, barkov.ErrStateNotFound)
	}
	return c.pickFollow(idx), nil
}

// Move reads the state string as K directly, since PackedEncoder lays
// tokens out exactly as K holds them in memory
func (c *indexedCore[K]) Move(state string) (TokenID, error) {
	if len(state) != c.stateSize*4 {
		return 0, fmt.Errorf("barkov: state %q not in model: %w", state, barkov.ErrStateNotFound)
	}
	key := *(*K)(unsafe.Pointer(unsafe.StringData(state)))
	idx, ok := c.Model.Get(key)
	if !ok {
		return 0, fmt.Errorf("barkov: state %q not in model: %w", state, barkov.ErrStateNotFound)
	}
	return c.pickFollow(idx), nil
}

// buildIndexedCore mirrors Chain[TokenID].BuildCompressed keyed on K. The
// state table doubles as the final Model: the corpus scan stores each
// state's dense index in Offset and the grouping pass overwrites it in place.
func buildIndexedCore[K comparable](corpus [][]TokenID) *indexedCore[K] {
	var zero K
	stateSize := int(unsafe.Sizeof(zero)) / 4
	sentinels := DefaultSentinels()

	beginSeq := make([]TokenID, stateSize)
	for i := range beginSeq {
		beginSeq[i] = sentinels.Begin
	}

	maxLen, totalObs := 0, 0
	for _, run := range corpus {
		if len(run) > maxLen {
			maxLen = len(run)
		}
		totalObs += len(run) + 1
	}

	estUnique := int(float64(totalObs)*stateRatioEst(stateSize)) + 64
	stateIdx := newStateMap[K, barkov.ChoicesIndex](estUnique)
	items := make([]TokenID, 0, stateSize+maxLen+1)

	pendingState := make([]uint32, 0, totalObs)
	pendingTok := make([]TokenID, 0, totalObs)

	var numStates uint32
	for _, run := range corpus {
		items = items[:0]
		items = append(items, beginSeq...)
		items = append(items, run...)
		items = append(items, sentinels.End)

		for i := 0; i < len(run)+1; i++ {
			key := *(*K)(unsafe.Pointer(&items[i]))
			follow := items[i+stateSize]
			// Offset = idx+1 so no state's placeholder equals the empty-slot zero value
			existing, found := stateIdx.GetOrSet(key, barkov.ChoicesIndex{Offset: numStates + 1})
			var idx uint32
			if found {
				idx = existing.Offset - 1
			} else {
				idx = numStates
				numStates++
			}
			pendingState = append(pendingState, idx)
			pendingTok = append(pendingTok, follow)
		}
	}

	// Chain each state's observations: head[s] is its latest and next[i]
	// the one before i, -1 ending the list
	head := make([]int32, numStates)
	for i := range head {
		head[i] = -1
	}
	next := make([]int32, len(pendingState))
	for i, s := range pendingState {
		next[i] = head[s]
		head[s] = int32(i)
	}

	cc := &indexedCore[K]{
		sentinels: sentinels,
		encoder:   PackedEncoder{},
		Choices:   make([]TokenID, 0, totalObs),
		CumDist:   make([]uint32, 0, totalObs),
		stateSize: stateSize,
	}

	// Small groups dedupe by linear scan and large ones by sorting, both
	// cheaper than a map per state
	const linearThreshold = 16
	var sortBuf []TokenID
	stateIdx.forEach(func(_ K, val *barkov.ChoicesIndex) {
		s := val.Offset - 1
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
			sortBuf = sortBuf[:0]
			for i := head[s]; i != -1; i = next[i] {
				sortBuf = append(sortBuf, pendingTok[i])
			}
			slices.Sort(sortBuf)

			var prev TokenID
			var cnt uint32
			for idx, tok := range sortBuf {
				if idx == 0 {
					prev = tok
					cnt = 1
					continue
				}
				if tok == prev {
					cnt++
				} else {
					cc.Choices = append(cc.Choices, prev)
					cc.CumDist = append(cc.CumDist, cnt)
					prev = tok
					cnt = 1
				}
			}
			if len(sortBuf) > 0 {
				cc.Choices = append(cc.Choices, prev)
				cc.CumDist = append(cc.CumDist, cnt)
			}
		}

		count := uint32(len(cc.Choices) - groupStart)
		var indexOffset uint32
		if count == 1 {
			// Pack the only follower into Offset for pickFollow and drop its slot
			indexOffset = uint32(cc.Choices[groupStart])
			cc.Choices = cc.Choices[:groupStart]
			cc.CumDist = cc.CumDist[:groupStart]
		} else {
			var total uint32
			for m := groupStart; m < len(cc.Choices); m++ {
				total += cc.CumDist[m]
				cc.CumDist[m] = total
			}
			indexOffset = offset
		}
		*val = barkov.ChoicesIndex{
			Offset: indexOffset,
			Count:  count,
			Obs:    uint32(groupSize),
		}
	})
	cc.Model = stateIdx
	return cc
}

// stateRatioEst is the unique-states/observations ratio measured on the
// public corpus, used only to presize the state table
func stateRatioEst(stateSize int) float64 {
	switch stateSize {
	case 1:
		return 0.10
	case 2:
		return 0.45
	case 3:
		return 0.75
	default:
		return 0.85
	}
}
