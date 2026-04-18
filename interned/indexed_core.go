package interned

import (
	"fmt"
	"math/rand/v2"
	"sort"
	"unsafe"

	barkov "github.com/soumitradev/barkov/v2"
)

// indexedCore is the key-type-parameterised core shared by every
// IndexedCompressedChainN. K is always a pointer-free fixed-size array
// type (e.g. [4]TokenID) so the state map's bucket array is invisible to
// the GC — mark cycles skip it entirely instead of walking hundreds of
// thousands of string headers per scan.
//
// The per-N wrapper types in indexed.go / indexed_variants.go embed
// *indexedCore[K] by pointer; the wrappers carry no data of their own
// and exist only to give each stateSize its own distinct type so that
// (for example) FastMoverKey[[3]TokenID, TokenID] and
// FastMoverKey[[4]TokenID, TokenID] are separate interfaces.
type indexedCore[K comparable] struct {
	Model     *stateMap[K, barkov.ChoicesIndex]
	Choices   []TokenID
	CumDist   []uint32
	sentinels barkov.Sentinels[TokenID]
	encoder   PackedEncoder
	rng       *rand.Rand
	stateSize int // = unsafe.Sizeof(K)/4; stored once for StateSize/Move
}

// SetRNG overrides the random source used by Move. Intended for
// deterministic benchmarks and tests; not safe for concurrent use.
func (c *indexedCore[K]) SetRNG(r *rand.Rand) { c.rng = r }

func (c *indexedCore[K]) StateSize() int                        { return c.stateSize }
func (c *indexedCore[K]) MaxOverlap() int                       { return c.stateSize + 2 }
func (c *indexedCore[K]) Sentinels() barkov.Sentinels[TokenID]  { return c.sentinels }
func (c *indexedCore[K]) Encoder() barkov.StateEncoder[TokenID] { return c.encoder }

// pickFollow resolves a ChoicesIndex to a TokenID. Fanout-1 states pack
// the follower TokenID into Offset at build time, so the hot path returns
// without touching Choices[] — one fewer cache miss per Count==1 lookup,
// which dominates most corpora.
func (c *indexedCore[K]) pickFollow(idx barkov.ChoicesIndex) TokenID {
	if idx.Count == 1 {
		return TokenID(idx.Offset)
	}
	cumDist := c.CumDist[idx.Offset : idx.Offset+uint32(idx.Count)]
	choices := c.Choices[idx.Offset : idx.Offset+uint32(idx.Count)]
	var choiceNum uint32
	if c.rng != nil {
		choiceNum = c.rng.Uint32N(cumDist[len(cumDist)-1])
	} else {
		choiceNum = rand.Uint32N(cumDist[len(cumDist)-1])
	}
	return choices[sort.Search(len(cumDist), func(i int) bool { return cumDist[i] > choiceNum })]
}

// MoveKey satisfies barkov.FastMoverKey[K, TokenID]. Gen uses this path
// when the chain's stateSize matches a registered FastMoverKey[[N]T, T],
// bypassing encoder.Encode + the Move(string) roundtrip.
func (c *indexedCore[K]) MoveKey(key K) (TokenID, error) {
	idx, ok := c.Model.Get(key)
	if !ok {
		return 0, fmt.Errorf("barkov: state %v not in model: %w", key, barkov.ErrStateNotFound)
	}
	return c.pickFollow(idx), nil
}

// Move satisfies barkov.GenerativeChain[TokenID]. The packed-encoder
// byte layout is identical to K's in-memory layout (both little-endian
// uint32s back-to-back), so we can reinterpret the string bytes as K
// without a decode loop. Little-endian is already assumed elsewhere via
// binary.LittleEndian.Put/AppendUint32.
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

// buildIndexedCore builds the indexed chain for any pointer-free fixed-size
// K = [N]TokenID. Mirrors Chain[TokenID].BuildCompressed, but keyed on K
// rather than a string. The caller is expected to pre-intern the corpus
// via Vocabulary.InternCorpus and to select a K whose byte size matches
// stateSize*4.
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

	// Observed ratio on the public corpus: numStates ≈ 0.82 * totalObs
	// (fanout barely >1). The old estimate of totalObs/4 undershot by ~3.3x,
	// forcing the map through several rehashes and the slice through
	// multiple doublings. Using totalObs as the upper bound wastes ~20% of
	// one slice + one map in slack, but eliminates all growth cost.
	// stateIdx is transient (freed when buildIndexedCore returns). Keep it
	// on Go's runtime map; the custom stateMap's 0.5 load-factor sizing
	// roughly doubles build-time memory for no gen-path payoff — the
	// build-path lookup isn't hot enough to offset the extra RAM.
	estUnique := totalObs + 64
	stateIdx := make(map[K]uint32, estUnique)
	stateKeys := make([]K, 0, estUnique)
	items := make([]TokenID, 0, stateSize+maxLen+1)

	pendingState := make([]uint32, 0, totalObs)
	pendingTok := make([]TokenID, 0, totalObs)

	for _, run := range corpus {
		items = items[:0]
		items = append(items, beginSeq...)
		items = append(items, run...)
		items = append(items, sentinels.End)

		for i := 0; i < len(run)+1; i++ {
			// Window items[i:i+stateSize] has length stateSize = sizeof(K)/4,
			// so reinterpreting the first stateSize*4 bytes as K is safe.
			key := *(*K)(unsafe.Pointer(&items[i]))
			follow := items[i+stateSize]
			existing, ok := stateIdx[key]
			var idx uint32
			if ok {
				idx = existing
			} else {
				idx = uint32(len(stateKeys))
				stateIdx[key] = idx
				stateKeys = append(stateKeys, key)
			}
			pendingState = append(pendingState, idx)
			pendingTok = append(pendingTok, follow)
		}
	}

	numStates := len(stateKeys)
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
		Model:     newStateMap[K, barkov.ChoicesIndex](numStates),
		Choices:   make([]TokenID, 0, totalObs),
		CumDist:   make([]uint32, 0, totalObs),
		stateSize: stateSize,
	}

	const linearThreshold = 16
	var countBuf map[TokenID]uint32
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
				countBuf = make(map[TokenID]uint32, groupSize)
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

		count := uint16(len(cc.Choices) - groupStart)
		var indexOffset uint32
		if count == 1 {
			// Fanout-1: pack the follower TokenID into Offset so pickFollow
			// returns without a Choices[] load. Roll back the unused slot.
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
		cc.Model.Put(stateKeys[s], barkov.ChoicesIndex{
			Offset: indexOffset,
			Count:  count,
		})
	}
	return cc
}
