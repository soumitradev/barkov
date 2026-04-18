package interned

import (
	"encoding/binary"
	"fmt"
	"math/rand/v2"
	"sort"

	barkov "github.com/soumitradev/barkov/v2"
)

// packedKey4 packs four TokenIDs into a pointer-free 16-byte value so the
// state map's GC scan is zero cost. Byte layout matches PackedEncoder's
// little-endian output, which keeps Move(state string) lookups exact.
type packedKey4 struct {
	lo, hi uint64
}

func packKey4(ids []TokenID) packedKey4 {
	_ = ids[3]
	return packedKey4{
		lo: uint64(ids[0]) | uint64(ids[1])<<32,
		hi: uint64(ids[2]) | uint64(ids[3])<<32,
	}
}

func packKey4FromBytes(b []byte) packedKey4 {
	_ = b[15]
	return packedKey4{
		lo: binary.LittleEndian.Uint64(b[0:8]),
		hi: binary.LittleEndian.Uint64(b[8:16]),
	}
}

// IndexedCompressedChain is a stateSize=4 specialisation of
// CompressedChain[TokenID] that stores its state map with fixed-size
// array-shaped keys instead of string-shaped keys. The map value is also
// pointer-free, so the entire bucket array is invisible to the GC — mark
// cycles skip it entirely instead of walking ~550K string headers per scan.
type IndexedCompressedChain struct {
	Model     map[packedKey4]barkov.ChoicesIndex
	Choices   []TokenID
	CumDist   []uint32
	sentinels barkov.Sentinels[TokenID]
	encoder   PackedEncoder
}

var _ barkov.GenerativeChain[TokenID] = (*IndexedCompressedChain)(nil)

func (cc *IndexedCompressedChain) StateSize() int                     { return 4 }
func (cc *IndexedCompressedChain) MaxOverlap() int                    { return 6 }
func (cc *IndexedCompressedChain) Sentinels() barkov.Sentinels[TokenID] { return cc.sentinels }
func (cc *IndexedCompressedChain) Encoder() barkov.StateEncoder[TokenID] {
	return cc.encoder
}

func (cc *IndexedCompressedChain) Move(state string) (TokenID, error) {
	if len(state) != 16 {
		return 0, fmt.Errorf("barkov: state %q not in model: %w", state, barkov.ErrStateNotFound)
	}
	key := packKey4FromBytes([]byte(state))
	idx, ok := cc.Model[key]
	if !ok {
		return 0, fmt.Errorf("barkov: state %q not in model: %w", state, barkov.ErrStateNotFound)
	}
	cumDist := cc.CumDist[idx.Offset : idx.Offset+uint32(idx.Count)]
	choices := cc.Choices[idx.Offset : idx.Offset+uint32(idx.Count)]
	return choices[chooseToken32(cumDist)], nil
}

func chooseToken32(cumDist []uint32) int {
	choiceNum := rand.Uint32N(cumDist[len(cumDist)-1])
	return sort.Search(len(cumDist), func(i int) bool {
		return cumDist[i] > choiceNum
	})
}

// BuildCompressedIndexed mirrors Chain[TokenID].BuildCompressed but produces
// an IndexedCompressedChain keyed on packedKey4 rather than a string-keyed
// CompressedChain[TokenID]. Only stateSize=4 is supported; the caller is
// expected to pre-intern the corpus via Vocabulary.InternCorpus.
func BuildCompressedIndexed(corpus [][]TokenID) *IndexedCompressedChain {
	const stateSize = 4
	sentinels := DefaultSentinels()

	beginSeq := [stateSize]TokenID{sentinels.Begin, sentinels.Begin, sentinels.Begin, sentinels.Begin}

	maxLen, totalObs := 0, 0
	for _, run := range corpus {
		if len(run) > maxLen {
			maxLen = len(run)
		}
		totalObs += len(run) + 1
	}

	estUnique := totalObs/4 + 64
	stateIdx := make(map[packedKey4]uint32, estUnique)
	stateKeys := make([]packedKey4, 0, estUnique)
	items := make([]TokenID, 0, stateSize+maxLen+1)

	pendingState := make([]uint32, 0, totalObs)
	pendingTok := make([]TokenID, 0, totalObs)

	for _, run := range corpus {
		items = items[:0]
		items = append(items, beginSeq[:]...)
		items = append(items, run...)
		items = append(items, sentinels.End)

		for i := 0; i < len(run)+1; i++ {
			key := packKey4(items[i : i+stateSize])
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

	cc := &IndexedCompressedChain{
		sentinels: sentinels,
		encoder:   PackedEncoder{},
		Model:     make(map[packedKey4]barkov.ChoicesIndex, numStates),
		Choices:   make([]TokenID, 0, totalObs),
		CumDist:   make([]uint32, 0, totalObs),
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

		var total uint32
		for m := groupStart; m < len(cc.Choices); m++ {
			total += cc.CumDist[m]
			cc.CumDist[m] = total
		}
		cc.Model[stateKeys[s]] = barkov.ChoicesIndex{
			Offset: offset,
			Count:  uint16(len(cc.Choices) - groupStart),
		}
	}
	return cc
}
