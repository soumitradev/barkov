package interned

import (
	"testing"
	"unsafe"
)

// TestIndexedEquivalence asserts that every per-N BuildCompressedIndexedN
// produces the same per-state token distributions as the string-keyed
// Chain[TokenID].BuildCompressed at the matching stateSize. Since the
// indexed Model is keyed by [N]TokenID and the baseline Model by a
// packed-encoder string, we bridge the two by reinterpreting the string
// bytes as [N]TokenID — the two layouts are identical by construction
// (little-endian uint32 per token).
func TestIndexedEquivalence(t *testing.T) {
	corpus := [][]string{
		{"the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"},
		{"the", "quick", "brown", "fox", "runs"},
		{"the", "quick", "brown", "dog", "barks"},
		{"a", "quick", "brown", "fox", "is", "faster", "than", "a", "lazy", "dog"},
		{"the", "lazy", "dog", "sleeps", "all", "day", "long"},
		{"the", "fox", "and", "the", "dog", "are", "friends", "today"},
	}

	t.Run("N=2", func(t *testing.T) { checkIndexedEquiv[[2]TokenID](t, 2, corpus) })
	t.Run("N=3", func(t *testing.T) { checkIndexedEquiv[[3]TokenID](t, 3, corpus) })
	t.Run("N=4", func(t *testing.T) { checkIndexedEquiv[[4]TokenID](t, 4, corpus) })
	t.Run("N=5", func(t *testing.T) { checkIndexedEquiv[[5]TokenID](t, 5, corpus) })
	t.Run("N=6", func(t *testing.T) { checkIndexedEquiv[[6]TokenID](t, 6, corpus) })
	t.Run("N=7", func(t *testing.T) { checkIndexedEquiv[[7]TokenID](t, 7, corpus) })
	t.Run("N=8", func(t *testing.T) { checkIndexedEquiv[[8]TokenID](t, 8, corpus) })
}

func checkIndexedEquiv[K comparable](t *testing.T, stateSize int, corpus [][]string) {
	vocab := NewVocabulary()
	encoded := vocab.InternCorpus(corpus)

	chain, _ := InitChain(stateSize)
	baseline := chain.BuildCompressed(encoded)
	indexed := buildIndexedCore[K](encoded)

	if len(baseline.Model) != len(indexed.Model) {
		t.Fatalf("model size mismatch: baseline=%d indexed=%d",
			len(baseline.Model), len(indexed.Model))
	}

	for stateKey, baseIdx := range baseline.Model {
		if len(stateKey) != stateSize*4 {
			t.Fatalf("unexpected state key length: got %d want %d", len(stateKey), stateSize*4)
		}
		key := *(*K)(unsafe.Pointer(unsafe.StringData(stateKey)))
		idxRef, ok := indexed.Model[key]
		if !ok {
			t.Errorf("indexed missing state key (baseline count=%d)", baseIdx.Count)
			continue
		}
		baseChoices := baseline.Choices[baseIdx.Offset : baseIdx.Offset+uint32(baseIdx.Count)]
		baseCumDist := baseline.CumDist[baseIdx.Offset : baseIdx.Offset+uint32(baseIdx.Count)]
		idxChoices := indexed.Choices[idxRef.Offset : idxRef.Offset+uint32(idxRef.Count)]
		idxCumDist := indexed.CumDist[idxRef.Offset : idxRef.Offset+uint32(idxRef.Count)]
		baseCounts := cumDistToCounts(baseChoices, baseCumDist)
		idxCounts := cumDistToCounts(idxChoices, idxCumDist)
		if len(baseCounts) != len(idxCounts) {
			t.Errorf("choice-count differs: base=%d indexed=%d", len(baseCounts), len(idxCounts))
			continue
		}
		for tok, c := range baseCounts {
			if idxCounts[tok] != c {
				t.Errorf("token %d count: base=%d indexed=%d", tok, c, idxCounts[tok])
			}
		}
	}
}

func cumDistToCounts(choices []TokenID, cumDist []uint32) map[TokenID]uint32 {
	out := make(map[TokenID]uint32, len(choices))
	var prev uint32
	for i, tok := range choices {
		out[tok] = cumDist[i] - prev
		prev = cumDist[i]
	}
	return out
}
