package interned

import (
	"testing"
)

// TestIndexedCompressedEquivalence asserts that BuildCompressedIndexed
// produces the same per-state token distributions as the string-keyed
// Chain[TokenID].BuildCompressed for every corpus case we care about.
func TestIndexedCompressedEquivalence(t *testing.T) {
	cases := []struct {
		name   string
		corpus [][]string
	}{
		{"empty", [][]string{}},
		{"single_short", [][]string{{"a", "b", "c", "d"}}},
		{"single_long", [][]string{{"a", "b", "c", "d", "e", "f", "g", "h"}}},
		{"duplicates", [][]string{
			{"a", "b", "c", "d", "e"},
			{"a", "b", "c", "d", "e"},
		}},
		{"mixed", [][]string{
			{"the", "quick", "brown", "fox", "jumps"},
			{"the", "quick", "brown", "dog", "runs"},
			{"a", "b", "c", "d", "e"},
		}},
		{"wide_fanout", func() [][]string {
			runs := make([][]string, 0, 20)
			for i := range 20 {
				r := []string{"the", "quick", "brown", "fox", string(rune('a' + i))}
				runs = append(runs, r)
			}
			return runs
		}()},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			vocab := NewVocabulary()
			encoded := vocab.InternCorpus(tc.corpus)

			chain, _ := InitChain(4)
			baseline := chain.BuildCompressed(encoded)
			indexed := BuildCompressedIndexed(encoded)

			if len(baseline.Model) != len(indexed.Model) {
				t.Fatalf("model size mismatch: baseline=%d indexed=%d",
					len(baseline.Model), len(indexed.Model))
			}

			for stateKey, baseIdx := range baseline.Model {
				baseChoices := baseline.Choices[baseIdx.Offset : baseIdx.Offset+uint32(baseIdx.Count)]
				baseCumDist := baseline.CumDist[baseIdx.Offset : baseIdx.Offset+uint32(baseIdx.Count)]
				baseCounts := cumDistToCounts(baseChoices, baseCumDist)

				idxKey := packKey4FromBytes([]byte(stateKey))
				idxRef, ok := indexed.Model[idxKey]
				if !ok {
					t.Errorf("indexed missing state key (baseline count=%d)", baseIdx.Count)
					continue
				}
				idxChoices := indexed.Choices[idxRef.Offset : idxRef.Offset+uint32(idxRef.Count)]
				idxCumDist := indexed.CumDist[idxRef.Offset : idxRef.Offset+uint32(idxRef.Count)]
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
		})
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
