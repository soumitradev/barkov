package barkov

import (
	"testing"
	"unsafe"
)

// TestChoicesIndexSizeof pins the SoA entry size: Offset uint32, Count uint32, Obs uint32 = 12 bytes.
func TestChoicesIndexSizeof(t *testing.T) {
	if got := unsafe.Sizeof(ChoicesIndex{}); got != 12 {
		t.Errorf("unsafe.Sizeof(ChoicesIndex{}) = %d, want 12", got)
	}
}

// TestChoicesIndexObsKnownCounts checks that Obs reflects the total corpus
// observation count of a state (not the distinct-follow Count) on both the
// Build+Compress and BuildCompressed paths.
func TestChoicesIndexObsKnownCounts(t *testing.T) {
	corpus := [][]string{
		{"a", "b", "c"},
		{"a", "b", "c"},
		{"a", "b", "c"},
	}
	enc := SepEncoder{Sep: SEP}
	beginKey := enc.Encode([]string{BEGIN, BEGIN})

	t.Run("Compress", func(t *testing.T) {
		cc := InitChain(2).BuildRaw(corpus).Compress()
		idx, ok := cc.Model[beginKey]
		if !ok {
			t.Fatalf("begin state missing from model")
		}
		if idx.Obs != 3 {
			t.Errorf("Obs = %d, want 3", idx.Obs)
		}
	})

	t.Run("BuildCompressed", func(t *testing.T) {
		cc := InitChain(2).BuildCompressed(corpus)
		idx, ok := cc.Model[beginKey]
		if !ok {
			t.Fatalf("begin state missing from model")
		}
		if idx.Obs != 3 {
			t.Errorf("Obs = %d, want 3", idx.Obs)
		}
	})
}

// TestChoicesIndexObsLargeCounts verifies Obs handles counts >65535 without
// wrapping or truncation, using 70,000 observations of a single state.
func TestChoicesIndexObsLargeCounts(t *testing.T) {
	const n = 70000
	row := []string{"a", "b"}
	corpus := make([][]string, n)
	for i := range corpus {
		corpus[i] = row
	}

	enc := SepEncoder{Sep: SEP}
	beginKey := enc.Encode([]string{BEGIN, BEGIN})

	cc := InitChain(2).BuildCompressed(corpus)
	idx, ok := cc.Model[beginKey]
	if !ok {
		t.Fatalf("begin state missing from model")
	}
	if idx.Obs != 70000 {
		t.Errorf("Obs = %d, want 70000", idx.Obs)
	}
}
