package barkov

import (
	"testing"
	"unsafe"
)

// TestChoicesIndexSizeof pins the SoA entry size: Obs occupies what was
// trailing padding, so the struct must stay 8 bytes.
func TestChoicesIndexSizeof(t *testing.T) {
	if got := unsafe.Sizeof(ChoicesIndex{}); got != 8 {
		t.Errorf("unsafe.Sizeof(ChoicesIndex{}) = %d, want 8", got)
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

// TestChoicesIndexObsSaturation verifies Obs saturates at 65535 rather than
// wrapping, using 70,000 observations of a single state. Rows share one
// backing slice to keep allocation low; neither builder mutates its input.
func TestChoicesIndexObsSaturation(t *testing.T) {
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
	if idx.Obs != 65535 {
		t.Errorf("Obs = %d, want 65535 (saturated)", idx.Obs)
	}
}
