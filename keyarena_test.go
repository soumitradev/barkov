package barkov

import (
	"strings"
	"testing"
)

func TestKeyArenaEmptyAppend(t *testing.T) {
	a := newKeyArena(0)
	if got := a.Append(nil); got != "" {
		t.Errorf("Append(nil) = %q; want %q", got, "")
	}
	if got := a.Append([]byte{}); got != "" {
		t.Errorf("Append([]byte{}) = %q; want %q", got, "")
	}
}

func TestKeyArenaBasic(t *testing.T) {
	a := newKeyArena(0)
	k1 := a.Append([]byte("hello"))
	k2 := a.Append([]byte("world"))
	if k1 != "hello" || k2 != "world" {
		t.Errorf("got %q, %q; want hello, world", k1, k2)
	}
}

func TestKeyArenaStabilityAcrossSlabGrowth(t *testing.T) {
	// Force multiple slab spills; every previously returned key must
	// remain byte-equal to what was appended.
	a := newKeyArena(16) // below floor; floor=4096 applies
	var keys []string
	var want []string
	for i := range 1000 {
		b := []byte(strings.Repeat("x", (i%50)+1))
		keys = append(keys, a.Append(b))
		want = append(want, string(b))
	}
	for i, k := range keys {
		if k != want[i] {
			t.Fatalf("key %d corrupted after slab growth: got %q want %q", i, k, want[i])
		}
	}
}

func TestKeyArenaKeyLargerThanInitialSlab(t *testing.T) {
	// Ask for an initial slab below the floor; then append a key larger
	// than the floor itself, forcing an oversize-key allocation path.
	a := newKeyArena(0) // → 4096 floor
	big := make([]byte, 8192)
	for i := range big {
		big[i] = byte(i)
	}
	k := a.Append(big)
	if len(k) != len(big) || k != string(big) {
		t.Fatalf("oversize key corrupted: len got %d want %d", len(k), len(big))
	}
	// Follow-up small key must still work.
	if got := a.Append([]byte("after")); got != "after" {
		t.Errorf("post-oversize small key: got %q want %q", got, "after")
	}
}

func TestKeyArenaSlabBoundary(t *testing.T) {
	// A key that exactly fills the remaining capacity of the current slab
	// must not corrupt and must not force an unnecessary reallocation.
	a := newKeyArena(16) // → 4096 floor
	filler := make([]byte, 4000)
	a.Append(filler)
	// Remaining capacity is 96 bytes; append exactly 96.
	boundary := make([]byte, 96)
	for i := range boundary {
		boundary[i] = 'B'
	}
	k := a.Append(boundary)
	if k != string(boundary) {
		t.Errorf("boundary key corrupted")
	}
	// One more byte forces a spill.
	spill := []byte("spill")
	if got := a.Append(spill); got != "spill" {
		t.Errorf("post-boundary spill: got %q want %q", got, "spill")
	}
}
