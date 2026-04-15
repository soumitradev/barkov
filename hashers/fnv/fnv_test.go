package fnv_test

import (
	"testing"

	"github.com/soumitradev/barkov/v2/hashers/fnv"
)

func TestFNVDeterministic(t *testing.T) {
	h := fnv.FNV{}
	a := h.Hash([]byte("hello"))
	b := h.Hash([]byte("hello"))
	if a != b {
		t.Error("expected deterministic output")
	}
}

func TestFNVDistinct(t *testing.T) {
	h := fnv.FNV{}
	if h.Hash([]byte("foo")) == h.Hash([]byte("bar")) {
		t.Error("expected distinct hashes for distinct inputs")
	}
}

func TestFNVEmpty(t *testing.T) {
	h := fnv.FNV{}
	_ = h.Hash(nil)
	_ = h.Hash([]byte{})
}
