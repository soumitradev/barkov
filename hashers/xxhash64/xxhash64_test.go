package xxhash64_test

import (
	"testing"

	"github.com/soumitradev/barkov/v2/hashers/xxhash64"
)

func TestXXHash64Deterministic(t *testing.T) {
	h := xxhash64.XXHash64{}
	if h.Hash([]byte("hello")) != h.Hash([]byte("hello")) {
		t.Error("expected deterministic output")
	}
}

func TestXXHash64Distinct(t *testing.T) {
	h := xxhash64.XXHash64{}
	if h.Hash([]byte("foo")) == h.Hash([]byte("bar")) {
		t.Error("expected distinct hashes for distinct inputs")
	}
}
