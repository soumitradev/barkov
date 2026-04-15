package xxh3_test

import (
	"testing"

	"github.com/soumitradev/barkov/v2/hashers/xxh3"
)

func TestXXH3Deterministic(t *testing.T) {
	h := xxh3.XXH3{}
	if h.Hash([]byte("hello")) != h.Hash([]byte("hello")) {
		t.Error("expected deterministic output")
	}
}

func TestXXH3Distinct(t *testing.T) {
	h := xxh3.XXH3{}
	if h.Hash([]byte("foo")) == h.Hash([]byte("bar")) {
		t.Error("expected distinct hashes for distinct inputs")
	}
}
