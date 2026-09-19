package interned

import (
	"errors"
	"math/rand/v2"
	"testing"

	barkov "github.com/soumitradev/barkov/v2"
)

func TestContextSamplingRespectsSuffixAndSupport(t *testing.T) {
	v := NewVocabulary()
	corpus := [][]string{{"rare", "seed", "after"}, {"left", "common", "seed", "after"}, {"left", "common", "seed", "after"}, {"left", "common", "seed", "after"}}
	c := buildBackoffCore(BackoffConfig{MaxOrder: 4, MinOrder: 3, MinSupport: 3, MaxVerbatim: -1}.withDefaults(), v.InternCorpus(corpus))
	c.SetRNG(rand.New(rand.NewPCG(1, 2)))
	seed := mustLookup(t, v, "seed")
	common := mustLookup(t, v, "common")
	left := mustLookup(t, v, "left")
	for range 30 {
		for _, suffix := range [][]TokenID{{seed}, {common, seed}} {
			history, err := c.SeedContext(suffix)
			if err != nil || len(history)+len(suffix) != 3 || history[0] != left {
				t.Fatalf("wrong suffix or unsupported context: %v %v", history, err)
			}
		}
	}
	if _, err := c.SeedContext([]TokenID{999999}); !errors.Is(err, barkov.ErrStateNotFound) {
		t.Fatalf("absent suffix misreported: %v", err)
	}
	if _, err := c.SeedContext([]TokenID{mustLookup(t, v, "rare"), seed}); !errors.Is(err, barkov.ErrInsufficientSupport) {
		t.Fatalf("support silently relaxed: %v", err)
	}
}

func BenchmarkSeedContext(b *testing.B) {
	corpus := make([][]TokenID, 20000)
	for i := range corpus {
		corpus[i] = []TokenID{TokenID(i + 3), TokenID(30000 + i%100), 40000}
	}
	c := buildBackoffCore(BackoffConfig{MaxOrder: 2, MinOrder: 2, MinSupport: 1, MaxVerbatim: -1}, corpus)
	seed := []TokenID{30042}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := c.SeedContext(seed); err != nil {
			b.Fatal(err)
		}
	}
}
