package interned_test

import (
	"bufio"
	"context"
	"fmt"
	"math/rand/v2"
	"os"
	"strings"
	"testing"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/interned"
)

var internedCorpus [][]string
var internedCompressed barkov.GenerativeChain[interned.TokenID]

func init() {
	f, err := os.Open("../testdata/corpus_public.txt")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, "***") || strings.HasPrefix(line, "  ") {
			continue
		}
		tokens := strings.Fields(line)
		if len(tokens) >= 4 {
			internedCorpus = append(internedCorpus, tokens)
		}
	}

	vocab := interned.NewVocabulary()
	encoded := vocab.InternCorpus(internedCorpus)
	internedCompressed = interned.Build(4, encoded)
}

// BenchmarkGenHeavyInterned amortises b.Loop overhead across many Gens to
// isolate per-Move cost on the indexed N=4 path.
// Mirrors BenchmarkGenHeavy on the root package but against the interned
// max-opt build. Seeded PCG so workload per iter is byte-identical.
func BenchmarkGenHeavyInterned(b *testing.B) {
	ctx := context.Background()
	rng := internedCompressed.(barkov.RNGSettable)
	for b.Loop() {
		rng.SetRNG(rand.New(rand.NewPCG(0xb4, 0xc0)))
		for range 10000 {
			barkov.Gen(ctx, internedCompressed) //nolint
		}
	}
	rng.SetRNG(nil)
}

// BenchmarkEndToEnd measures the full pipeline using the interned package:
// intern corpus → build chain → compress → generate one sentence.
// Compare against the root package BenchmarkEndToEnd to track the speedup
// from integer token IDs and packed state keys.
func BenchmarkEndToEnd(b *testing.B) {
	ctx := context.Background()
	for b.Loop() {
		vocab := interned.NewVocabulary()
		chain := barkov.NewChain(barkov.ChainConfig[interned.TokenID]{
			StateSize: 4,
			Sentinels: interned.DefaultSentinels(),
			Encoder:   interned.PackedEncoder{},
		})
		encoded := vocab.InternCorpus(internedCorpus)
		chain.BuildRaw(encoded)
		compressed := chain.Compress()
		barkov.Gen(ctx, compressed) //nolint
	}
}

// BenchmarkBuildIndexedByN measures build cost for the indexed path at
// every supported stateSize. Gives a regression signal across N, not just
// the N=4 hot path that BenchmarkEndToEnd covers.
func BenchmarkBuildIndexedByN(b *testing.B) {
	vocab := interned.NewVocabulary()
	encoded := vocab.InternCorpus(internedCorpus)

	for _, n := range []int{2, 3, 4, 5, 6, 7, 8} {
		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			for b.Loop() {
				_ = interned.Build(n, encoded)
			}
		})
	}
}

// BenchmarkGenHeavyIndexedByN mirrors BenchmarkGenHeavyInterned for all
// supported N. Each subtest uses a freshly-built chain; the inner loop
// amortises b.Loop overhead so the per-Gen cost is visible. Validates
// that genIterSingleFast dispatches for every N.
func BenchmarkGenHeavyIndexedByN(b *testing.B) {
	ctx := context.Background()
	vocab := interned.NewVocabulary()
	encoded := vocab.InternCorpus(internedCorpus)

	for _, n := range []int{2, 3, 4, 5, 6, 7, 8} {
		chain := interned.Build(n, encoded)
		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			for b.Loop() {
				for range 10000 {
					barkov.Gen(ctx, chain) //nolint
				}
			}
		})
	}
}
