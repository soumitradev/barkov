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
var internedCompressed *interned.IndexedCompressedChain

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
	internedCompressed = interned.BuildCompressedIndexed(encoded)
}

// BenchmarkGenHeavyInterned amortises b.Loop overhead across many Gens to
// isolate per-Move cost on the IndexedCompressedChain (stateSize=4) path.
// Mirrors BenchmarkGenHeavy on the root package but against the interned
// max-opt build. Seeded PCG so workload per iter is byte-identical.
func BenchmarkGenHeavyInterned(b *testing.B) {
	ctx := context.Background()
	for b.Loop() {
		internedCompressed.SetRNG(rand.New(rand.NewPCG(0xb4, 0xc0)))
		for range 10000 {
			barkov.Gen(ctx, internedCompressed) //nolint
		}
	}
	internedCompressed.SetRNG(nil)
}

// BenchmarkEndToEnd measures the full pipeline using the interned package:
// intern corpus → build chain → compress → generate one sentence.
// Compare against the root package BenchmarkEndToEnd to track the speedup
// from integer token IDs and packed state keys.
func BenchmarkEndToEnd(b *testing.B) {
	ctx := context.Background()
	for b.Loop() {
		chain, vocab := interned.InitChain(4)
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

	type buildFn func([][]interned.TokenID) barkov.GenerativeChain[interned.TokenID]
	cases := []struct {
		n     int
		build buildFn
	}{
		{2, func(c [][]interned.TokenID) barkov.GenerativeChain[interned.TokenID] { return interned.BuildCompressedIndexed2(c) }},
		{3, func(c [][]interned.TokenID) barkov.GenerativeChain[interned.TokenID] { return interned.BuildCompressedIndexed3(c) }},
		{4, func(c [][]interned.TokenID) barkov.GenerativeChain[interned.TokenID] { return interned.BuildCompressedIndexed4(c) }},
		{5, func(c [][]interned.TokenID) barkov.GenerativeChain[interned.TokenID] { return interned.BuildCompressedIndexed5(c) }},
		{6, func(c [][]interned.TokenID) barkov.GenerativeChain[interned.TokenID] { return interned.BuildCompressedIndexed6(c) }},
		{7, func(c [][]interned.TokenID) barkov.GenerativeChain[interned.TokenID] { return interned.BuildCompressedIndexed7(c) }},
		{8, func(c [][]interned.TokenID) barkov.GenerativeChain[interned.TokenID] { return interned.BuildCompressedIndexed8(c) }},
	}
	for _, tc := range cases {
		b.Run(fmt.Sprintf("N=%d", tc.n), func(b *testing.B) {
			for b.Loop() {
				_ = tc.build(encoded)
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

	cases := []struct {
		n     int
		chain barkov.GenerativeChain[interned.TokenID]
	}{
		{2, interned.BuildCompressedIndexed2(encoded)},
		{3, interned.BuildCompressedIndexed3(encoded)},
		{4, interned.BuildCompressedIndexed4(encoded)},
		{5, interned.BuildCompressedIndexed5(encoded)},
		{6, interned.BuildCompressedIndexed6(encoded)},
		{7, interned.BuildCompressedIndexed7(encoded)},
		{8, interned.BuildCompressedIndexed8(encoded)},
	}
	for _, tc := range cases {
		b.Run(fmt.Sprintf("N=%d", tc.n), func(b *testing.B) {
			for b.Loop() {
				for range 10000 {
					barkov.Gen(ctx, tc.chain) //nolint
				}
			}
		})
	}
}
