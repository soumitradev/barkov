package interned_test

import (
	"bufio"
	"context"
	"os"
	"strings"
	"testing"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/interned"
)

var internedCorpus [][]string

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
