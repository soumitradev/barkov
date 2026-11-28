package interned_test

import (
	"bufio"
	"context"
	"os"
	"strings"
	"testing"

	"github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/interned"
)

func loadCorpus(tb testing.TB) [][]string {
	f, err := os.Open("../testdata/corpus_public.txt")
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()

	var corpus [][]string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		// Skip Project Gutenberg metadata lines and blank lines
		if line == "" || strings.HasPrefix(line, "***") || strings.HasPrefix(line, "  ") {
			continue
		}
		if tokens := strings.Fields(line); len(tokens) >= 4 {
			corpus = append(corpus, tokens)
		}
	}
	if err := scanner.Err(); err != nil {
		tb.Fatal(err)
	}
	return corpus
}

// BenchmarkGenInterned runs 10000 sentences per op to match canary's
// BenchmarkGenHeavyInterned, so the numbers compare directly
func BenchmarkGenInterned(b *testing.B) {
	vocab := interned.NewVocabulary()
	chain := interned.Build(4, vocab.InternCorpus(loadCorpus(b)))
	ctx := context.Background()
	for b.Loop() {
		for range 10000 {
			barkov.Gen(ctx, chain) //nolint
		}
	}
}
