package barkov

import (
	"bufio"
	"context"
	"os"
	"strings"
	"testing"
)

var testCorpus [][]string
var testChain *Chain[string]
var testCompressed *CompressedChain[string]
var testValidator func([]string) bool
var testSeedTokens []string

func init() {
	f, err := os.Open("testdata/corpus_public.txt")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		// Skip Project Gutenberg metadata lines and blank lines
		if line == "" || strings.HasPrefix(line, "***") || strings.HasPrefix(line, "  ") {
			continue
		}
		tokens := strings.Fields(line)
		if len(tokens) >= 4 {
			testCorpus = append(testCorpus, tokens)
		}
	}

	testChain = InitChain(4).Build(testCorpus)
	testCompressed = testChain.Compress()

	// Build map-based n-gram validator (string equivalent of NGramSet)
	n := testChain.StateSize() + 2
	grams := make(map[string]struct{})
	for _, tokens := range testCorpus {
		for i := 0; i <= len(tokens)-n; i++ {
			grams[strings.Join(tokens[i:i+n], SEP)] = struct{}{}
		}
	}
	testValidator = func(gram []string) bool {
		_, found := grams[strings.Join(gram, SEP)]
		return !found
	}

	// Find a seed state with low END-transition probability so seeded benchmarks
	// produce full sentences rather than fast-failing with ErrSentenceTooShort.
	for _, tokens := range testCorpus {
		if len(tokens) < 20 {
			continue
		}
		// Take a 4-gram from the middle of a long sentence
		mid := len(tokens) / 2
		state := ConstructState(tokens[mid : mid+testChain.stateSize])
		choices, ok := testChain.Model[state]
		if !ok {
			continue
		}
		var total, endCount uint32
		for tok, count := range choices {
			total += count
			if tok == END {
				endCount += count
			}
		}
		if total >= 5 && float64(endCount)/float64(total) < 0.15 {
			testSeedTokens = tokens[mid : mid+testChain.stateSize]
			break
		}
	}
}

// --- Build / Compress ---

func BenchmarkBuild(b *testing.B) {
	for b.Loop() {
		InitChain(4).Build(testCorpus)
	}
}

func BenchmarkCompress(b *testing.B) {
	chain := InitChain(4).Build(testCorpus)
	b.ResetTimer()
	for b.Loop() {
		chain.Compress()
	}
}

// --- State helpers ---

func BenchmarkConstructState(b *testing.B) {
	tokens := []string{"Call", "me", "Ishmael", "Some"}
	for b.Loop() {
		ConstructState(tokens)
	}
}

func BenchmarkDeconstructState(b *testing.B) {
	state := ConstructState([]string{"Call", "me", "Ishmael", "Some"})
	for b.Loop() {
		DeconstructState(state)
	}
}

// --- Generation ---

func BenchmarkGen(b *testing.B) {
	ctx := context.Background()
	for b.Loop() {
		Gen(ctx, testCompressed) //nolint
	}
}

func BenchmarkGenIter(b *testing.B) {
	ctx := context.Background()
	for b.Loop() {
		var out []string
		for tok, err := range GenIter(ctx, testCompressed) {
			if err != nil {
				break
			}
			out = append(out, tok)
		}
		_ = out
	}
}

func BenchmarkGenWithValidator(b *testing.B) {
	ctx := context.Background()
	for b.Loop() {
		Gen(ctx, testCompressed, WithValidator(testValidator)) //nolint
	}
}

func BenchmarkGenThreaded(b *testing.B) {
	ctx := context.Background()
	for b.Loop() {
		Gen(ctx, testCompressed, WithValidator(testValidator), WithThreaded[string]()) //nolint
	}
}

func BenchmarkGenWithSeed(b *testing.B) {
	ctx := context.Background()
	for b.Loop() {
		Gen(ctx, testCompressed, WithSeed(testSeedTokens)) //nolint
	}
}

// --- End-to-end ---

// BenchmarkEndToEnd measures the full pipeline: build the chain from the
// corpus, compress it, and generate one sentence. Tracks cumulative
// speedup across versions.
func BenchmarkEndToEnd(b *testing.B) {
	ctx := context.Background()
	for b.Loop() {
		chain := InitChain(4).Build(testCorpus)
		compressed := chain.Compress()
		Gen(ctx, compressed) //nolint
	}
}

// --- Validator ---

func BenchmarkValidatorCall(b *testing.B) {
	gram := strings.Fields("Call me Ishmael Some years ago never mind")
	b.ResetTimer()
	for b.Loop() {
		testValidator(gram)
	}
}
