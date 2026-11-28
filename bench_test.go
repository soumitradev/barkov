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

// Construct a validator that rejects a width-sized token window that appears verbatim in the corpus
func novelNGrams(corpus [][]string, width int) func([]string) bool {
	seen := make(map[string]struct{})
	for _, tokens := range corpus {
		for i := 0; i+width <= len(tokens); i++ {
			seen[strings.Join(tokens[i:i+width], SEP)] = struct{}{}
		}
	}
	return func(history []string) bool {
		if len(history) < width {
			return true
		}
		_, copied := seen[strings.Join(history[len(history)-width:], SEP)]
		return !copied
	}
}

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
	if err := scanner.Err(); err != nil {
		panic(err)
	}

	testChain = InitChain(4).Build(testCorpus)
	testCompressed = testChain.Compress()

	testValidator = novelNGrams(testCorpus, testChain.StateSize()+2)

	// Find a seed state with low END-transition probability so seeded benchmarks
	// produce full sentences rather than fast-failing with ErrSentenceTooShort
	for _, tokens := range testCorpus {
		if len(tokens) < 20 {
			continue
		}
		// Take a 4-gram from the middle of a long sentence
		mid := len(tokens) / 2
		state := joinKey(tokens[mid : mid+testChain.stateSize])
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

// --- Generation ---

func BenchmarkGen(b *testing.B) {
	ctx := context.Background()
	for b.Loop() {
		Gen(ctx, testCompressed)
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
		Gen(ctx, testCompressed, WithValidator(testValidator))
	}
}

func BenchmarkGenThreaded(b *testing.B) {
	ctx := context.Background()
	for b.Loop() {
		Gen(ctx, testCompressed, WithValidator(testValidator), WithParallelism[string](8))
	}
}

func BenchmarkGenWithSeed(b *testing.B) {
	ctx := context.Background()
	for b.Loop() {
		Gen(ctx, testCompressed, WithSeed(testSeedTokens))
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
