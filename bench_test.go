package barkov

import (
	"bufio"
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

var testCorpus [][]string
var testChain *Chain
var testCompressed *CompressedChain
var testValidator func([]string) bool
var testSeedState State // a valid starting state for seeded generation benchmarks

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
	n := testChain.getStateSize() + 2
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
		total, endCount := 0, 0
		for tok, count := range choices {
			total += count
			if tok == END {
				endCount += count
			}
		}
		if total >= 5 && float64(endCount)/float64(total) < 0.15 {
			testSeedState = state
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

func BenchmarkChooseToken(b *testing.B) {
	cumDist := []int{10, 25, 50, 80, 100, 150, 200, 300, 500, 1000}
	for b.Loop() {
		chooseToken(cumDist)
	}
}

// --- Move ---

func BenchmarkMoveUncompressed(b *testing.B) {
	b.ResetTimer()
	for b.Loop() {
		testChain.move(testSeedState) //nolint
	}
}

func BenchmarkMoveCompressed(b *testing.B) {
	b.ResetTimer()
	for b.Loop() {
		testCompressed.move(testSeedState) //nolint
	}
}

// --- Unseeded generation ---

func BenchmarkGenUncompressed(b *testing.B) {
	for b.Loop() {
		Gen(testChain) //nolint
	}
}

func BenchmarkGenCompressed(b *testing.B) {
	for b.Loop() {
		Gen(testCompressed) //nolint
	}
}

func BenchmarkGenPruned(b *testing.B) {
	for b.Loop() {
		GenPruned(testCompressed, testValidator) //nolint
	}
}

func BenchmarkGenThreaded(b *testing.B) {
	for b.Loop() {
		GenThreaded(testCompressed, testValidator, 30*time.Second) //nolint
	}
}

// --- Seeded generation ---

func BenchmarkGenWithStart(b *testing.B) {
	for b.Loop() {
		GenWithStart(testCompressed, testSeedState) //nolint
	}
}

func BenchmarkGenPrunedWithStart(b *testing.B) {
	for b.Loop() {
		GenPrunedWithStart(testCompressed, testSeedState, testValidator) //nolint
	}
}

func BenchmarkGenThreadedWithStart(b *testing.B) {
	for b.Loop() {
		GenThreadedWithStart(testCompressed, testSeedState, testValidator, 30*time.Second) //nolint
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

// --- Generic API benchmarks (Phase B) ---

var testGenericChain *GenericChain[string]
var testGenericCompressed *GenericCompressedChain[string]
var testGenericSeedTokens []string

func init() {
	cfg := ChainConfig[string]{
		StateSize: 4,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Encoder:   SepEncoder{Sep: SEP},
	}
	testGenericChain = NewGenericChain(cfg).BuildRaw(testCorpus)
	testGenericCompressed = testGenericChain.Compress()

	// Convert seed state to tokens
	testGenericSeedTokens = DeconstructState(testSeedState)
}

func BenchmarkGenGenericString(b *testing.B) {
	ctx := context.Background()
	for b.Loop() {
		GenGeneric(ctx, testGenericCompressed) //nolint
	}
}

func BenchmarkGenGenericStringIter(b *testing.B) {
	ctx := context.Background()
	for b.Loop() {
		var out []string
		for tok, err := range GenIterGeneric(ctx, testGenericCompressed) {
			if err != nil {
				break
			}
			out = append(out, tok)
		}
		_ = out
	}
}

func BenchmarkGenGenericStringWithValidator(b *testing.B) {
	ctx := context.Background()
	for b.Loop() {
		GenGeneric(ctx, testGenericCompressed, WithGenericValidator(testValidator)) //nolint
	}
}

func BenchmarkGenGenericStringThreaded(b *testing.B) {
	ctx := context.Background()
	for b.Loop() {
		GenGeneric(ctx, testGenericCompressed, WithGenericValidator(testValidator), WithGenericThreaded[string]()) //nolint
	}
}

func BenchmarkGenGenericStringWithSeed(b *testing.B) {
	ctx := context.Background()
	for b.Loop() {
		GenGeneric(ctx, testGenericCompressed, WithGenericSeed(testGenericSeedTokens)) //nolint
	}
}
