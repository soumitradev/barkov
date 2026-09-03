package text_test

import (
	"bufio"
	"context"
	"math/rand/v2"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/text"
)

// smallCorpus has branching (the->quick->brown|red) so generation is not a
// single deterministic path, and contains "the" for the desugar round-trip.
var smallCorpus = [][]string{
	{"the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"},
	{"the", "quick", "brown", "fox", "runs", "away", "fast"},
	{"the", "quick", "red", "fox", "hunts", "at", "night"},
	{"a", "lazy", "dog", "sleeps", "all", "day", "long"},
	{"the", "brown", "dog", "barks", "at", "the", "fox"},
}

func loadPublicCorpus(t *testing.T) [][]string {
	t.Helper()
	f, err := os.Open("../testdata/corpus_public.txt")
	if err != nil {
		t.Fatalf("open corpus: %v", err)
	}
	defer f.Close()

	var corpus [][]string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, "***") || strings.HasPrefix(line, "  ") {
			continue
		}
		if tokens := strings.Fields(line); len(tokens) >= 4 {
			corpus = append(corpus, tokens)
		}
	}
	return corpus
}

func TestGoldenPath(t *testing.T) {
	g, err := text.New(smallCorpus, text.Order(2), text.AntiVerbatim(0))
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	for i := 0; i < 20; i++ {
		out, err := g.Generate(ctx)
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if len(out) == 0 {
			t.Fatal("empty output")
		}
		for _, w := range out {
			if _, ok := g.Vocab().Lookup(w); !ok {
				t.Errorf("output token %q not in vocabulary", w)
			}
		}
	}

	// Sentence is Generate joined with spaces.
	s, err := g.Sentence(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if s == "" {
		t.Error("empty sentence")
	}
}

func TestSeed(t *testing.T) {
	g, err := text.New(smallCorpus, text.Order(2), text.AntiVerbatim(0))
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	// Surviving seed words lead the output.
	out, err := g.Generate(ctx, text.Seed([]string{"the", "quick"}))
	if err != nil {
		t.Fatal(err)
	}
	if len(out) < 2 || out[0] != "the" || out[1] != "quick" {
		t.Errorf("seeded output should begin with [the quick], got %v", out)
	}

	// Unknown seed words are dropped; the rest still leads.
	out, err = g.Generate(ctx, text.Seed([]string{"the", "notaword", "quick"}))
	if err != nil {
		t.Fatal(err)
	}
	if len(out) < 2 || out[0] != "the" || out[1] != "quick" {
		t.Errorf("unknown seed words should be ignored, got %v", out)
	}

	// All-unknown seed falls back to unseeded generation within the same call.
	out, err = g.Generate(ctx, text.Seed([]string{"zzz", "yyy"}))
	if err != nil {
		t.Fatalf("all-unknown seed should fall back to unseeded, got error: %v", err)
	}
	if len(out) == 0 {
		t.Error("unseeded fallback produced empty output")
	}
}

func TestAntiVerbatimOracle(t *testing.T) {
	corpus := loadPublicCorpus(t)
	if len(corpus) == 0 {
		t.Skip("public corpus unavailable")
	}
	const order = 4
	const window = order + 2

	g, err := text.New(corpus, text.Order(order))
	if err != nil {
		t.Fatal(err)
	}

	// Independent oracles over the raw string corpus.
	oracle := barkov.NewNGramSet(corpus, window, barkov.SepEncoder{Sep: barkov.SEP})
	enc := barkov.SepEncoder{Sep: barkov.SEP}
	messages := make(map[string]struct{}, len(corpus))
	for _, m := range corpus {
		messages[enc.Encode(m)] = struct{}{}
	}

	ctx := context.Background()
	successes := 0
	for i := 0; i < 500; i++ {
		out, err := g.Generate(ctx)
		if err != nil {
			continue // a rejected attempt is fine; it just doesn't count
		}
		successes++

		for j := 0; j+window <= len(out); j++ {
			if oracle.Contains(out[j : j+window]) {
				t.Fatalf("output contains a verbatim %d-gram: %v", window, out[j:j+window])
			}
		}
		if _, bad := messages[enc.Encode(out)]; bad {
			t.Fatalf("output reproduces a complete corpus message: %v", out)
		}
	}
	if successes < 100 {
		t.Fatalf("only %d/500 generations succeeded; expected the corpus to be generable", successes)
	}
}

func TestAntiVerbatimDisabled(t *testing.T) {
	// A single-message corpus: the only path IS the corpus message.
	corpus := [][]string{{"a", "b", "c", "d", "e"}}
	ctx := context.Background()

	// Off: the verbatim-only path is allowed.
	off, err := text.New(corpus, text.Order(2), text.AntiVerbatim(0))
	if err != nil {
		t.Fatal(err)
	}
	out, err := off.Generate(ctx)
	if err != nil {
		t.Fatalf("AntiVerbatim(0) should allow the only path: %v", err)
	}
	if strings.Join(out, " ") != "a b c d e" {
		t.Errorf("got %q, want %q", strings.Join(out, " "), "a b c d e")
	}

	// On (default): every output is verbatim, so all attempts are rejected.
	on, err := text.New(corpus, text.Order(2), text.Retries(8))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := on.Generate(ctx); err == nil {
		t.Error("default anti-verbatim should reject the verbatim-only corpus")
	}
}

func TestTimeout(t *testing.T) {
	// Verbatim-only corpus guarantees every attempt is rejected, so a huge
	// retry budget would spin forever without the timeout.
	corpus := [][]string{{"a", "b", "c", "d", "e"}}
	g, err := text.New(corpus, text.Order(2), text.Timeout(50*time.Millisecond), text.Retries(1_000_000))
	if err != nil {
		t.Fatal(err)
	}

	start := time.Now()
	_, err = g.Generate(context.Background())
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected an error from an always-rejecting generator")
	}
	if elapsed > 2*time.Second {
		t.Errorf("Generate took %v; expected it to respect the ~50ms timeout", elapsed)
	}
}

func TestDesugar(t *testing.T) {
	g, err := text.New(smallCorpus, text.Order(2), text.AntiVerbatim(0))
	if err != nil {
		t.Fatal(err)
	}

	// Chain() drops down to the raw core API.
	out, err := barkov.Gen(context.Background(), g.Chain())
	if err != nil {
		t.Fatalf("raw Gen on Chain(): %v", err)
	}
	if len(out) == 0 {
		t.Fatal("empty output from raw Gen")
	}

	// Vocab() round-trips tokens.
	vocab := g.Vocab()
	id, ok := vocab.Lookup("the")
	if !ok {
		t.Fatal("expected 'the' in vocabulary")
	}
	if got := vocab.Token(id); got != "the" {
		t.Errorf("Token(Lookup(\"the\")) = %q, want \"the\"", got)
	}
	if decoded := vocab.DecodeTokens(out); len(decoded) != len(out) {
		t.Errorf("DecodeTokens length %d != output length %d", len(decoded), len(out))
	}
}

func TestDeterministicRNG(t *testing.T) {
	// A fixed RNG makes output reproducible across generators.
	mk := func() *text.Generator {
		g, err := text.New(smallCorpus,
			text.Order(2),
			text.AntiVerbatim(0),
			text.RNG(rand.New(rand.NewPCG(42, 42))),
		)
		if err != nil {
			t.Fatal(err)
		}
		return g
	}
	a, b := mk(), mk()
	ctx := context.Background()
	for i := 0; i < 10; i++ {
		oa, err := a.Generate(ctx)
		if err != nil {
			t.Fatal(err)
		}
		ob, err := b.Generate(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Join(oa, " ") != strings.Join(ob, " ") {
			t.Fatalf("fixed RNG diverged at %d: %v vs %v", i, oa, ob)
		}
	}
}

func TestNewErrors(t *testing.T) {
	cases := []struct {
		name   string
		corpus [][]string
		opts   []text.Option
	}{
		{"empty corpus", nil, nil},
		{"all-empty messages", [][]string{{}, {}}, nil},
		{"order too low", smallCorpus, []text.Option{text.Order(1)}},
		{"order too high", smallCorpus, []text.Option{text.Order(9)}},
		{"zero retries", smallCorpus, []text.Option{text.Retries(0)}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := text.New(tc.corpus, tc.opts...); err == nil {
				t.Error("expected an error, got nil")
			}
		})
	}
}

func TestMaxTokens(t *testing.T) {
	ctx := context.Background()
	// Construction-time cap: all generated outputs must be <= 5 tokens
	g, err := text.New(smallCorpus, text.Order(2), text.AntiVerbatim(0), text.MaxTokens(5), text.Retries(32))
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 20; i++ {
		out, err := g.Generate(ctx)
		if err != nil {
			t.Fatalf("Generate with MaxTokens(5): %v", err)
		}
		if len(out) > 5 {
			t.Fatalf("expected output length <= 5, got %d: %v", len(out), out)
		}
	}

	// Per-call override: limit to 4 tokens
	for i := 0; i < 10; i++ {
		out, err := g.Generate(ctx, text.WithMaxTokens(4))
		if err != nil {
			t.Fatalf("Generate with WithMaxTokens(4): %v", err)
		}
		if len(out) > 4 {
			t.Fatalf("expected output length <= 4, got %d: %v", len(out), out)
		}
	}
}

func TestSeedLowercaseFallback(t *testing.T) {
	g, err := text.New(smallCorpus, text.Order(2), text.AntiVerbatim(0))
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	// Uppercase seed should resolve to lowercase corpus word "quick"
	out, err := g.Generate(ctx, text.Seed([]string{"QUICK"}))
	if err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 || out[0] != "quick" {
		t.Errorf("expected 'QUICK' to resolve to 'quick' at start of output, got %v", out)
	}
}

func TestStatesIteration(t *testing.T) {
	g, err := text.New(smallCorpus, text.Order(2), text.AntiVerbatim(0))
	if err != nil {
		t.Fatal(err)
	}

	order1Count := 0
	for words, count := range g.States(1) {
		if len(words) != 1 {
			t.Fatalf("expected 1 word for order 1, got %d: %v", len(words), words)
		}
		if count == 0 {
			t.Errorf("expected non-zero count for state %v", words)
		}
		order1Count++
	}
	if order1Count == 0 {
		t.Error("expected >0 states at order 1")
	}

	order2Count := 0
	for words, count := range g.States(2) {
		if len(words) != 2 {
			t.Fatalf("expected 2 words for order 2, got %d: %v", len(words), words)
		}
		if count == 0 {
			t.Errorf("expected non-zero count for state %v", words)
		}
		order2Count++
	}
	if order2Count == 0 {
		t.Error("expected >0 states at order 2")
	}
}

func TestMessageSet(t *testing.T) {
	g, err := text.New(smallCorpus, text.Order(2))
	if err != nil {
		t.Fatal(err)
	}
	ms := g.MessageSet()
	if ms == nil {
		t.Fatal("expected MessageSet to be non-nil when AntiVerbatim is active")
	}
	if ms.Size() != len(smallCorpus) {
		t.Errorf("expected MessageSet size %d, got %d", len(smallCorpus), ms.Size())
	}
}

func TestBidirectionalSeedMiddle(t *testing.T) {
	ctx := context.Background()

	// Generator without Bidirectional() should reject SeedMiddle
	unibi, err := text.New(smallCorpus, text.Order(2), text.AntiVerbatim(0))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := unibi.Generate(ctx, text.SeedMiddle([]string{"fox"})); err == nil {
		t.Error("expected error when calling SeedMiddle without Bidirectional()")
	}

	// Generator with Bidirectional()
	bidi, err := text.New(smallCorpus, text.Order(2), text.AntiVerbatim(0), text.Bidirectional())
	if err != nil {
		t.Fatal(err)
	}
	if bidi.ReverseChain() == nil {
		t.Fatal("expected ReverseChain to be non-nil under Bidirectional()")
	}

	// SeedMiddle should produce an output containing the seed
	for i := 0; i < 20; i++ {
		out, err := bidi.Generate(ctx, text.SeedMiddle([]string{"fox"}))
		if err != nil {
			t.Fatalf("SeedMiddle: %v", err)
		}
		found := false
		for _, w := range out {
			if w == "fox" {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected 'fox' in middle-seeded output, got %v", out)
		}
	}
}

func TestGeneratorContains(t *testing.T) {
	g, err := text.New(smallCorpus, text.Order(2), text.AntiVerbatim(0))
	if err != nil {
		t.Fatal(err)
	}

	if !g.Contains("the") {
		t.Error("expected Contains('the') to be true")
	}
	if !g.Contains("THE") {
		t.Error("expected Contains('THE') to be true (lowercase fallback)")
	}
	if g.Contains("nonexistenttoken123") {
		t.Error("expected Contains('nonexistenttoken123') to be false")
	}
}

func TestGeneratorSetRNG(t *testing.T) {
	g1, err := text.New(smallCorpus, text.Order(2), text.AntiVerbatim(0))
	if err != nil {
		t.Fatal(err)
	}
	g2, err := text.New(smallCorpus, text.Order(2), text.AntiVerbatim(0))
	if err != nil {
		t.Fatal(err)
	}

	g1.SetRNG(rand.New(rand.NewPCG(123, 456)))
	g2.SetRNG(rand.New(rand.NewPCG(123, 456)))

	ctx := context.Background()
	for i := 0; i < 10; i++ {
		out1, err := g1.Generate(ctx)
		if err != nil {
			t.Fatal(err)
		}
		out2, err := g2.Generate(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(out1, out2) {
			t.Fatalf("outputs diverged after SetRNG at step %d: %v vs %v", i, out1, out2)
		}
	}
}
