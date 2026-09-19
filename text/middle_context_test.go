package text_test

import (
	"context"
	"errors"
	"math/rand/v2"
	"slices"
	"sync"
	"testing"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/text"
)

func TestMiddleSeedUsesObservedInteriorContext(t *testing.T) {
	for _, floor := range []int{2, 3} {
		g, err := text.New([][]string{{"hello", "bonnie", "likes", "music"}, {"dear", "bonnie", "likes", "coffee"}}, text.MinOrder(floor), text.AntiVerbatim(0), text.Bidirectional())
		if err != nil {
			t.Fatal(err)
		}
		if _, err := g.Generate(context.Background(), text.Seed([]string{"bonnie"})); !errors.Is(err, barkov.ErrStateNotFound) {
			t.Fatalf("start seed unexpectedly acquired interior context: %v", err)
		}
		for range 20 {
			out, err := g.Generate(context.Background(), text.SeedMiddle([]string{"bonnie"}))
			i := slices.Index(out, "bonnie")
			if err != nil || i < 0 || i >= len(out)-1 {
				t.Fatalf("short interior seed failed at floor %d: %v %v", floor, out, err)
			}
		}
	}
}

func TestMissingReverseContextDoesNotDiscardForwardResult(t *testing.T) {
	g, err := text.New([][]string{{"before", "seed", "two", "after"}, {"invented", "elsewhere"}}, text.MinOrder(2), text.AntiVerbatim(0), text.Bidirectional())
	if err != nil {
		t.Fatal(err)
	}
	out, err := g.Generate(context.Background(), text.SeedMiddle([]string{"invented", "seed", "two"}))
	if err != nil || !slices.Equal(out, []string{"invented", "seed", "two", "after"}) {
		t.Fatalf("missing reverse context discarded valid forward half: %v %v", out, err)
	}
}

func TestMiddleValidatesCompleteOutputNotHalves(t *testing.T) {
	for _, tc := range []struct {
		corpus [][]string
		want   []string
	}{
		{[][]string{{"seed", "after"}, {"before", "seed"}}, []string{"before", "seed", "after"}},
		{[][]string{{"before", "seed", "after"}}, []string{"seed", "after"}},
	} {
		g, err := text.New(tc.corpus, text.Bidirectional(), text.Retries(1), text.RNG(rand.New(rand.NewPCG(5, 1))))
		if err != nil {
			t.Fatal(err)
		}
		out, err := g.Generate(context.Background(), text.SeedMiddle([]string{"seed"}))
		if err != nil || !slices.Equal(out, tc.want) {
			t.Fatalf("discarded a valid complete output: %v %v", out, err)
		}
	}
}

func TestVerbatimFallbackIsPerCall(t *testing.T) {
	g, err := text.New([][]string{{"seed", "after"}}, text.Bidirectional(), text.Retries(2))
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			opts := []text.GenOption{text.Seed([]string{"seed"})}
			allow := i%2 == 0
			if allow {
				opts = append(opts, text.WithVerbatimFallback())
			}
			out, err := g.Generate(context.Background(), opts...)
			if allow {
				if err != nil || !slices.Equal(out, []string{"seed", "after"}) {
					t.Errorf("copy fallback failed: %v %v", out, err)
				}
			} else if !errors.Is(err, barkov.ErrSentenceFailedValidation) || out != nil {
				t.Errorf("fallback changed another call's copy policy: %v %v", out, err)
			}
		}()
	}
	wg.Wait()
}

func TestVerbatimFallbackKeepsOtherConstraints(t *testing.T) {
	for _, tc := range []struct {
		name string
		opts []text.Option
		seed []string
		want error
	}{
		{"support", []text.Option{text.MinSupport(3)}, []string{"seed"}, barkov.ErrInsufficientSupport},
		{"length", []text.Option{text.MaxTokens(1)}, []string{"seed"}, barkov.ErrSentenceFailedValidation},
		{"terminal", nil, []string{"after"}, barkov.ErrSentenceFailedValidation},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := append([]text.Option{text.Bidirectional(), text.MinOrder(2), text.Retries(2)}, tc.opts...)
			g, err := text.New([][]string{{"before", "seed", "after"}}, opts...)
			if err != nil {
				t.Fatal(err)
			}
			out, err := g.Generate(context.Background(), text.SeedMiddle(tc.seed), text.WithVerbatimFallback())
			if out != nil || !errors.Is(err, tc.want) {
				t.Fatalf("fallback weakened constraint: %v %v", out, err)
			}
		})
	}
	g, err := text.New([][]string{{"seed", "after"}}, text.Bidirectional())
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if out, err := g.Generate(ctx, text.Seed([]string{"seed"}), text.WithVerbatimFallback()); out != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("fallback restarted canceled request: %v %v", out, err)
	}
}

func TestMiddleSeedDoesNotInventSentenceBoundary(t *testing.T) {
	g, err := text.New([][]string{{"still", "water"}, {"drink", "still", "water", "please"}}, text.MinOrder(2), text.Bidirectional(), text.RNG(rand.New(rand.NewPCG(5, 1))))
	if err != nil {
		t.Fatal(err)
	}
	out, err := g.Generate(context.Background(), text.SeedMiddle([]string{"still", "water"}))
	if err != nil || len(out) < 3 || out[len(out)-1] != "please" {
		t.Fatalf("invented start boundary hid a valid continuation: %v %v", out, err)
	}
}

func TestVerbatimFallbackBacksOffTerminalSeedWithinFloor(t *testing.T) {
	seed := []string{"if", "you", "squeeze", "some", "lemon", "in", "your", "eyes"}
	g, err := text.New([][]string{seed, {"wash", "your", "eyes", "carefully"}}, text.Order(7), text.MinOrder(2), text.AntiVerbatim(6), text.Bidirectional(), text.RNG(rand.New(rand.NewPCG(5, 1))))
	if err != nil {
		t.Fatal(err)
	}
	if out, err := g.Generate(context.Background(), text.SeedMiddle(seed)); out != nil || !errors.Is(err, barkov.ErrSentenceFailedValidation) {
		t.Fatalf("strict generation unexpectedly copied terminal seed: %v %v", out, err)
	}
	out, err := g.Generate(context.Background(), text.SeedMiddle(seed), text.WithVerbatimFallback())
	if err != nil || len(out) != len(seed)+1 || !slices.Equal(out[:len(seed)], seed) || out[len(seed)] != "carefully" {
		t.Fatalf("fallback kept selecting the terminal context: %v %v", out, err)
	}
}
