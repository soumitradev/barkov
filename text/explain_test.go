package text_test

import (
	"context"
	"errors"
	"math/rand/v2"
	"slices"
	"testing"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/text"
)

// A walk only emits what the context it sampled from offers, so replaying a
// real output with the walk its trace recorded must explain every step
func TestExplainAccountsForEveryStepOfGeneratedOutput(t *testing.T) {
	corpus := loadPublicCorpus(t)
	g, err := text.New(corpus, text.Order(4), text.Bidirectional())
	if err != nil {
		t.Fatal(err)
	}
	g.SetRNG(rand.New(rand.NewPCG(7, 7)))
	for _, tc := range []struct {
		name string
		opt  func() text.GenOption
	}{
		{"unseeded", func() text.GenOption { return text.WithMaxTokens(30) }},
		{"seeded start", func() text.GenOption { return text.Seed([]string{"the"}) }},
		{"seeded middle", func() text.GenOption { return text.SeedMiddle([]string{"of", "the"}) }},
		{"unsteered middle", func() text.GenOption { return text.SeedMiddle([]string{"said"}) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backward := false
			for range 40 {
				var trace text.Trace
				opts := []text.GenOption{tc.opt(), text.WithTrace(&trace)}
				if tc.name == "unsteered middle" {
					opts = append(opts, text.WithoutCopySteering())
				}
				out, err := g.Generate(context.Background(), opts...)
				if err != nil {
					t.Fatal(err)
				}
				steps, err := g.Explain(out, trace.Walk)
				if err != nil {
					t.Fatal(err)
				}
				backward = backward || trace.Walk.Backward
				if last := trace.Attempts[len(trace.Attempts)-1]; last.Err != nil || !slices.Equal(last.Output, out) {
					t.Fatalf("last attempt %+v: want the returned %q", last, out)
				}
				for _, s := range steps {
					if s.Used == 0 {
						t.Fatalf("%q walked as %+v: step %d (%q, end %v) offered by no sampled context", out, trace.Walk, s.Index, s.Token, s.End)
					}
				}
				if want := len(out) - trace.Walk.SeedLen + 1 + boolInt(trace.Walk.Backward); len(steps) != want {
					t.Fatalf("%q walked as %+v: %d steps, want one per walked token and end (%d)", out, trace.Walk, len(steps), want)
				}
			}
			if tc.name != "unseeded" && tc.name != "seeded start" && !backward {
				t.Fatal("no output walked backwards from the seed, so the backward replay went untested")
			}
		})
	}
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func TestExplainFlagsTokensTheWalkCouldNotHaveChosen(t *testing.T) {
	g, err := text.New([][]string{{"the", "cat", "sat"}, {"a", "cat", "ran"}}, text.Order(2), text.AntiVerbatim(0))
	if err != nil {
		t.Fatal(err)
	}
	// After "the cat" the longest context only ever saw "sat"
	steps, err := g.Explain([]string{"the", "cat", "ran"}, text.Walk{})
	if err != nil {
		t.Fatal(err)
	}
	if got := steps[2]; got.Token != "ran" || got.Used != 0 {
		t.Fatalf("step %+v: want ran unexplained", got)
	}
	steps, err = g.Explain([]string{"the", "cat", "sat"}, text.Walk{})
	if err != nil {
		t.Fatal(err)
	}
	if got := steps[2]; got.Token != "sat" || got.Used == 0 {
		t.Fatalf("step %+v: want sat explained", got)
	}
}

func TestExplainDoesNotCountTokensCopySteeringVetoes(t *testing.T) {
	corpus := [][]string{{"a", "cat", "sat"}, {"b", "cat", "ran"}, {"cat", "naps"}}
	g, err := text.New(corpus, text.Order(2), text.AntiVerbatim(3))
	if err != nil {
		t.Fatal(err)
	}
	// "a cat sat" would copy a corpus run, while "a cat ran" recombines
	for _, tc := range []struct {
		token     string
		explained bool
	}{{"sat", false}, {"ran", true}} {
		steps, err := g.Explain([]string{"a", "cat", tc.token}, text.Walk{})
		if err != nil {
			t.Fatal(err)
		}
		if got := steps[2]; (got.Used != 0) != tc.explained {
			t.Errorf("step %+v: explained %v, want %v", got, got.Used != 0, tc.explained)
		}
	}
}

func TestExplainShowsTheEndingsALongerContextOffered(t *testing.T) {
	g, err := text.New([][]string{{"boiled", "uncle", "jokes"}, {"bad", "uncle", "jokes", "are", "funny"}, {"uncle", "jokes", "again"}}, text.Order(2), text.AntiVerbatim(0))
	if err != nil {
		t.Fatal(err)
	}
	steps, err := g.Explain([]string{"boiled", "uncle", "jokes"}, text.Walk{})
	if err != nil {
		t.Fatal(err)
	}
	end := steps[len(steps)-1]
	if !end.End || end.Index != 3 {
		t.Fatalf("last step %+v: want the end after three tokens", end)
	}
	// The longest context is the whole first message, which only ever ended,
	// while the shorter "uncle jokes" ended once in three
	if longest := end.Contexts[0]; len(longest.Choices) != 1 || !longest.Choices[0].End || end.Used != longest.Order {
		t.Fatalf("step %+v: want the walk to sample the longest context, which only ends", end)
	}
	i := slices.IndexFunc(end.Contexts, func(c text.Context) bool { return slices.Equal(c.Words, []string{"uncle", "jokes"}) })
	if i < 0 {
		t.Fatalf("contexts %+v: want uncle jokes among them", end.Contexts)
	}
	ended := 0
	for _, c := range end.Contexts[i].Choices {
		if c.End {
			ended += int(c.Count)
		}
	}
	if ended != 1 || end.Contexts[i].Observations != 3 {
		t.Fatalf("uncle jokes %+v: want 1 ending in 3", end.Contexts[i])
	}
}

func TestExplainMarksStartAndUnknownContext(t *testing.T) {
	g, err := text.New([][]string{{"x", "seed", "y"}, {"seed", "z"}}, text.Order(3), text.AntiVerbatim(0), text.Bidirectional())
	if err != nil {
		t.Fatal(err)
	}
	steps, err := g.Explain([]string{"x", "seed", "y"}, text.Walk{Seed: 1, SeedLen: 1, Middle: true, Backward: true})
	if err != nil {
		t.Fatal(err)
	}
	if first := steps[0]; first.Index != -1 || !first.Backward || first.Contexts[0].Words[len(first.Contexts[0].Words)-1] != text.Unknown {
		t.Fatalf("first step %+v: want the backward start, with unknown history after the seed", first)
	}
	steps, err = g.Explain([]string{"seed", "z"}, text.Walk{SeedLen: 1})
	if err != nil {
		t.Fatal(err)
	}
	if words := steps[0].Contexts[0].Words; words[0] != text.Start {
		t.Fatalf("context %q: want a message start before a start seed", words)
	}
}

func TestExplainRejectsSeedOutsideTokens(t *testing.T) {
	g, err := text.New(smallCorpus, text.Bidirectional())
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []text.Walk{{Seed: 2, SeedLen: 2}, {Seed: 1, SeedLen: 1}, {Seed: -1, SeedLen: 1}} {
		if _, err := g.Explain([]string{"the", "quick", "fox"}, w); err == nil {
			t.Errorf("walk %+v: want an error", w)
		}
	}
	if _, err := g.Explain([]string{"the", "quick", "fox"}, text.Walk{Seed: 1, SeedLen: 1, Middle: true, Backward: true}); err != nil {
		t.Errorf("seed inside tokens: %v", err)
	}
}

func TestTraceRecordsRejectedAttemptsAndTheKeptOne(t *testing.T) {
	corpus := [][]string{{"go", "home", "now"}, {"go", "home", "later", "tonight", "again"}, {"we", "go", "home", "now", "ok"}}
	g, err := text.New(corpus, text.AntiVerbatim(0), text.MaxTokens(3), text.Retries(500))
	if err != nil {
		t.Fatal(err)
	}
	g.SetRNG(rand.New(rand.NewPCG(3, 3)))
	rejected := 0
	for range 20 {
		var trace text.Trace
		out, err := g.Generate(context.Background(), text.WithTrace(&trace))
		if err != nil {
			t.Fatal(err)
		}
		last := trace.Attempts[len(trace.Attempts)-1]
		if last.Err != nil || !slices.Equal(last.Output, out) {
			t.Fatalf("last attempt %+v: want the returned %q", last, out)
		}
		for _, a := range trace.Attempts[:len(trace.Attempts)-1] {
			if !errors.Is(a.Err, barkov.ErrSentenceFailedValidation) {
				t.Fatalf("rejected attempt %+v: want a validation rejection", a)
			}
			rejected++
		}
	}
	// Two of the three messages run past 3 tokens
	if rejected == 0 {
		t.Fatal("no attempt was rejected for length")
	}
}

func TestTraceStartsEmptyForEachGeneration(t *testing.T) {
	g, err := text.New(smallCorpus)
	if err != nil {
		t.Fatal(err)
	}
	stale := text.Attempt{Output: []string{"stale"}}
	trace := text.Trace{Attempts: []text.Attempt{stale}}
	if _, err := g.Generate(context.Background(), text.WithTrace(&trace)); err != nil {
		t.Fatal(err)
	}
	if len(trace.Attempts) == 0 || slices.Equal(trace.Attempts[0].Output, stale.Output) {
		t.Fatalf("attempts %+v: want only this generation's", trace.Attempts)
	}
}
