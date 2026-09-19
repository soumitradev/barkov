package text_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/interned"
	"github.com/soumitradev/barkov/v2/text"
)

func TestTextMinimumOrderWiredToBothDirections(t *testing.T) {
	corpus := [][]string{}
	for range 3 {
		corpus = append(corpus, []string{"p", "a", "b", "c"}, []string{"q", "z", "b", "d"})
	}
	for _, floor := range []int{1, 2} {
		g, err := text.New(corpus, text.Order(2), text.MinOrder(floor), text.AntiVerbatim(3), text.Bidirectional(), text.Retries(1))
		if err != nil {
			t.Fatal(err)
		}
		out, err := g.Generate(context.Background(), text.Seed([]string{"p", "a", "b"}))
		if floor == 1 {
			if err != nil || strings.Join(out, " ") != "p a b d" {
				t.Fatalf("default fallback: %v %v", out, err)
			}
		} else if !errors.Is(err, barkov.ErrSentenceFailedValidation) {
			t.Fatalf("minimum ignored in text.Generate: %v %v", out, err)
		}
		p, _ := g.Vocab().Lookup("p")
		b, _ := g.Vocab().Lookup("b")
		window := []interned.TokenID{p, p, p, b}
		reverse := g.ReverseChain()
		_, err = reverse.Move(reverse.Encoder().Encode(window))
		if floor == 1 && err != nil {
			t.Fatalf("reverse default fallback: %v", err)
		}
		if floor == 2 && !errors.Is(err, barkov.ErrStateNotFound) {
			t.Fatalf("reverse minimum ignored: %v", err)
		}
	}
	for _, floor := range []int{0, -1, 5} {
		if _, err := text.New(corpus, text.Order(2), text.MinOrder(floor)); err == nil {
			t.Errorf("accepted floor %d above/below 1..4", floor)
		}
	}
}
