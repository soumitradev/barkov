package text_test

import (
	"errors"
	"testing"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/text"
)

func TestValidateAppliesTheOutputContract(t *testing.T) {
	corpus := [][]string{{"a", "b", "c", "d"}, {"x", "b", "y"}}
	g, err := text.New(corpus, text.Order(3), text.AntiVerbatim(3))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		tokens []string
		opts   []text.GenOption
		ok     bool
	}{
		{[]string{"a", "b", "y"}, nil, true},
		{[]string{"a", "b", "c", "d"}, nil, false}, // whole message
		{[]string{"x", "a", "b", "c"}, nil, false}, // copied 3-gram
		{[]string{"x", "a", "b", "c"}, []text.GenOption{text.WithoutCopySteering()}, true},
		{[]string{"a", "b"}, nil, true},
		{[]string{"a", "b"}, []text.GenOption{text.WithoutFragments()}, false}, // piece of one message
		{[]string{"a", "b", "y", "b"}, []text.GenOption{text.WithMaxTokens(3)}, false},
		{[]string{"a", "new", "b", "c"}, nil, true}, // unknown words match nothing
	} {
		err := g.Validate(tc.tokens, tc.opts...)
		if tc.ok != (err == nil) || (err != nil && !errors.Is(err, barkov.ErrSentenceFailedValidation)) {
			t.Errorf("Validate(%q) = %v, want ok=%v", tc.tokens, err, tc.ok)
		}
	}
}
