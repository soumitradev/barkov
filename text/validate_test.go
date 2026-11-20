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
		cause  error // nil when the tokens are valid
	}{
		{[]string{"a", "b", "y"}, nil, nil},
		{[]string{"a", "b", "c", "d"}, nil, barkov.ErrCopiesCorpus}, // whole message
		{[]string{"x", "a", "b", "c"}, nil, barkov.ErrCopiesCorpus}, // copied 3-gram
		{[]string{"x", "a", "b", "c"}, []text.GenOption{text.WithoutCopySteering()}, nil},
		{[]string{"a", "b"}, nil, nil},
		{[]string{"a", "b"}, []text.GenOption{text.WithoutFragments()}, barkov.ErrCopiesCorpus}, // piece of one message
		{[]string{"a", "b", "y", "b"}, []text.GenOption{text.WithMaxTokens(3)}, barkov.ErrTokenLimit},
		{[]string{"a", "new", "b", "c"}, nil, nil}, // unknown words match nothing
	} {
		err := g.Validate(tc.tokens, tc.opts...)
		if tc.cause == nil {
			if err != nil {
				t.Errorf("Validate(%q) = %v, want nil", tc.tokens, err)
			}
			continue
		}
		if !errors.Is(err, barkov.ErrSentenceFailedValidation) || !errors.Is(err, tc.cause) {
			t.Errorf("Validate(%q) = %v, want %v", tc.tokens, err, tc.cause)
		}
	}
}
