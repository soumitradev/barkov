package nhash_test

import (
	"strings"
	"testing"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/hashers/fnv"
	"github.com/soumitradev/barkov/v2/nhash"
)

func TestHashNGramSetContains(t *testing.T) {
	corpus := [][]string{
		{"the", "quick", "brown", "fox"},
		{"the", "lazy", "dog"},
	}
	enc := barkov.SepEncoder{Sep: barkov.SEP}
	s := nhash.New(corpus, 3, enc, fnv.FNV{})

	if !s.Contains([]string{"the", "quick", "brown"}) {
		t.Error("expected to contain corpus n-gram")
	}
	if s.Contains([]string{"never", "in", "corpus"}) {
		t.Error("expected to not contain unseen n-gram")
	}
}

func TestHashNGramSetValidator(t *testing.T) {
	corpus := [][]string{{"a", "b", "c", "d"}}
	enc := barkov.SepEncoder{Sep: barkov.SEP}
	s := nhash.New(corpus, 3, enc, fnv.FNV{})
	v := s.Validator()

	if v([]string{"a", "b", "c"}) {
		t.Error("validator should reject corpus n-gram")
	}
	if !v([]string{"x", "y", "z"}) {
		t.Error("validator should accept unseen n-gram")
	}
}

func TestHashNGramSetSize(t *testing.T) {
	corpus := [][]string{{"a", "b", "c", "d"}}
	enc := barkov.SepEncoder{Sep: barkov.SEP}
	s := nhash.New(corpus, 2, enc, fnv.FNV{})
	if s.Size() != 3 {
		t.Errorf("expected 3 n-grams, got %d", s.Size())
	}
}

// plainSepEncoder mirrors barkov.SepEncoder's Encode/Decode output but
// intentionally does NOT implement barkov.AppendEncoder, forcing nhash.New
// and Validator onto the fallback path.
type plainSepEncoder struct{ sep string }

func (e plainSepEncoder) Encode(tokens []string) string {
	return strings.Join(tokens, e.sep)
}

func (e plainSepEncoder) Decode(state string) []string {
	if state == "" {
		return nil
	}
	return strings.Split(state, e.sep)
}

// TestHashNGramSetValidatorClosureEquivalence asserts that the
// AppendEncoder fast-path closure and the fallback closure produce
// identical verdicts across a table of probes.
func TestHashNGramSetValidatorClosureEquivalence(t *testing.T) {
	corpus := [][]string{
		{"the", "quick", "brown", "fox", "jumps"},
		{"the", "lazy", "dog", "sleeps", "soundly"},
		{"a", "b", "c", "d", "e"},
	}
	const n = 3

	fastSet := nhash.New(corpus, n, barkov.SepEncoder{Sep: barkov.SEP}, fnv.FNV{})
	slowSet := nhash.New(corpus, n, plainSepEncoder{sep: barkov.SEP}, fnv.FNV{})

	if fastSet.Size() != slowSet.Size() {
		t.Fatalf("set size mismatch: fast=%d slow=%d", fastSet.Size(), slowSet.Size())
	}

	fast := fastSet.Validator()
	slow := slowSet.Validator()

	probes := [][]string{
		{"the", "quick", "brown"},
		{"quick", "brown", "fox"},
		{"the", "lazy", "dog"},
		{"a", "b", "c"},
		{"c", "d", "e"},
		{"never", "seen", "before"},
		{"the", "quick", "dog"},
		{"", "", ""},
		{"x", "y", "z"},
	}
	for _, p := range probes {
		if fast(p) != slow(p) {
			t.Errorf("verdict mismatch on %v: fast=%v slow=%v", p, fast(p), slow(p))
		}
	}
}
