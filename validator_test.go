package barkov

import (
	"strings"
	"testing"
)

func TestNGramSetContains(t *testing.T) {
	corpus := [][]string{
		{"the", "quick", "brown", "fox"},
		{"the", "lazy", "dog"},
	}
	enc := SepEncoder{Sep: SEP}
	s := NewNGramSet(corpus, 3, enc)

	if !s.Contains([]string{"the", "quick", "brown"}) {
		t.Error("expected to contain corpus n-gram")
	}
	if s.Contains([]string{"never", "in", "corpus"}) {
		t.Error("expected to not contain unseen n-gram")
	}
}

func TestNGramSetValidator(t *testing.T) {
	corpus := [][]string{{"a", "b", "c", "d"}}
	enc := SepEncoder{Sep: SEP}
	s := NewNGramSet(corpus, 3, enc)
	v := s.Validator()

	if v([]string{"a", "b", "c"}) {
		t.Error("validator should reject corpus n-gram")
	}
	if !v([]string{"x", "y", "z"}) {
		t.Error("validator should accept unseen n-gram")
	}
}

func TestNGramSetSize(t *testing.T) {
	corpus := [][]string{{"a", "b", "c", "d"}}
	enc := SepEncoder{Sep: SEP}
	s := NewNGramSet(corpus, 2, enc)
	// n-grams: [a,b], [b,c], [c,d] = 3
	if s.Size() != 3 {
		t.Errorf("expected 3 n-grams, got %d", s.Size())
	}
}

// plainSepEncoder mirrors SepEncoder's Encode/Decode output but intentionally
// does NOT implement AppendEncoder, forcing NewNGramSet and Validator onto
// the fallback path. Used to verify fast-path and fallback agree.
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

// TestNGramSetValidatorClosureEquivalence asserts that the AppendEncoder
// fast-path closure and the fallback closure produce identical verdicts
// across a table of probes. Both variants must see the same encoded keys
// because plainSepEncoder's Encode matches SepEncoder's Encode byte-for-byte.
func TestNGramSetValidatorClosureEquivalence(t *testing.T) {
	corpus := [][]string{
		{"the", "quick", "brown", "fox", "jumps"},
		{"the", "lazy", "dog", "sleeps", "soundly"},
		{"a", "b", "c", "d", "e"},
	}
	const n = 3

	fastSet := NewNGramSet(corpus, n, SepEncoder{Sep: SEP})
	slowSet := NewNGramSet(corpus, n, plainSepEncoder{sep: SEP})

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
