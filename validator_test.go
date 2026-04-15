package barkov

import "testing"

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
