package nhash_test

import (
	"strings"
	"sync"
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

func TestHashNGramSetValidateMatchesValidator(t *testing.T) {
	corpus := [][]string{{"a", "b", "c", "d"}}
	enc := barkov.SepEncoder{Sep: barkov.SEP}
	s := nhash.New(corpus, 3, enc, fnv.FNV{})
	v := s.Validator()

	probes := [][]string{
		{"a", "b", "c"},
		{"b", "c", "d"},
		{"x", "y", "z"},
	}
	for _, p := range probes {
		if s.Validate(p) != v(p) {
			t.Errorf("Validate(%v)=%v disagrees with Validator()(%v)=%v", p, s.Validate(p), p, v(p))
		}
	}
}

func TestHashNGramSetN(t *testing.T) {
	corpus := [][]string{{"a", "b", "c", "d"}}
	enc := barkov.SepEncoder{Sep: barkov.SEP}
	s := nhash.New(corpus, 3, enc, fnv.FNV{})
	if s.N() != 3 {
		t.Errorf("expected N()=3, got %d", s.N())
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
		if fastSet.Contains(p) != slowSet.Contains(p) {
			t.Errorf("Contains mismatch on %v: fast=%v slow=%v", p, fastSet.Contains(p), slowSet.Contains(p))
		}
	}
}

func TestHashNGramSetLargeAndEmptyGrams(t *testing.T) {
	mid := strings.Repeat("a", 150)
	huge := strings.Repeat("b", 2000)
	corpus := [][]string{
		{"short", "hit", "gram"},
		{mid, mid, mid},
		{huge, huge, huge},
	}

	fast := nhash.New(corpus, 3, barkov.DefaultStringEncoder, fnv.FNV{})
	slow := nhash.New(corpus, 3, plainSepEncoder{sep: barkov.SEP}, fnv.FNV{})

	probes := [][]string{
		{"short", "hit", "gram"},
		{mid, mid, mid},
		{huge, huge, huge},
		{"short", "hit", "gram"},
		{huge, huge, "miss"},
		{},
		nil,
	}
	for _, p := range probes {
		if fast.Contains(p) != slow.Contains(p) {
			t.Errorf("Contains mismatch on len=%d: fast=%v slow=%v", len(p), fast.Contains(p), slow.Contains(p))
		}
		if fast.Validate(p) != !fast.Contains(p) {
			t.Errorf("Validate did not invert Contains on len=%d", len(p))
		}
	}
}

func TestHashNGramSetValidateConcurrent(t *testing.T) {
	longTok := strings.Repeat("x", 150)
	corpus := [][]string{
		{"the", "quick", "brown", "fox"},
		{longTok, longTok, longTok},
	}
	s := nhash.New(corpus, 3, barkov.DefaultStringEncoder, fnv.FNV{})
	v := s.Validator()

	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 200 {
				if s.Validate([]string{"the", "quick", "brown"}) {
					t.Errorf("accepted corpus hit")
					return
				}
				if !v([]string{"never", "seen", "gram"}) {
					t.Errorf("rejected unseen gram")
					return
				}
				if !s.Contains([]string{longTok, longTok, longTok}) {
					t.Errorf("missed oversized corpus hit")
					return
				}
			}
		}()
	}
	wg.Wait()
}

func BenchmarkHashNGramSetValidate(b *testing.B) {
	corpus := [][]string{
		{"the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"},
		{"a", "quick", "brown", "fox", "is", "faster", "than", "a", "lazy", "dog"},
	}
	hit := []string{"the", "quick", "brown"}
	miss := []string{"never", "in", "corpus"}

	b.Run("AppendEncoder", func(b *testing.B) {
		s := nhash.New(corpus, 3, barkov.DefaultStringEncoder, fnv.FNV{})
		b.ResetTimer()
		for b.Loop() {
			_ = s.Validate(hit)
			_ = s.Validate(miss)
		}
	})

	b.Run("AppendEncoderParallel", func(b *testing.B) {
		s := nhash.New(corpus, 3, barkov.DefaultStringEncoder, fnv.FNV{})
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				_ = s.Validate(hit)
				_ = s.Validate(miss)
			}
		})
	})

	b.Run("Fallback", func(b *testing.B) {
		s := nhash.New(corpus, 3, plainSepEncoder{sep: barkov.SEP}, fnv.FNV{})
		b.ResetTimer()
		for b.Loop() {
			_ = s.Validate(hit)
			_ = s.Validate(miss)
		}
	})
}
