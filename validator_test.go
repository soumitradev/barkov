package barkov

import (
	"strings"
	"sync"
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

func TestNGramSetValidateMatchesValidator(t *testing.T) {
	corpus := [][]string{{"a", "b", "c", "d"}}
	enc := SepEncoder{Sep: SEP}
	s := NewNGramSet(corpus, 3, enc)
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

func TestNGramSetN(t *testing.T) {
	corpus := [][]string{{"a", "b", "c", "d"}}
	enc := SepEncoder{Sep: SEP}
	s := NewNGramSet(corpus, 3, enc)
	if s.N() != 3 {
		t.Errorf("expected N()=3, got %d", s.N())
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
		if fastSet.Contains(p) != slowSet.Contains(p) {
			t.Errorf("Contains mismatch on %v: fast=%v slow=%v", p, fastSet.Contains(p), slowSet.Contains(p))
		}
	}
}

func TestNGramSetLargeAndEmptyGrams(t *testing.T) {
	mid := strings.Repeat("a", 150)   // 3 tokens > 256B initial scratch cap
	huge := strings.Repeat("b", 2000) // 3 tokens > 4096B maxPooledScratch
	corpus := [][]string{
		{"short", "hit", "gram"},
		{mid, mid, mid},
		{huge, huge, huge},
	}

	fast := NewNGramSet(corpus, 3, DefaultStringEncoder)
	slow := NewNGramSet(corpus, 3, plainSepEncoder{sep: SEP})

	probes := [][]string{
		{"short", "hit", "gram"},
		{mid, mid, mid},
		{huge, huge, huge},
		{"short", "hit", "gram"}, // re-probe after oversized scratch was discarded
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

func TestNGramSetValidateConcurrent(t *testing.T) {
	longTok := strings.Repeat("x", 150)
	corpus := [][]string{
		{"the", "quick", "brown", "fox"},
		{longTok, longTok, longTok},
	}
	s := NewNGramSet(corpus, 3, DefaultStringEncoder)
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

func BenchmarkNGramSetValidate(b *testing.B) {
	corpus := [][]string{
		{"the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"},
		{"a", "quick", "brown", "fox", "is", "faster", "than", "a", "lazy", "dog"},
	}
	hit := []string{"the", "quick", "brown"}
	miss := []string{"never", "in", "corpus"}

	b.Run("AppendEncoder", func(b *testing.B) {
		s := NewNGramSet(corpus, 3, DefaultStringEncoder)
		b.ResetTimer()
		for b.Loop() {
			_ = s.Validate(hit)
			_ = s.Validate(miss)
		}
	})

	b.Run("AppendEncoderParallel", func(b *testing.B) {
		s := NewNGramSet(corpus, 3, DefaultStringEncoder)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				_ = s.Validate(hit)
				_ = s.Validate(miss)
			}
		})
	})

	b.Run("Fallback", func(b *testing.B) {
		s := NewNGramSet(corpus, 3, plainSepEncoder{sep: SEP})
		b.ResetTimer()
		for b.Loop() {
			_ = s.Validate(hit)
			_ = s.Validate(miss)
		}
	})
}
