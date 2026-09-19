package text

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math/rand/v2"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

var snapshotCorpus = [][]string{
	{"the", "quick", "brown", "fox", "jumps"},
	{"the", "quick", "red", "fox", "rests"},
	{"a", "slow", "brown", "dog", "rests"},
	{"a", "quick", "brown", "dog", "jumps"},
	{"the", "quick", "brown", "fox", "jumps"},
	{"a", "slow", "brown", "dog", "rests"},
	{"hello", "世界", "🙂", "friend"},
}

func saveFixture(t *testing.T, opts ...Option) (*Generator, []byte) {
	t.Helper()
	g, err := New(snapshotCorpus, opts...)
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := g.Save(&buf); err != nil {
		t.Fatal(err)
	}
	return g, buf.Bytes()
}

func TestSnapshotPreservesGeneration(t *testing.T) {
	for order := 2; order <= 8; order++ {
		for _, bidirectional := range []bool{false, true} {
			for _, copying := range []int{0, 3} {
				opts := []Option{Order(order), MinOrder(2), MinSupport(2), AntiVerbatim(copying), MaxTokens(20), Retries(8), Timeout(0)}
				if bidirectional {
					opts = append(opts, Bidirectional())
				}
				original, data := saveFixture(t, opts...)
				restored, err := Load(bytes.NewReader(data))
				if err != nil {
					t.Fatalf("order=%d reverse=%v copy=%d: %v", order, bidirectional, copying, err)
				}
				if original.defaults != restored.defaults || original.minOrder != restored.minOrder {
					t.Fatal("generation defaults changed")
				}
				if !restored.Contains("世界") || !restored.Contains("🙂") {
					t.Fatal("Unicode vocabulary changed")
				}
				original.SetRNG(rand.New(rand.NewPCG(1, 2)))
				restored.SetRNG(rand.New(rand.NewPCG(1, 2)))
				seeds := [][]GenOption{nil, {Seed([]string{"the", "quick"})}, {Seed([]string{"unknown"})}, {SeedMiddle([]string{"brown"})}}
				for range 10 {
					for _, seed := range seeds {
						seed = append(seed, WithVerbatimFallback())
						a, ae := original.Generate(context.Background(), seed...)
						b, be := restored.Generate(context.Background(), seed...)
						if !reflect.DeepEqual(a, b) || errorText(ae) != errorText(be) {
							t.Fatalf("round-trip changed generation: %v %v -> %v %v", a, ae, b, be)
						}
					}
				}
			}
		}
	}
}

func errorText(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func TestSnapshotRetainsCopyChecksAndFallback(t *testing.T) {
	g, err := New([][]string{{"one", "two", "three", "four"}}, Order(2), Bidirectional(), Retries(2), Timeout(time.Second))
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := g.Save(&buf); err != nil {
		t.Fatal(err)
	}
	g, err = Load(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.Generate(context.Background()); err == nil {
		t.Fatal("copy checks disappeared")
	}
	out, err := g.Generate(context.Background(), SeedMiddle([]string{"two"}), WithVerbatimFallback())
	if err != nil || !strings.Contains(strings.Join(out, " "), "two three") {
		t.Fatalf("fallback or middle seed lost: %v %v", out, err)
	}
	if !g.messages.Contains(g.vocab.InternCorpus([][]string{{"one", "two", "three", "four"}})[0]) {
		t.Fatal("whole-message lookup changed")
	}
}

func TestSnapshotRejectsDamage(t *testing.T) {
	_, data := saveFixture(t, Bidirectional())
	for _, length := range []int{0, 7, 35, 36, len(data) / 2, len(data) - 4, len(data) - 1} {
		if g, err := Load(bytes.NewReader(data[:length])); err == nil || g != nil {
			t.Fatalf("accepted truncated length %d", length)
		}
	}
	for _, offset := range []int{0, 6, 12, 35, 36, 42, len(data) / 2, len(data) - 1} {
		corrupt := bytes.Clone(data)
		corrupt[offset] ^= 0x80
		if g, err := Load(bytes.NewReader(corrupt)); err == nil || g != nil {
			t.Fatalf("accepted damage at %d", offset)
		}
	}
	if _, err := Load(bytes.NewReader(append(bytes.Clone(data), 1))); err == nil {
		t.Fatal("accepted trailing data")
	}
}

type failWriter struct{ remaining int }

func (w *failWriter) Write(p []byte) (int, error) {
	n := min(w.remaining, len(p))
	w.remaining -= n
	if n < len(p) {
		return n, io.ErrClosedPipe
	}
	return n, nil
}

type shortWriter struct{}

func (shortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

func TestSnapshotWriteFailures(t *testing.T) {
	g, _ := saveFixture(t, Bidirectional())
	for _, length := range []int{0, 10, 40, 100} {
		if err := g.Save(&failWriter{remaining: length}); !errors.Is(err, io.ErrClosedPipe) {
			t.Fatalf("write error lost at %d: %v", length, err)
		}
	}
	if err := g.Save(shortWriter{}); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short write = %v", err)
	}
}

func TestSnapshotSaveAlongsideGeneration(t *testing.T) {
	g, _ := saveFixture(t, Bidirectional(), AntiVerbatim(0))
	var wg sync.WaitGroup
	for range 4 {
		wg.Go(func() {
			for range 30 {
				if _, err := g.Generate(context.Background(), SeedMiddle([]string{"brown"})); err != nil {
					t.Error(err)
				}
			}
		})
	}
	for range 10 {
		var buf bytes.Buffer
		if err := g.Save(&buf); err != nil {
			t.Fatal(err)
		}
		restored, err := Load(&buf)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := restored.Generate(context.Background()); err != nil {
			t.Error(err)
		}
	}
	wg.Wait()
}

func FuzzLoadSnapshot(f *testing.F) {
	g, err := New(snapshotCorpus, Bidirectional())
	if err != nil {
		f.Fatal(err)
	}
	var buf bytes.Buffer
	if err := g.Save(&buf); err != nil {
		f.Fatal(err)
	}
	f.Add(buf.Bytes())
	f.Add([]byte("BRKTXT"))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 1<<20 {
			t.Skip()
		}
		g, err := Load(bytes.NewReader(data))
		if err == nil {
			_, _ = g.Generate(context.Background(), WithTimeout(time.Millisecond), WithRetries(1), WithMaxTokens(10))
		}
	})
}
