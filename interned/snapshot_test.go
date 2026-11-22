package interned

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"
	"testing/iotest"

	barkov "github.com/soumitradev/barkov/v2"
)

func modelFixture() ModelSnapshot {
	v := NewVocabulary()
	corpus := v.InternCorpus([][]string{{"one", "two", "three"}, {"one", "four", "five"}, {"one", "two", "three"}})
	return ModelSnapshot{Vocabulary: v, Forward: BuildBackoff(BackoffConfig{MaxOrder: 2, MaxVerbatim: -1}, corpus).(BackoffChain)}
}

func TestSnapshotRejectsInvalidTables(t *testing.T) {
	for _, mutate := range []func(*indexedCore[[1]TokenID]){
		func(c *indexedCore[[1]TokenID]) {
			c.Model.forEach(func(_ [1]TokenID, idx *barkov.ChoicesIndex) { idx.Count = 0 })
		},
		func(c *indexedCore[[1]TokenID]) {
			c.Model.forEach(func(_ [1]TokenID, idx *barkov.ChoicesIndex) { idx.Offset = 1<<32 - 1 })
		},
		func(c *indexedCore[[1]TokenID]) {
			c.Model.forEach(func(_ [1]TokenID, idx *barkov.ChoicesIndex) { idx.Obs = 0 })
		},
		func(c *indexedCore[[1]TokenID]) { c.CumDist[0] = 0 },
		func(c *indexedCore[[1]TokenID]) { c.CumDist[1] = c.CumDist[0] },
		func(c *indexedCore[[1]TokenID]) {
			c.Model.GetOrSet([1]TokenID{1<<32 - 1}, barkov.ChoicesIndex{Offset: 1, Count: 1, Obs: 1})
		},
	} {
		s := modelFixture()
		core, err := snapshotCore(s.Forward)
		if err != nil {
			t.Fatal(err)
		}
		mutate(core.tables[0].(*indexedCore[[1]TokenID]))
		var buf bytes.Buffer
		if _, err := s.WriteTo(&buf); err != nil {
			t.Fatal(err)
		}
		if _, err := ReadModelSnapshot(bytes.NewReader(buf.Bytes())); !errors.Is(err, ErrInvalidSnapshot) {
			t.Fatalf("invalid table with valid checksum accepted: %v", err)
		}
	}
}

func TestSnapshotStreamsAndCountsActualWrites(t *testing.T) {
	s := modelFixture()
	var buf bytes.Buffer
	n, err := s.WriteTo(&buf)
	if err != nil || n != int64(buf.Len()) {
		t.Fatalf("byte count = %d/%d %v", n, buf.Len(), err)
	}
	// No Len/Seek support and short reads exercise incremental decoding.
	got, err := ReadModelSnapshot(iotest.OneByteReader(bytes.NewReader(buf.Bytes())))
	if err != nil || got.Vocabulary.Size() != s.Vocabulary.Size() {
		t.Fatalf("stream restore failed: %v", err)
	}
	for _, limit := range []int{0, 10, buf.Len() - 1} {
		w := &limitedSnapshotWriter{left: limit}
		n, err := s.WriteTo(w)
		if !errors.Is(err, io.ErrClosedPipe) || n != int64(limit) {
			t.Fatalf("partial write = %d/%d %v", n, limit, err)
		}
	}
}

type limitedSnapshotWriter struct{ left int }

func (w *limitedSnapshotWriter) Write(p []byte) (int, error) {
	n := min(w.left, len(p))
	w.left -= n
	if n != len(p) {
		return n, io.ErrClosedPipe
	}
	return n, nil
}

func TestSnapshotTrailingDataAndReadErrors(t *testing.T) {
	s := modelFixture()
	var buf bytes.Buffer
	if _, err := s.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	valid := buf.Bytes()

	t.Run("trailing data reports clean error", func(t *testing.T) {
		_, err := ReadModelSnapshot(bytes.NewReader(append(bytes.Clone(valid), 0x42)))
		if !errors.Is(err, ErrInvalidSnapshot) {
			t.Fatalf("got %v, want ErrInvalidSnapshot", err)
		}
		if strings.Contains(err.Error(), "<nil>") || !strings.Contains(err.Error(), "trailing data") {
			t.Fatalf("unexpected trailing data error text: %q", err.Error())
		}
	})

	t.Run("trailing read error unwraps cause", func(t *testing.T) {
		r := io.MultiReader(bytes.NewReader(valid), iotest.ErrReader(io.ErrClosedPipe))
		_, err := ReadModelSnapshot(r)
		if !errors.Is(err, ErrInvalidSnapshot) || !errors.Is(err, io.ErrClosedPipe) {
			t.Fatalf("got %v, want ErrInvalidSnapshot wrapping io.ErrClosedPipe", err)
		}
	})

	t.Run("magic read error is not disguised as bad format", func(t *testing.T) {
		_, err := ReadModelSnapshot(iotest.ErrReader(io.ErrClosedPipe))
		if !errors.Is(err, io.ErrClosedPipe) || errors.Is(err, ErrInvalidSnapshot) {
			t.Fatalf("got %v, want io.ErrClosedPipe", err)
		}
		if _, err := ReadModelSnapshot(bytes.NewReader([]byte("BADMAGIC"))); !errors.Is(err, ErrInvalidSnapshot) {
			t.Fatalf("wrong magic got %v, want ErrInvalidSnapshot", err)
		}
	})
}

func TestSnapshotWriteToPreservesWriterError(t *testing.T) {
	s := modelFixture()
	for i := range 80 {
		s.Vocabulary.Intern(fmt.Sprintf("%04d_%s", i, strings.Repeat("a", 1024)))
	}
	s.Vocabulary.Intern(strings.Repeat("x", 1<<20+1))

	if _, err := s.WriteTo(&limitedSnapshotWriter{left: 1000}); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatalf("writer error swallowed: got %v, want io.ErrClosedPipe", err)
	}
	if _, err := s.WriteTo(io.Discard); !errors.Is(err, ErrInvalidSnapshot) {
		t.Fatalf("oversized token with healthy writer: got %v, want ErrInvalidSnapshot", err)
	}
}
