package interned

import (
	"bytes"
	"errors"
	"io"
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
