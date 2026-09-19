package text_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/soumitradev/barkov/v2/text"
)

func BenchmarkModelSnapshot(b *testing.B) {
	corpus := loadPublicCorpusBench()
	if len(corpus) == 0 {
		b.Fatal("public corpus is empty")
	}
	opts := []text.Option{text.Order(7), text.MinOrder(2), text.AntiVerbatim(6), text.Bidirectional()}
	g, err := text.New(corpus, opts...)
	if err != nil {
		b.Fatal(err)
	}
	f, err := os.Create(filepath.Join(b.TempDir(), "model.bin"))
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()
	if err := g.Save(f); err != nil {
		b.Fatal(err)
	}
	info, err := f.Stat()
	if err != nil {
		b.Fatal(err)
	}
	b.Run("build", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := text.New(corpus, opts...); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("save", func(b *testing.B) {
		b.SetBytes(info.Size())
		b.ReportAllocs()
		for b.Loop() {
			if _, err := f.Seek(0, 0); err != nil {
				b.Fatal(err)
			}
			if err := g.Save(f); err != nil {
				b.Fatal(err)
			}
			if err := f.Sync(); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("load", func(b *testing.B) {
		b.SetBytes(info.Size())
		b.ReportAllocs()
		for b.Loop() {
			if _, err := f.Seek(0, 0); err != nil {
				b.Fatal(err)
			}
			if _, err := text.Load(f); err != nil {
				b.Fatal(err)
			}
		}
	})
}
