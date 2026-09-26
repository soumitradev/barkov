package interned

import (
	"runtime"
	"slices"
	"sync"
)

// Compact shrinks a freshly trained chain to what restoring its snapshot
// builds: sparse state tables are rehashed at the normal load factor, as a
// snapshot write would, and transition slices lose the spare capacity
// training reserves for every observation. Tables are compacted a few at a
// time, each replacing its training copy as soon as it is ready. Generation
// is unaffected. Compact must not run concurrently with any use of chain.
func Compact(chain BackoffChain) error {
	core, err := snapshotCore(chain)
	if err != nil {
		return err
	}
	jobs := make(chan interface{ compact() })
	var wg sync.WaitGroup
	for range min(4, runtime.GOMAXPROCS(0)) {
		wg.Go(func() {
			for table := range jobs {
				table.compact()
			}
		})
	}
	for _, table := range core.tables {
		jobs <- table.(interface{ compact() })
	}
	close(jobs)
	wg.Wait()
	return nil
}

func (c *indexedCore[K]) compact() {
	c.Model = compactSnapshotTable(c.Model)
	c.Choices = slices.Clip(slices.Clone(c.Choices))
	c.CumDist = slices.Clip(slices.Clone(c.CumDist))
}
