package interned

import (
	"runtime"
	"slices"
	"sync"
)

// Compact shrinks a freshly trained chain to the size restoring its snapshot
// would give: state tables are rehashed at the normal load factor and
// transition slices drop their training slack. Generation is unaffected.
// Compact must not run concurrently with any use of chain.
func Compact(chain BackoffChain) error {
	core, err := snapshotCore(chain)
	if err != nil {
		return err
	}
	jobs := make(chan orderTable)
	// A few tables at a time, so only a few training copies and their
	// compacted replacements are alive together
	var wg sync.WaitGroup
	for range min(4, runtime.GOMAXPROCS(0)) {
		wg.Go(func() {
			for table := range jobs {
				table.compact()
			}
		})
	}
	for _, table := range core.tables {
		jobs <- table
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
