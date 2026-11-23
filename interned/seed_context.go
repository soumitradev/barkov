package interned

import (
	"fmt"
	"sort"

	barkov "github.com/soumitradev/barkov/v2"
)

// contextIndex holds the MinOrder states sorted by their tokens read from
// last to first, so every suffix is one contiguous range
type contextIndex struct {
	order      int
	rows       []TokenID // each row is order tokens then its observation count
	cumulative []uint64  // observations of supported rows, for weighted draws
}

func newContextIndex(table orderTable, order, minSupport int) *contextIndex {
	x := &contextIndex{order: order}
	for tokens, count := range table.states() {
		x.rows = append(x.rows, tokens...)
		x.rows = append(x.rows, TokenID(count))
	}
	sort.Sort(x)
	x.cumulative = make([]uint64, x.Len()+1)
	for i := 0; i < x.Len(); i++ {
		count := uint64(x.rows[i*(order+1)+order])
		x.cumulative[i+1] = x.cumulative[i]
		if count >= uint64(minSupport) {
			x.cumulative[i+1] += count
		}
	}
	return x
}

func (x *contextIndex) Len() int { return len(x.rows) / (x.order + 1) }
func (x *contextIndex) Less(i, j int) bool {
	a, b := x.row(i), x.row(j)
	for k := x.order - 1; k >= 0; k-- {
		if a[k] != b[k] {
			return a[k] < b[k]
		}
	}
	return false
}
func (x *contextIndex) Swap(i, j int) {
	width := x.order + 1
	for k := 0; k < width; k++ {
		x.rows[i*width+k], x.rows[j*width+k] = x.rows[j*width+k], x.rows[i*width+k]
	}
}
func (x *contextIndex) row(i int) []TokenID { return x.rows[i*(x.order+1) : i*(x.order+1)+x.order] }
func (x *contextIndex) compare(i int, suffix []TokenID) int {
	row := x.row(i)
	for j := len(suffix) - 1; j >= 0; j-- {
		token := row[x.order-len(suffix)+j]
		if token < suffix[j] {
			return -1
		}
		if token > suffix[j] {
			return 1
		}
	}
	return 0
}

// SeedContext samples observed tokens preceding a seed shorter than MinOrder,
// completing a MinOrder context. The history is not part of the emitted seed,
// and callers still pass both through the normal generation walk.
func (b *backoffCore) SeedContext(seed []TokenID) ([]TokenID, error) {
	if len(seed) == 0 || len(seed) >= b.cfg.MinOrder {
		return nil, nil
	}
	x := b.contexts
	lo := sort.Search(x.Len(), func(i int) bool { return x.compare(i, seed) >= 0 })
	hi := sort.Search(x.Len(), func(i int) bool { return x.compare(i, seed) > 0 })
	if lo == hi {
		return nil, fmt.Errorf("barkov: seed has no observed context: %w", barkov.ErrStateNotFound)
	}
	total := x.cumulative[hi] - x.cumulative[lo]
	if total == 0 {
		return nil, fmt.Errorf("barkov: seed contexts have insufficient support: %w", barkov.ErrInsufficientSupport)
	}
	target := x.cumulative[lo] + b.uint64n(total)
	i := lo + sort.Search(hi-lo, func(i int) bool { return x.cumulative[lo+i+1] > target })
	return append([]TokenID(nil), x.row(i)[:x.order-len(seed)]...), nil
}
