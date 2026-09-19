package interned

import "encoding/binary"

func (x *contextIndex) writeSnapshot(e *snapshotEncoder) {
	e.u32(uint32(x.Len()))
	block := make([]byte, 0, 64<<10)
	for _, word := range x.rows {
		block = binary.LittleEndian.AppendUint32(block, uint32(word))
		if len(block) == cap(block) {
			e.bytes(block)
			block = block[:0]
		}
		if e.err != nil {
			return
		}
	}
	e.bytes(block)
}

func (c *indexedCore[K]) snapshotStateCount() int { return c.Model.Len() }

// Restore the already sorted seed index. Checking membership and strict order
// verifies every state occurs once; cumulative weights need only a linear pass.
func readContextSnapshot(d *snapshotDecoder, table orderTable, cfg BackoffConfig) *contextIndex {
	n := d.u32()
	if int(n) != table.(interface{ snapshotStateCount() int }).snapshotStateCount() {
		d.fail("seed context count differs from state table")
	}
	width := cfg.MinOrder + 1
	if !d.checkSize(uint64(n) * uint64(width) * 4) {
		return nil
	}
	capacity := int(n)
	if d.remaining < 0 {
		capacity = min(capacity, 4096)
	}
	x := &contextIndex{order: cfg.MinOrder, rows: make([]TokenID, 0, capacity*width), cumulative: make([]uint64, 1, capacity+1)}
	block := make([]byte, snapshotBlockRows*width*4)
	for remaining := int(n); remaining > 0 && d.err == nil; {
		rows := min(remaining, snapshotBlockRows)
		d.bytes(block[:rows*width*4])
		if d.err != nil {
			break
		}
		for i := range rows * width {
			x.rows = append(x.rows, TokenID(binary.LittleEndian.Uint32(block[i*4:])))
		}
		remaining -= rows
	}
	for i := 0; i < x.Len() && d.err == nil; i++ {
		row := x.row(i)
		count := uint32(x.rows[i*width+cfg.MinOrder])
		idx, ok := table.lookupTail(row)
		if !ok || idx.Obs != count || (i > 0 && !x.Less(i-1, i)) {
			d.fail("invalid seed context row or order")
			break
		}
		total := x.cumulative[len(x.cumulative)-1]
		if count >= uint32(cfg.MinSupport) {
			total += uint64(count)
		}
		x.cumulative = append(x.cumulative, total)
	}
	return x
}
