package interned

import (
	"encoding/binary"
	"unsafe"

	barkov "github.com/soumitradev/barkov/v2"
)

const snapshotBlockRows = 1024

func (b *backoffCore) writeTables(e *snapshotEncoder) {
	for _, table := range b.tables {
		table.(interface{ writeTable(*snapshotEncoder) }).writeTable(e)
		if e.err != nil {
			return
		}
	}
	if b.contexts != nil {
		b.contexts.writeSnapshot(e)
	}
}

func (c *indexedCore[K]) writeTable(e *snapshotEncoder) {
	table := compactSnapshotTable(c.Model)
	e.u32(uint32(table.Len()))
	e.u32(uint32(len(c.Choices)))
	e.u32(uint32(len(table.entries)))
	e.u64(table.seed)
	width := (c.stateSize + 4) * 4
	block := make([]byte, 0, snapshotBlockRows*width)
	for i, token := range c.Choices {
		block = binary.LittleEndian.AppendUint32(block, uint32(token))
		block = binary.LittleEndian.AppendUint32(block, c.CumDist[i])
		if len(block) >= snapshotBlockRows*8 {
			e.bytes(block)
			block = block[:0]
		}
		if e.err != nil {
			return
		}
	}
	e.bytes(block)
	block = block[:0]
	for slot, entry := range table.entries {
		if entry.val == (barkov.ChoicesIndex{}) {
			continue
		}
		key, idx := entry.key, entry.val
		block = binary.LittleEndian.AppendUint32(block, uint32(slot))
		for _, token := range unsafe.Slice((*TokenID)(unsafe.Pointer(&key)), c.stateSize) {
			block = binary.LittleEndian.AppendUint32(block, uint32(token))
		}
		block = binary.LittleEndian.AppendUint32(block, idx.Offset)
		block = binary.LittleEndian.AppendUint32(block, idx.Count)
		block = binary.LittleEndian.AppendUint32(block, idx.Obs)
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

func readBackoff(d *snapshotDecoder, cfg BackoffConfig, vocab uint32) BackoffChain {
	b := &backoffCore{cfg: cfg, sentinels: DefaultSentinels(), tables: make([]orderTable, cfg.MaxOrder)}
	for order := 1; order <= cfg.MaxOrder && d.err == nil; order++ {
		b.tables[order-1] = readOrderTable(d, order, vocab)
	}
	if d.err != nil {
		return nil
	}
	if cfg.MinOrder > 1 {
		b.contexts = readContextSnapshot(d, b.tables[cfg.MinOrder-1], cfg)
	}
	return wrapBackoff(b)
}

func readOrderTable(d *snapshotDecoder, order int, vocab uint32) orderTable {
	switch order {
	case 1:
		return readTable[[1]TokenID](d, vocab)
	case 2:
		return readTable[[2]TokenID](d, vocab)
	case 3:
		return readTable[[3]TokenID](d, vocab)
	case 4:
		return readTable[[4]TokenID](d, vocab)
	case 5:
		return readTable[[5]TokenID](d, vocab)
	case 6:
		return readTable[[6]TokenID](d, vocab)
	case 7:
		return readTable[[7]TokenID](d, vocab)
	case 8:
		return readTable[[8]TokenID](d, vocab)
	default:
		panic("validated order")
	}
}

func readTable[K comparable](d *snapshotDecoder, vocab uint32) *indexedCore[K] {
	var zero K
	stateSize := int(unsafe.Sizeof(zero)) / 4
	width := (stateSize + 4) * 4
	n, choices, capacity := d.u32(), d.u32(), d.u32()
	seed := d.u64()
	if n == 0 || n > 1<<29 || capacity < 16 || capacity > 1<<30 || capacity&(capacity-1) != 0 || uint64(capacity) > max(16, uint64(n)*4) || n > capacity*loadFactorNum/loadFactorDen {
		d.fail("unsupported state table geometry")
	}
	if !d.checkSize(uint64(n)*uint64(width) + uint64(choices)*8) {
		return nil
	}
	choiceCapacity := int(choices)
	var table *stateMap[K, barkov.ChoicesIndex]
	if d.remaining < 0 {
		table = newStateMap[K, barkov.ChoicesIndex](min(int(n), 4096))
		choiceCapacity = min(choiceCapacity, 4096)
	} else {
		table = &stateMap[K, barkov.ChoicesIndex]{entries: make([]stateEntry[K, barkov.ChoicesIndex], capacity), mask: capacity - 1, growAt: capacity * loadFactorNum / loadFactorDen, seed: seed}
	}
	c := &indexedCore[K]{Model: table, stateSize: stateSize, sentinels: DefaultSentinels(), Choices: make([]TokenID, 0, choiceCapacity), CumDist: make([]uint32, 0, choiceCapacity)}
	block := make([]byte, snapshotBlockRows*width)
	for remaining := choices; remaining > 0 && d.err == nil; {
		rows := min(remaining, snapshotBlockRows)
		d.bytes(block[:rows*8])
		if d.err != nil {
			break
		}
		for i := uint32(0); i < rows; i++ {
			token, count := binary.LittleEndian.Uint32(block[i*8:]), binary.LittleEndian.Uint32(block[i*8+4:])
			if token >= vocab || count == 0 {
				d.fail("invalid transition")
				break
			}
			c.Choices = append(c.Choices, TokenID(token))
			c.CumDist = append(c.CumDist, count)
		}
		remaining -= rows
	}
	var previousSlot uint32
	haveSlot := false
	for remaining := n; remaining > 0 && d.err == nil; {
		rows := min(int(remaining), snapshotBlockRows)
		d.bytes(block[:rows*width])
		if d.err != nil {
			break
		}
		for i := range rows {
			record := block[i*width : (i+1)*width]
			slot := binary.LittleEndian.Uint32(record)
			if slot >= capacity || (haveSlot && slot <= previousSlot) {
				d.fail("invalid or repeated table slot")
				break
			}
			previousSlot, haveSlot = slot, true
			record = record[4:]
			var key K
			tokens := unsafe.Slice((*TokenID)(unsafe.Pointer(&key)), stateSize)
			for j := range stateSize {
				tokens[j] = TokenID(binary.LittleEndian.Uint32(record[j*4:]))
				if uint32(tokens[j]) >= vocab {
					d.fail("state token outside vocabulary")
				}
			}
			fields := record[stateSize*4:]
			idx := barkov.ChoicesIndex{Offset: binary.LittleEndian.Uint32(fields), Count: binary.LittleEndian.Uint32(fields[4:]), Obs: binary.LittleEndian.Uint32(fields[8:])}
			validateIndex(d, idx, vocab, c.CumDist)
			if d.err != nil {
				break
			}
			if d.remaining < 0 {
				if _, exists := c.Model.GetOrSet(key, idx); exists {
					d.fail("duplicate state")
					break
				}
			} else {
				c.Model.entries[slot] = stateEntry[K, barkov.ChoicesIndex]{key: key, val: idx}
				c.Model.count++
			}
		}
		remaining -= uint32(rows)
	}
	if d.err == nil && d.remaining >= 0 {
		validateSnapshotSlots(d, c.Model)
	}
	if d.err == nil {
		if _, exists := c.Model.Get(zero); !exists {
			d.fail("missing sentence-start state")
		}
	}
	return c
}

func validateIndex(d *snapshotDecoder, idx barkov.ChoicesIndex, vocab uint32, weights []uint32) {
	if idx.Count == 0 || idx.Obs < idx.Count {
		d.fail("invalid transition counts")
	} else if idx.Count == 1 {
		if idx.Offset >= vocab {
			d.fail("follower outside vocabulary")
		}
	} else if uint64(idx.Offset)+uint64(idx.Count) > uint64(len(weights)) {
		d.fail("transition span outside table")
	} else {
		var prev uint32
		for _, count := range weights[idx.Offset : idx.Offset+idx.Count] {
			if count <= prev {
				d.fail("non-increasing transition weights")
				break
			}
			prev = count
		}
		if prev != idx.Obs {
			d.fail("observation count disagrees with weights")
		}
	}
}

// Persist slot placement and the hash seed to avoid millions of random writes
// while loading. Only unusually sparse training tables need compaction on save.
// This format is coupled to stateMap.hash; change its version if the mixer changes.
func compactSnapshotTable[K comparable](table *stateMap[K, barkov.ChoicesIndex]) *stateMap[K, barkov.ChoicesIndex] {
	if len(table.entries) <= max(16, table.Len()*4) {
		return table
	}
	compact := newStateMap[K, barkov.ChoicesIndex](table.Len())
	for key, idx := range table.All() {
		compact.putNoGrow(key, idx)
	}
	return compact
}

// Every key must be reachable at its recorded slot before an empty or duplicate
// key. Checking in slot order keeps these probes close in memory, unlike rehashing.
func validateSnapshotSlots[K comparable](d *snapshotDecoder, table *stateMap[K, barkov.ChoicesIndex]) {
	for slot, entry := range table.entries {
		if entry.val == (barkov.ChoicesIndex{}) {
			continue
		}
		for probe := uint32(table.hash(entry.key)) & table.mask; probe != uint32(slot); probe = (probe + 1) & table.mask {
			prior := table.entries[probe]
			if prior.val == (barkov.ChoicesIndex{}) || prior.key == entry.key {
				d.fail("unreachable or duplicate table state")
				return
			}
		}
	}
}
