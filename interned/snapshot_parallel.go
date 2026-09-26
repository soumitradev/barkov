package interned

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"runtime"
)

// Restoring a sized snapshot reads it in order, checksumming and bounding
// every section as the sequential reader does, and decodes each table (and
// each seed index, once its table is ready) on a worker from those bytes.
// Decoding and validation dominate restore time; they are independent per
// table. restoreWorkers bounds the undecoded sections held in memory.
var restoreWorkers = min(6, runtime.GOMAXPROCS(0))

type section struct {
	done chan struct{}
	err  error
}

// sectionReader holds at most cap(buffers) undecoded sections; each worker
// hands its buffer back for the next section, so restoring allocates only
// those few buffers rather than a copy of the whole file.
type sectionReader struct {
	d       *snapshotDecoder
	buffers chan []byte
	all     []*section
}

func newSectionReader(d *snapshotDecoder) *sectionReader {
	r := &sectionReader{d: d, buffers: make(chan []byte, restoreWorkers)}
	for range restoreWorkers {
		r.buffers <- nil
	}
	return r
}

// read takes a section of header bytes followed by the body length body
// computes from them, and decodes it on a worker once after (if non-nil)
// has finished. It returns nil when the section could not be read.
func (r *sectionReader) read(header int, body func([]byte) uint64, after *section, decode func(*snapshotDecoder)) *section {
	d := r.d
	head := make([]byte, header)
	d.bytes(head)
	size := body(head)
	if d.err != nil || !d.checkSize(size) {
		return nil
	}
	buf := <-r.buffers
	if need := uint64(header) + size; uint64(cap(buf)) < need {
		buf = make([]byte, need)
	} else {
		buf = buf[:need]
	}
	copy(buf, head)
	d.bytes(buf[header:])
	if d.err != nil {
		r.buffers <- buf
		return nil
	}
	s := &section{done: make(chan struct{})}
	r.all = append(r.all, s)
	go func() {
		defer close(s.done)
		defer func() { r.buffers <- buf }()
		if after != nil {
			<-after.done
			if after.err != nil {
				s.err = after.err
				return
			}
		}
		sub := &snapshotDecoder{r: bufio.NewReaderSize(bytes.NewReader(buf), 64<<10), remaining: int64(len(buf))}
		decode(sub)
		s.err = sub.err
	}()
	return s
}

// wait returns the first failure in file order, after every worker ends.
func (r *sectionReader) wait() error {
	var first error
	for _, s := range r.all {
		<-s.done
		if first == nil {
			first = s.err
		}
	}
	return first
}

// readBackoffSections queues one chain's tables and seed index; the chain
// is complete once r.wait returns nil.
func (r *sectionReader) readBackoffSections(cfg BackoffConfig, vocab uint32) *backoffCore {
	b := &backoffCore{cfg: cfg, sentinels: DefaultSentinels(), tables: make([]orderTable, cfg.MaxOrder)}
	sections := make([]*section, cfg.MaxOrder)
	for order := 1; order <= cfg.MaxOrder && r.d.err == nil; order++ {
		width := uint64(order+4) * 4
		sections[order-1] = r.read(20, func(h []byte) uint64 {
			n, choices := binary.LittleEndian.Uint32(h), binary.LittleEndian.Uint32(h[4:])
			return uint64(n)*width + uint64(choices)*8
		}, nil, func(d *snapshotDecoder) {
			b.tables[order-1] = readOrderTable(d, order, vocab)
		})
	}
	if r.d.err == nil && cfg.MinOrder > 1 {
		width := uint64(cfg.MinOrder+1) * 4
		r.read(4, func(h []byte) uint64 {
			return uint64(binary.LittleEndian.Uint32(h)) * width
		}, sections[cfg.MinOrder-1], func(d *snapshotDecoder) {
			b.contexts = readContextSnapshot(d, b.tables[cfg.MinOrder-1], cfg)
		})
	}
	return b
}
