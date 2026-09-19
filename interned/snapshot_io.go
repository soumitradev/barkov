package interned

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"hash"
	"io"
)

type snapshotEncoder struct {
	w       io.Writer
	err     error
	scratch [8]byte
}

func (e *snapshotEncoder) bytes(p []byte) {
	if e.err != nil {
		return
	}
	n, err := e.w.Write(p)
	e.err = err
	if err == nil && n != len(p) {
		e.err = io.ErrShortWrite
	}
}
func (e *snapshotEncoder) u32(v uint32) {
	binary.LittleEndian.PutUint32(e.scratch[:4], v)
	e.bytes(e.scratch[:4])
}
func (e *snapshotEncoder) u64(v uint64) {
	binary.LittleEndian.PutUint64(e.scratch[:], v)
	e.bytes(e.scratch[:])
}
func (e *snapshotEncoder) boolean(v bool) {
	if v {
		e.u32(1)
	} else {
		e.u32(0)
	}
}

type snapshotDecoder struct {
	r         *bufio.Reader
	hash      hash.Hash32
	err       error
	scratch   [8]byte
	remaining int64 // -1 when the input does not expose its actual length
}

func (d *snapshotDecoder) bytes(p []byte) {
	if d.err != nil {
		return
	}
	if d.remaining >= 0 {
		if int64(len(p)) > d.remaining {
			d.err = io.ErrUnexpectedEOF
			return
		}
		d.remaining -= int64(len(p))
	}
	_, d.err = io.ReadFull(d.r, p)
	if d.err == nil && d.hash != nil {
		_, _ = d.hash.Write(p)
	}
}
func (d *snapshotDecoder) u32() uint32 {
	d.bytes(d.scratch[:4])
	return binary.LittleEndian.Uint32(d.scratch[:4])
}
func (d *snapshotDecoder) u64() uint64 {
	d.bytes(d.scratch[:])
	return binary.LittleEndian.Uint64(d.scratch[:])
}
func (d *snapshotDecoder) fail(reason string) {
	if d.err == nil {
		d.err = fmt.Errorf("%w: %s", ErrInvalidSnapshot, reason)
	}
}

// Size information permits one allocation per table without trusting a corrupt
// length prefix. Non-seekable streams grow incrementally as records arrive.
func snapshotRemaining(r io.Reader) (int64, error) {
	if sized, ok := r.(interface{ Len() int }); ok {
		return int64(sized.Len()), nil
	}
	seeker, ok := r.(io.Seeker)
	if !ok {
		return -1, nil
	}
	start, err := seeker.Seek(0, io.SeekCurrent)
	if err != nil {
		return -1, nil
	}
	end, seekErr := seeker.Seek(0, io.SeekEnd)
	if _, err := seeker.Seek(start, io.SeekStart); err != nil {
		return 0, err
	}
	if seekErr != nil {
		return -1, nil
	}
	if end < start {
		return 0, io.ErrUnexpectedEOF
	}
	return end - start, nil
}

func (d *snapshotDecoder) checkSize(n uint64) bool {
	if d.err == nil && d.remaining >= 0 && n > uint64(d.remaining) {
		d.err = io.ErrUnexpectedEOF
	}
	return d.err == nil
}

type snapshotCounter struct {
	w io.Writer
	n int64
}

func (w *snapshotCounter) Write(p []byte) (int, error) {
	n, err := w.w.Write(p)
	w.n += int64(n)
	return n, err
}
