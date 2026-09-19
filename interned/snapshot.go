package interned

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"slices"

	barkov "github.com/soumitradev/barkov/v2"
)

// ModelSnapshot groups the trained data shared by a text generator. Saving reads
// immutable tables; RNG state is not saved. Do not mutate the vocabulary while saving.
type ModelSnapshot struct {
	Vocabulary       *Vocabulary
	Forward, Reverse BackoffChain
	Messages         *MessageSet
}

var ErrInvalidSnapshot = errors.New("barkov: invalid model snapshot")

var snapshotCRC = crc32.MakeTable(crc32.Castagnoli)

// WriteTo streams versioned, checksummed model tables without copying the corpus.
// The format is independent of Go's struct layout; message hashes require the
// same native byte order on save and load. Unknown versions are rejected.
func (s ModelSnapshot) WriteTo(w io.Writer) (int64, error) {
	forward, err := snapshotCore(s.Forward)
	if err != nil || s.Vocabulary == nil {
		return 0, fmt.Errorf("%w: missing vocabulary or forward chain", ErrInvalidSnapshot)
	}
	var reverse *backoffCore
	if s.Reverse != nil {
		reverse, err = snapshotCore(s.Reverse)
		if err != nil || reverse.cfg != forward.cfg {
			return 0, fmt.Errorf("%w: incompatible reverse chain", ErrInvalidSnapshot)
		}
	}
	if (s.Messages != nil) != (forward.cfg.MaxVerbatim > 0) {
		return 0, fmt.Errorf("%w: copy policy and message set disagree", ErrInvalidSnapshot)
	}
	h := crc32.New(snapshotCRC)
	counted := &snapshotCounter{w: w}
	b := bufio.NewWriterSize(io.MultiWriter(counted, h), 64<<10)
	e := snapshotEncoder{w: b}
	e.bytes(snapshotMagic())
	e.u32(uint32(s.Vocabulary.Size()))
	for _, token := range s.Vocabulary.idToToken {
		if uint64(len(token)) > 1<<20 {
			return counted.n, fmt.Errorf("%w: token exceeds 1 MiB", ErrInvalidSnapshot)
		}
		e.u32(uint32(len(token)))
		e.bytes([]byte(token))
	}
	e.u32(uint32(forward.cfg.MaxOrder))
	e.u32(uint32(forward.cfg.MinOrder))
	e.u32(uint32(forward.cfg.MinSupport))
	e.u32(uint32(int32(forward.cfg.MaxVerbatim)))
	e.boolean(reverse != nil)
	forward.writeTables(&e)
	if reverse != nil {
		reverse.writeTables(&e)
	}
	e.u32(uint32(s.Messages.Size()))
	if s.Messages != nil {
		for digest := range s.Messages.digests {
			e.u64(digest)
		}
	}
	if e.err == nil {
		e.err = b.Flush()
	}
	checksum := h.Sum32()
	e.u32(checksum)
	if e.err == nil {
		e.err = b.Flush()
	}
	return counted.n, e.err
}

// ReadModelSnapshot restores trained tables directly. It rejects truncated,
// corrupt and structurally invalid input before returning any usable model.
// Input sizes are consumed incrementally rather than trusted for large allocations.
func ReadModelSnapshot(r io.Reader) (ModelSnapshot, error) {
	h := crc32.New(snapshotCRC)
	// Hash decoded bytes, not buffered read-ahead (the final checksum is excluded).
	remaining, err := snapshotRemaining(r)
	if err != nil {
		return ModelSnapshot{}, err
	}
	d := snapshotDecoder{r: bufio.NewReaderSize(r, 64<<10), hash: h, remaining: remaining}
	var s ModelSnapshot
	magic := make([]byte, 8)
	d.bytes(magic)
	if !slices.Equal(magic, snapshotMagic()) {
		return s, fmt.Errorf("%w: unsupported format version or byte order", ErrInvalidSnapshot)
	}
	n := d.u32()
	if n < 2 || n == 1<<32-1 {
		d.fail("vocabulary needs both sentinels")
	}
	if !d.checkSize(uint64(n) * 4) {
		return s, d.err
	}
	v := &Vocabulary{tokenToID: make(map[string]TokenID)}
	for i := uint32(0); i < n && d.err == nil; i++ {
		size := d.u32()
		if size > 1<<20 {
			d.fail("token exceeds 1 MiB")
			break
		}
		if !d.checkSize(uint64(size)) {
			break
		}
		buf := make([]byte, size)
		d.bytes(buf)
		token := string(buf)
		if _, exists := v.tokenToID[token]; exists {
			d.fail("duplicate vocabulary token")
			break
		}
		v.idToToken = append(v.idToToken, token)
		v.tokenToID[token] = TokenID(i)
	}
	if d.err == nil && (v.Token(BeginTokenID) != barkov.BEGIN || v.Token(EndTokenID) != barkov.END) {
		d.fail("invalid vocabulary sentinels")
	}
	cfg := BackoffConfig{MaxOrder: int(d.u32()), MinOrder: int(d.u32()), MinSupport: int(d.u32()), MaxVerbatim: int(int32(d.u32()))}
	if cfg.MaxOrder < 2 || cfg.MaxOrder > 8 || cfg.MinOrder < 1 || cfg.MinOrder > cfg.MaxOrder || cfg.MinSupport < 1 || cfg.MinSupport > 65535 || (cfg.MaxVerbatim != -1 && (cfg.MaxVerbatim < 2 || cfg.MaxVerbatim > cfg.MaxOrder)) {
		d.fail("invalid backoff configuration")
	}
	reverse := d.u32()
	if reverse > 1 {
		d.fail("invalid reverse-chain flag")
	}
	if d.err != nil {
		return s, d.err
	}
	s.Vocabulary = v
	s.Forward = readBackoff(&d, cfg, n)
	if reverse != 0 {
		s.Reverse = readBackoff(&d, cfg, n)
	}
	count := d.u32()
	if cfg.MaxVerbatim < 0 && count != 0 {
		d.fail("message digests with copy prevention disabled")
	}
	if cfg.MaxVerbatim > 0 {
		s.Messages = NewMessageSet(nil)
		for i := uint32(0); i < count && d.err == nil; i++ {
			digest := d.u64()
			if _, exists := s.Messages.digests[digest]; exists {
				d.fail("duplicate message digest")
			}
			s.Messages.digests[digest] = struct{}{}
		}
		if count == 0 {
			d.fail("missing whole-message digests")
		}
	}
	checksum := h.Sum32()
	d.hash = nil
	if want := d.u32(); d.err == nil && checksum != want {
		d.fail("checksum mismatch")
	}
	if d.err != nil {
		return ModelSnapshot{}, d.err
	}
	// This is a complete snapshot frame, not a prefix of another stream.
	if _, err := d.r.ReadByte(); err != io.EOF {
		return ModelSnapshot{}, fmt.Errorf("%w: trailing data or read error: %v", ErrInvalidSnapshot, err)
	}
	return s, nil
}

func snapshotMagic() []byte {
	order := byte(1)
	if binary.NativeEndian.Uint32([]byte{1, 0, 0, 0}) != 1 {
		order = 2
	}
	return []byte{'B', 'R', 'K', 'M', 'O', 'D', 1, order}
}

func (b *backoffCore) snapshotCore() *backoffCore { return b }

// Config returns the forward chain's training configuration, or zero for an invalid snapshot.
func (s ModelSnapshot) Config() BackoffConfig {
	c, err := snapshotCore(s.Forward)
	if err != nil {
		return BackoffConfig{}
	}
	return c.cfg
}

func snapshotCore(chain BackoffChain) (*backoffCore, error) {
	if c, ok := chain.(interface{ snapshotCore() *backoffCore }); ok {
		return c.snapshotCore(), nil
	}
	return nil, fmt.Errorf("%w: unsupported chain implementation", ErrInvalidSnapshot)
}
