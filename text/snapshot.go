package text

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"time"

	"github.com/soumitradev/barkov/v2/interned"
)

// Save writes a complete trained generator. It preserves generation defaults,
// vocabulary, both directions and copy checks, but not the RNG's current state.
// Saving may run alongside generation if the model/vocabulary are not mutated.
// The caller owns flushing, closing and atomically replacing any destination file.
func (g *Generator) Save(w io.Writer) error {
	if g == nil || g.chain == nil {
		return fmt.Errorf("text: cannot save an empty generator")
	}
	for _, n := range []int{g.defaults.maxTokens, g.defaults.retries, g.defaults.threads} {
		if n < 0 || uint64(n) > 1<<32-1 {
			return fmt.Errorf("text: generation defaults exceed snapshot format limits")
		}
	}
	var header [36]byte
	copy(header[:8], "BRKTXT\x01\x00")
	binary.LittleEndian.PutUint64(header[8:16], uint64(g.defaults.timeout))
	binary.LittleEndian.PutUint32(header[16:20], uint32(g.defaults.maxTokens))
	binary.LittleEndian.PutUint32(header[20:24], uint32(g.defaults.retries))
	binary.LittleEndian.PutUint32(header[24:28], uint32(g.defaults.threads))
	binary.LittleEndian.PutUint32(header[28:32], uint32(g.minOrder))
	binary.LittleEndian.PutUint32(header[32:], crc32.ChecksumIEEE(header[:32]))
	if n, err := w.Write(header[:]); err != nil {
		return err
	} else if n != len(header) {
		return io.ErrShortWrite
	}
	_, err := (interned.ModelSnapshot{Vocabulary: g.vocab, Forward: g.chain, Reverse: g.reverseChain, Messages: g.messages}).WriteTo(w)
	return err
}

// Load restores a generator without its corpus or retraining. It verifies format,
// checksums and table invariants before returning. Supply SetRNG after loading for
// deterministic generation; the restored default RNG supports concurrent calls.
func Load(r io.Reader) (*Generator, error) {
	var header [36]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return nil, err
	}
	if string(header[:8]) != "BRKTXT\x01\x00" || binary.LittleEndian.Uint32(header[32:]) != crc32.ChecksumIEEE(header[:32]) {
		return nil, fmt.Errorf("%w: text header version or checksum", interned.ErrInvalidSnapshot)
	}
	defaults := generationConfig{
		timeout:   time.Duration(binary.LittleEndian.Uint64(header[8:16])),
		maxTokens: int(binary.LittleEndian.Uint32(header[16:20])),
		retries:   int(binary.LittleEndian.Uint32(header[20:24])),
		threads:   int(binary.LittleEndian.Uint32(header[24:28])),
	}
	if err := defaults.validate(false); err != nil {
		return nil, err
	}
	s, err := interned.ReadModelSnapshot(r)
	if err != nil {
		return nil, err
	}
	minOrder := int(binary.LittleEndian.Uint32(header[28:32]))
	if minOrder != s.Config().MinOrder {
		return nil, fmt.Errorf("%w: text and chain minimum orders disagree", interned.ErrInvalidSnapshot)
	}
	return &Generator{chain: s.Forward, reverseChain: s.Reverse, vocab: s.Vocabulary, messages: s.Messages, defaults: defaults, minOrder: minOrder, unknownContext: interned.TokenID(s.Vocabulary.Size())}, nil
}

// Compact shrinks a freshly trained generator's tables to what loading its
// snapshot builds, reducing memory without changing generation. Call it
// before sharing the generator; it must not run concurrently with other use.
func (g *Generator) Compact() error {
	for _, chain := range []interned.BackoffChain{g.chain, g.reverseChain} {
		if chain == nil {
			continue
		}
		if err := interned.Compact(chain); err != nil {
			return err
		}
	}
	return nil
}
