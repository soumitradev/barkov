package interned

import (
	"encoding/binary"
	"unsafe"
)

// PackedEncoder encodes sequences of TokenIDs as little-endian uint32 bytes,
// 4 bytes per token. Implements barkov.StateEncoder[TokenID].
type PackedEncoder struct{}

// Encode packs a slice of TokenIDs into a string key for map lookups.
func (PackedEncoder) Encode(ids []TokenID) string {
	if len(ids) == 0 {
		return ""
	}
	buf := make([]byte, len(ids)*4)
	for i, id := range ids {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(id))
	}
	// unsafe.String shares backing bytes with buf, skipping the copy that
	// string(buf) performs. Safe because buf is freshly allocated, not
	// aliased, and not mutated after this point.
	return unsafe.String(&buf[0], len(buf))
}

// AppendEncoded writes the packed form of ids into dst without any
// intermediate allocation. Satisfies barkov.AppendEncoder[TokenID].
func (PackedEncoder) AppendEncoded(dst []byte, ids []TokenID) []byte {
	for _, id := range ids {
		dst = binary.LittleEndian.AppendUint32(dst, uint32(id))
	}
	return dst
}

// Decode unpacks a string key back into a slice of TokenIDs.
func (PackedEncoder) Decode(state string) []TokenID {
	n := len(state) / 4
	ids := make([]TokenID, n)
	for i := range n {
		ids[i] = TokenID(binary.LittleEndian.Uint32([]byte(state)[i*4:]))
	}
	return ids
}
