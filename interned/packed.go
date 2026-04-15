package interned

import "encoding/binary"

// PackedEncoder encodes sequences of TokenIDs as little-endian uint32 bytes,
// 4 bytes per token. Implements barkov.StateEncoder[TokenID].
type PackedEncoder struct{}

// Encode packs a slice of TokenIDs into a string key for map lookups.
func (PackedEncoder) Encode(ids []TokenID) string {
	buf := make([]byte, len(ids)*4)
	for i, id := range ids {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(id))
	}
	return string(buf)
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
