package interned

import (
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
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(ids))), len(ids)*4)
	return string(b)
}

// AppendEncoded writes the packed form of ids into dst without any
// intermediate allocation. Satisfies barkov.AppendEncoder[TokenID].
func (PackedEncoder) AppendEncoded(dst []byte, ids []TokenID) []byte {
	if len(ids) == 0 {
		return dst
	}
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(ids))), len(ids)*4)
	return append(dst, b...)
}

// Decode unpacks a string key back into a slice of TokenIDs.
func (PackedEncoder) Decode(state string) []TokenID {
	n := len(state) / 4
	if n == 0 {
		return nil
	}
	ids := make([]TokenID, n)
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&ids[0])), len(state)), state)
	return ids
}
