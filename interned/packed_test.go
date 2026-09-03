package interned

import (
	"slices"
	"testing"
)

func TestPackedEncoderRoundTrip(t *testing.T) {
	enc := PackedEncoder{}
	cases := [][]TokenID{
		{},
		{0},
		{1, 2, 3, 4},
		{0xffffffff, 0, 1},
	}
	for _, ids := range cases {
		decoded := enc.Decode(enc.Encode(ids))
		if len(ids) == 0 {
			if len(decoded) != 0 {
				t.Errorf("expected empty decode for empty input, got %v", decoded)
			}
			continue
		}
		if !slices.Equal(decoded, ids) {
			t.Errorf("round-trip failed: got %v, want %v", decoded, ids)
		}
	}
}

// TestPackedEncoderAppendEquivalence verifies the AppendEncoder fast path
// produces byte-identical output to Encode — the contract BuildRaw relies on.
func TestPackedEncoderAppendEquivalence(t *testing.T) {
	enc := PackedEncoder{}
	cases := [][]TokenID{
		nil,
		{},
		{0},
		{42},
		{1, 2, 3, 4},
		{0xffffffff, 0, 1, 0xdeadbeef},
	}
	for _, ids := range cases {
		want := enc.Encode(ids)
		got := string(enc.AppendEncoded(nil, ids))
		if got != want {
			t.Errorf("AppendEncoded(nil, %v) = %x; Encode = %x", ids, got, want)
		}
		prefix := []byte("PREFIX:")
		out := enc.AppendEncoded(prefix, ids)
		if string(out) != "PREFIX:"+want {
			t.Errorf("prefix preservation failed for %v", ids)
		}
	}
}

func BenchmarkPackedEncoderAppend(b *testing.B) {
	enc := PackedEncoder{}
	ids := make([]TokenID, 30)
	for i := range ids {
		ids[i] = TokenID(i + 10)
	}
	var buf [256]byte
	b.ResetTimer()
	for b.Loop() {
		_ = enc.AppendEncoded(buf[:0], ids)
	}
}

func BenchmarkPackedEncoderDecode(b *testing.B) {
	enc := PackedEncoder{}
	ids := make([]TokenID, 30)
	for i := range ids {
		ids[i] = TokenID(i + 10)
	}
	str := enc.Encode(ids)
	b.ResetTimer()
	for b.Loop() {
		_ = enc.Decode(str)
	}
}
