// Tier 3: custom token type. The corpus is int64 (could be anything
// comparable — user IDs, event codes, enum values). The user provides
// their own encoder; barkov never assumes strings.
//
// The encoder implements both StateEncoder and AppendEncoder; the
// AppendEncoded path is optional but unlocks arena interning and
// zero-alloc validators when present.
package main

import (
	"context"
	"encoding/binary"
	"fmt"

	barkov "github.com/soumitradev/barkov/v2"
)

type int64Encoder struct{}

func (int64Encoder) Encode(tokens []int64) string {
	buf := make([]byte, 8*len(tokens))
	for i, t := range tokens {
		binary.LittleEndian.PutUint64(buf[i*8:], uint64(t))
	}
	return string(buf)
}

func (int64Encoder) AppendEncoded(dst []byte, tokens []int64) []byte {
	var buf [8]byte
	for _, t := range tokens {
		binary.LittleEndian.PutUint64(buf[:], uint64(t))
		dst = append(dst, buf[:]...)
	}
	return dst
}

func (int64Encoder) Decode(state string) []int64 {
	out := make([]int64, len(state)/8)
	for i := range out {
		out[i] = int64(binary.LittleEndian.Uint64([]byte(state[i*8 : i*8+8])))
	}
	return out
}

func main() {
	corpus := [][]int64{
		{10, 20, 30, 40, 50, 60},
		{10, 20, 30, 41, 51, 61},
		{10, 20, 31, 40, 50, 60},
		{11, 21, 30, 40, 52, 62},
		{10, 21, 30, 40, 50, 61},
		{11, 20, 31, 41, 51, 62},
	}

	chain := barkov.NewChain(barkov.ChainConfig[int64]{
		StateSize: 2,
		Sentinels: barkov.Sentinels[int64]{Begin: -1, End: -2},
		Encoder:   int64Encoder{},
	})
	compressed := chain.BuildCompressed(corpus)

	fmt.Println("Tier 3 — int64 chain with custom encoder:")
	for i := range 5 {
		out, err := barkov.Gen(context.Background(), compressed)
		if err != nil {
			fmt.Printf("%d) [error] %v\n", i+1, err)
			continue
		}
		fmt.Printf("%d) %v\n", i+1, out)
	}
}
