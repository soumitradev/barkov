package interned

import (
	"encoding/binary"
	"fmt"

	"github.com/soumitradev/barkov/v2"
)

type TokenID uint32

// Reserved IDs
const BEGIN_TOKEN TokenID = 0
const END_TOKEN TokenID = 1

// Vocabulary for interned tokens, tokens acts like a map from ID to str
type Vocabulary struct {
	ids    map[string]TokenID
	tokens []string
}

func NewVocabulary() *Vocabulary {
	return &Vocabulary{
		ids:    make(map[string]TokenID),
		tokens: []string{barkov.BEGIN, barkov.END},
	}
}

func (v *Vocabulary) Intern(word string) TokenID {
	if res, ok := v.ids[word]; !ok {
		new := TokenID(len(v.tokens))
		v.ids[word] = new
		v.tokens = append(v.tokens, word)
		return new
	} else {
		return res
	}
}

func (v *Vocabulary) ID(word string) (TokenID, bool) {
	id, ok := v.ids[word]
	return id, ok
}

func (v *Vocabulary) Word(id TokenID) (string, bool) {
	if int(id) >= len(v.tokens) {
		return "", false
	}
	return v.tokens[id], true
}

func (v *Vocabulary) InternCorpus(corpus [][]string) [][]TokenID {
	res := make([][]TokenID, len(corpus))
	for i, run := range corpus {
		line := make([]TokenID, len(run))
		for j, word := range run {
			line[j] = v.Intern(word)
		}
		res[i] = line
	}
	return res
}

type Chain struct {
	*barkov.CompressedChain[TokenID]
}

func (c *Chain) Move(state []TokenID) (TokenID, error) {
	// clever optimization to hold upto order 16 on the stack, spills to heap if bigger
	var scratch [64]byte
	buf := scratch[:0]
	for _, id := range state {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(id))
	}
	// string(buf) is zero alloc, go apparently optimizes byte -> string
	idx, ok := c.Model[string(buf)]
	if !ok {
		return 0, fmt.Errorf("state %v not in model: %w", state, barkov.ErrStateNotFound)
	}

	// If there's only one choice, directly convert from TokenID
	if idx.Count == 1 {
		return TokenID(idx.Offset), nil
	}
	return c.Pick(idx), nil
}

func Build(stateSize int, corpus [][]TokenID) *Chain {
	cfg := barkov.ChainConfig[TokenID]{
		StateSize: stateSize,
		Sentinels: barkov.Sentinels[TokenID]{
			Begin: BEGIN_TOKEN, End: END_TOKEN,
		},
		Key: packedKey,
	}
	compressed := barkov.NewChain(cfg).Build(corpus).Compress()
	for key, idx := range compressed.Model {
		// Clever optimization thanks Claude
		if idx.Count == 1 {
			// If there's only one choice set the token ID directly
			idx.Offset = uint32(compressed.Choices[idx.Offset])
			compressed.Model[key] = idx
		}
	}
	return &Chain{compressed}
}

// Just a way to pack tokenIDs into string
func packedKey(state []TokenID) string {
	buf := make([]byte, 0, 4*len(state))
	for _, id := range state {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(id))
	}
	return string(buf)
}
