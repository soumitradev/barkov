package interned

import barkov "github.com/soumitradev/barkov/v2"

// InitChain returns a new TokenID-based chain paired with its vocabulary.
// The returned chain's Model is empty — call BuildRaw with an interned corpus.
//
// Typical usage:
//
//	chain, vocab := interned.InitChain(4)
//	encoded := vocab.InternCorpus(stringCorpus)
//	chain.BuildRaw(encoded)
//	compressed := chain.Compress()
func InitChain(stateSize int) (*barkov.Chain[TokenID], *Vocabulary) {
	vocab := NewVocabulary()
	chain := barkov.NewChain(barkov.ChainConfig[TokenID]{
		StateSize: stateSize,
		Sentinels: DefaultSentinels(),
		Encoder:   PackedEncoder{},
	})
	return chain, vocab
}
