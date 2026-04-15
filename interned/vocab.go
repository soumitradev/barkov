package interned

import barkov "github.com/soumitradev/barkov/v2"

// Vocabulary maps strings to TokenIDs and back. Once a token is interned,
// its ID is stable for the lifetime of the Vocabulary.
type Vocabulary struct {
	tokenToID map[string]TokenID
	idToToken []string
}

// NewVocabulary creates a vocabulary with the default string sentinels
// (barkov.BEGIN and barkov.END) pre-registered at IDs 0 and 1.
func NewVocabulary() *Vocabulary {
	v := &Vocabulary{
		tokenToID: make(map[string]TokenID, 1024),
		idToToken: make([]string, 0, 1024),
	}
	v.idToToken = append(v.idToToken, barkov.BEGIN, barkov.END)
	v.tokenToID[barkov.BEGIN] = BeginTokenID
	v.tokenToID[barkov.END] = EndTokenID
	return v
}

// Intern returns the ID for a token, creating one if it doesn't exist.
// Not safe for concurrent use.
func (v *Vocabulary) Intern(token string) TokenID {
	if id, ok := v.tokenToID[token]; ok {
		return id
	}
	id := TokenID(len(v.idToToken))
	v.idToToken = append(v.idToToken, token)
	v.tokenToID[token] = id
	return id
}

// Lookup returns the ID for an existing token, or (0, false) if not present.
func (v *Vocabulary) Lookup(token string) (TokenID, bool) {
	id, ok := v.tokenToID[token]
	return id, ok
}

// Token returns the string for an ID, or "" if the ID is invalid.
func (v *Vocabulary) Token(id TokenID) string {
	if int(id) >= len(v.idToToken) {
		return ""
	}
	return v.idToToken[id]
}

// Size returns the number of tokens in the vocabulary (including sentinels).
func (v *Vocabulary) Size() int {
	return len(v.idToToken)
}

// InternCorpus interns all tokens in a string corpus and returns the ID-encoded version.
func (v *Vocabulary) InternCorpus(corpus [][]string) [][]TokenID {
	encoded := make([][]TokenID, len(corpus))
	for i, sentence := range corpus {
		ids := make([]TokenID, len(sentence))
		for j, token := range sentence {
			ids[j] = v.Intern(token)
		}
		encoded[i] = ids
	}
	return encoded
}

// DecodeTokens converts a slice of TokenIDs back to strings.
func (v *Vocabulary) DecodeTokens(ids []TokenID) []string {
	tokens := make([]string, len(ids))
	for i, id := range ids {
		tokens[i] = v.Token(id)
	}
	return tokens
}
