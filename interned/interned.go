package interned

import barkov "github.com/soumitradev/barkov/v2"

// TokenID is the integer identifier for an interned token.
// uint32 supports vocabularies up to 4 billion tokens.
type TokenID uint32

// Reserved sentinel IDs.
const (
	BeginTokenID TokenID = 0
	EndTokenID   TokenID = 1
	FirstUserID  TokenID = 2
)

// DefaultSentinels returns the standard TokenID sentinels.
func DefaultSentinels() barkov.Sentinels[TokenID] {
	return barkov.Sentinels[TokenID]{Begin: BeginTokenID, End: EndTokenID}
}
