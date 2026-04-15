package barkov

import (
	"strings"
)

// ConstructState joins tokens using the default separator.
// Kept for backwards compatibility.
func ConstructState(tokens []string) string {
	return strings.Join(tokens, SEP)
}

// DeconstructState splits a state string using the default separator.
// Kept for backwards compatibility.
func DeconstructState(state string) []string {
	if state == "" {
		return nil
	}
	return strings.Split(state, SEP)
}
