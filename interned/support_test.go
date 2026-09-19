package interned

import (
	"errors"
	"testing"

	barkov "github.com/soumitradev/barkov/v2"
)

func TestSupportThresholdDoesNotChangeContextFloor(t *testing.T) {
	v := NewVocabulary()
	encoded := v.InternCorpus([][]string{{"a", "b", "c", "d"}})
	window := []TokenID{mustLookup(t, v, "a"), mustLookup(t, v, "b"), mustLookup(t, v, "c")}
	for _, floor := range []int{1, 2, 3} {
		observed := BuildBackoff(BackoffConfig{MaxOrder: 3, MinOrder: floor, MaxVerbatim: -1}, encoded)
		if token, err := observed.Move(observed.Encoder().Encode(window)); err != nil || token != mustLookup(t, v, "d") {
			t.Fatalf("rare observed context rejected by default: %v %v", token, err)
		}
		strict := BuildBackoff(BackoffConfig{MaxOrder: 3, MinOrder: floor, MinSupport: 3, MaxVerbatim: -1}, encoded)
		if _, err := strict.Move(strict.Encoder().Encode(window)); !errors.Is(err, barkov.ErrInsufficientSupport) || errors.Is(err, barkov.ErrStateNotFound) {
			t.Fatalf("floor %d weakened support or misreported absence: %v", floor, err)
		}
		absent := []TokenID{999999, 999999, 999999}
		if _, err := strict.Move(strict.Encoder().Encode(absent)); !errors.Is(err, barkov.ErrStateNotFound) || errors.Is(err, barkov.ErrInsufficientSupport) {
			t.Fatalf("missing context confused with low support: %v", err)
		}
	}
}
