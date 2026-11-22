package barkov

import (
	"context"
	"testing"
)

// countingDetector marks a state stuck after its first failure.
type countingDetector struct{ failures map[string]int }

func (d *countingDetector) RecordFailure(state string) bool {
	d.failures[state]++
	return true
}
func (d *countingDetector) RecordSuccess(state string) { delete(d.failures, state) }
func (d *countingDetector) IsStuck(state string) bool  { return d.failures[state] > 0 }

func TestStuckCacheSeparatesInitialStates(t *testing.T) {
	chain := NewChain(ChainConfig[string]{
		StateSize: 1,
		Sentinels: Sentinels[string]{Begin: BEGIN, End: END},
		Encoder:   SepEncoder{Sep: SEP},
	})
	chain.BuildRaw([][]string{{"a", "b"}})
	detector := &countingDetector{failures: map[string]int{}}
	gen := func(initial string) error {
		_, err := Gen(context.Background(), chain,
			WithInitialState([]string{initial}),
			WithParallelism[string](2),
			WithStuckDetector[string](detector))
		return err
	}

	if err := gen("missing"); err == nil {
		t.Fatal("walk from an unseen state succeeded")
	}
	if err := gen("a"); err != nil {
		t.Fatalf("a failure from another initial state blocked this one: %v", err)
	}
}
