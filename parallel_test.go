package barkov

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/soumitradev/barkov/v2/stuck"
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

func TestStuckCacheCountsEachParallelCallOnce(t *testing.T) {
	chain := InitChain(1).BuildCompressed([][]string{{"a", "b", "c"}})
	reject := WithOutputValidator(func([]string) bool { return false })
	gen := func(ctx context.Context, cache *stuck.Cache, opts ...GenOption[string]) error {
		opts = append(opts, WithParallelism[string](8), WithStuckDetector[string](cache))
		_, err := Gen(ctx, chain, opts...)
		return err
	}
	failCalls := func(t *testing.T, ctx context.Context, cache *stuck.Cache, calls int) {
		t.Helper()
		for range calls {
			if err := gen(ctx, cache, reject); err == nil {
				t.Fatal("rejected output succeeded")
			}
		}
	}

	t.Run("below threshold", func(t *testing.T) {
		cache := stuck.NewCacheWithOptions(5, time.Hour)
		failCalls(t, context.Background(), cache, 4)
		if err := gen(context.Background(), cache); err != nil {
			t.Fatalf("4 failed calls marked the seed stuck at a threshold of 5: %v", err)
		}
	})
	t.Run("at threshold", func(t *testing.T) {
		cache := stuck.NewCacheWithOptions(5, time.Hour)
		failCalls(t, context.Background(), cache, 5)
		if err := gen(context.Background(), cache); !errors.Is(err, ErrStateNotFound) {
			t.Fatalf("5 failed calls did not mark the seed stuck: %v", err)
		}
	})
	t.Run("canceled", func(t *testing.T) {
		cache := stuck.NewCacheWithOptions(1, time.Hour)
		if err := gen(canceledContext(), cache); !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled call returned %v, want context.Canceled", err)
		}
		if err := gen(context.Background(), cache); err != nil {
			t.Fatalf("a canceled call marked the seed stuck: %v", err)
		}
	})
}
