package barkov

import (
	"context"
	"errors"
	"strconv"
	"testing"
)

// scriptChain emits next(i) as its i-th token, ignoring its state. fast
// also gives it the FastMoverKey path Gen takes without options.
type scriptChain struct {
	stateSize int
	i         int
	next      func(i int) string
}

func (c *scriptChain) StateSize() int {
	if c.stateSize > 0 {
		return c.stateSize
	}
	return 3
}
func (*scriptChain) Sentinels() Sentinels[string]  { return Sentinels[string]{Begin: BEGIN, End: END} }
func (*scriptChain) Encoder() StateEncoder[string] { return SepEncoder{Sep: SEP} }
func (c *scriptChain) Move(string) (string, error) {
	c.i++
	return c.next(c.i - 1), nil
}

type fastScriptChain struct{ scriptChain }

func (c *fastScriptChain) MoveKey([3]string) (string, error) { return c.Move("") }

// stormy loops on a three-token cycle, as copy steering once made a walk
// do. It ends after 100000 tokens so a broken guard fails the test rather
// than hanging it.
func stormy(i int) string {
	if i == 100000 {
		return END
	}
	return []string{"stormy", "and", "wet"}[i%3]
}

// counting never repeats and ends after n tokens.
func counting(n int) func(int) string {
	return func(i int) string {
		if i == n {
			return END
		}
		return strconv.Itoa(i)
	}
}

func chains(next func(int) string) map[string]GenerativeChain[string] {
	return map[string]GenerativeChain[string]{
		"slow": &scriptChain{next: next},
		"fast": &fastScriptChain{scriptChain{next: next}},
	}
}

func TestWalkStuckInACycleFails(t *testing.T) {
	for name, opts := range map[string][]GenOption[string]{
		"no options": nil,
		"options":    {WithValidator(func([]string) bool { return true })},
		"unlimited":  {WithoutTokenLimit[string]()},
	} {
		for kind, chain := range chains(stormy) {
			out, err := Gen(context.Background(), chain, opts...)
			if !errors.Is(err, ErrWalkCycle) || !errors.Is(err, ErrSentenceFailedValidation) {
				t.Errorf("%s, %s chain: %d tokens, %v; want ErrWalkCycle", name, kind, len(out), err)
			}
		}
	}
}

func TestLongWalkWithoutACycleSucceeds(t *testing.T) {
	for kind, chain := range chains(counting(DefaultMaxTokens)) {
		out, err := Gen(context.Background(), chain)
		if err != nil || len(out) != DefaultMaxTokens {
			t.Errorf("%s chain: %d tokens, %v; want %d", kind, len(out), err, DefaultMaxTokens)
		}
	}
}

func TestWalkPastTheDefaultLimitFails(t *testing.T) {
	for kind, chain := range chains(counting(DefaultMaxTokens + 1)) {
		if _, err := Gen(context.Background(), chain); !errors.Is(err, ErrTokenLimit) {
			t.Errorf("%s chain: %v, want ErrTokenLimit", kind, err)
		}
	}
	out, err := Gen(context.Background(), &scriptChain{next: counting(3 * DefaultMaxTokens)}, WithoutTokenLimit[string]())
	if err != nil || len(out) != 3*DefaultMaxTokens {
		t.Fatalf("WithoutTokenLimit: %d tokens, %v", len(out), err)
	}
}

func TestCycleGuardLargeStateSize(t *testing.T) {
	out, err := Gen(context.Background(), &scriptChain{stateSize: 12, next: counting(100)})
	if err != nil || len(out) != 100 {
		t.Fatalf("stateSize=12 non-cyclic: len=%d err=%v", len(out), err)
	}
	if _, err := Gen(context.Background(), &scriptChain{stateSize: 12, next: stormy}); !errors.Is(err, ErrWalkCycle) {
		t.Fatalf("stateSize=12 cyclic: err=%v, want ErrWalkCycle", err)
	}
}
