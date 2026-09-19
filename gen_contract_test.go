package barkov

import (
	"context"
	"errors"
	"slices"
	"sync/atomic"
	"testing"
	"testing/synctest"
)

type retainingChain struct{ keys []string }

func (*retainingChain) StateSize() int                { return 1 }
func (*retainingChain) Sentinels() Sentinels[string]  { return Sentinels[string]{Begin: "a", End: "d"} }
func (*retainingChain) Encoder() StateEncoder[string] { return SepEncoder{Sep: ","} }
func (c *retainingChain) Move(key string) (string, error) {
	c.keys = append(c.keys, key)
	return string(key[0] + 1), nil
}

func TestCustomChainMayRetainStateKeys(t *testing.T) {
	chain := &retainingChain{}
	_, err := Gen(context.Background(), chain)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(chain.keys, []string{"a", "b", "c"}) {
		t.Fatalf("Move arguments changed after return: %q", chain.keys)
	}
}

type workerChain struct {
	calls                atomic.Int32
	active               atomic.Int32
	slowStarted, release chan struct{}
}

func (*workerChain) StateSize() int                { return 1 }
func (*workerChain) Sentinels() Sentinels[string]  { return Sentinels[string]{Begin: BEGIN, End: END} }
func (*workerChain) Encoder() StateEncoder[string] { return SepEncoder{Sep: SEP} }
func (c *workerChain) Move(string) (string, error) {
	if c.calls.Add(1) == 1 {
		c.active.Add(1)
		defer c.active.Add(-1)
		close(c.slowStarted)
		<-c.release
		return "slow", nil
	}
	<-c.slowStarted
	return END, nil
}
func TestThreadedGenerationJoinsWorkers(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		c := &workerChain{slowStarted: make(chan struct{}), release: make(chan struct{})}
		done := make(chan error, 1)
		go func() { _, err := Gen(context.Background(), c, WithParallelism[string](2)); done <- err }()
		synctest.Wait()
		select {
		case err := <-done:
			close(c.release)
			t.Fatalf("returned with active worker: %v", err)
		default:
		}
		close(c.release)
		synctest.Wait()
		if err := <-done; err != nil {
			t.Fatal(err)
		}
		if c.active.Load() != 0 {
			t.Fatal("worker still active")
		}
	})
}

func TestSeedRespectsCancellationAndLimit(t *testing.T) {
	chain := InitChain(1).BuildCompressed([][]string{{"a", "b", "c"}})
	for _, tc := range []struct {
		name string
		ctx  context.Context
		opts []GenOption[string]
		want error
	}{
		{"limit", context.Background(), []GenOption[string]{WithSeed([]string{"a", "b"}), WithMaxTokens[string](1)}, ErrSentenceFailedValidation},
		{"canceled", canceledContext(), []GenOption[string]{WithSeed([]string{"a"})}, context.Canceled},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var emitted []string
			var gotErr error
			for tok, err := range GenIter(tc.ctx, chain, tc.opts...) {
				if err != nil {
					gotErr = err
					break
				}
				emitted = append(emitted, tok)
			}
			if len(emitted) != 0 || !errors.Is(gotErr, tc.want) {
				t.Fatalf("emitted=%v err=%v", emitted, gotErr)
			}
		})
	}
}
func canceledContext() context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	return ctx
}

type smallPool struct{ state, history []string }

func (p *smallPool) GetState() *[]string      { return &p.state }
func (p *smallPool) PutState(v *[]string)     { p.state = *v }
func (p *smallPool) GetGenerated() *[]string  { return &p.history }
func (p *smallPool) PutGenerated(v *[]string) { p.history = *v }

func TestSmallPoolGrowsAndReturnsBuffers(t *testing.T) {
	pool := &smallPool{}
	chain := InitChain(24).BuildCompressed([][]string{{"a", "b", "c"}})
	out, err := Gen(context.Background(), chain, WithSlicePool[string](pool), WithValidator(func([]string) bool { return true }))
	if err != nil || !slices.Equal(out, []string{"a", "b", "c"}) {
		t.Fatalf("out=%v err=%v", out, err)
	}
	if cap(pool.state) < 24 || cap(pool.history) < 3 {
		t.Fatalf("pool lost grown buffers: state=%d history=%d", cap(pool.state), cap(pool.history))
	}
}

type streamChain struct{ count int }

func (*streamChain) StateSize() int                { return 1 }
func (*streamChain) Sentinels() Sentinels[string]  { return Sentinels[string]{Begin: BEGIN, End: END} }
func (*streamChain) Encoder() StateEncoder[string] { return SepEncoder{Sep: SEP} }
func (c *streamChain) Move(string) (string, error) { return c.MoveKey([1]string{}) }
func (c *streamChain) MoveKey([1]string) (string, error) {
	c.count++
	if c.count > 10000 {
		return END, nil
	}
	return "token", nil
}

func TestValidationHistoryRemainsBounded(t *testing.T) {
	pool := &smallPool{}
	calls := 0
	out, err := Gen(context.Background(), &streamChain{}, WithSlicePool[string](pool), WithValidator(func(window []string) bool { calls++; return slices.Equal(window, []string{"token", "token", "token"}) }))
	if err != nil || len(out) != 10000 || calls != 9998 {
		t.Fatalf("tokens=%d checks=%d err=%v", len(out), calls, err)
	}
	if cap(pool.history) > 128 {
		t.Fatalf("retained full generation in pool: capacity=%d", cap(pool.history))
	}
}

func TestBufferedIteratorHonorsCancellationDuringReplay(t *testing.T) {
	chain := InitChain(1).BuildCompressed([][]string{{"a", "b", "c"}})
	for _, option := range []GenOption[string]{WithOutputValidator(func([]string) bool { return true }), WithParallelism[string](2)} {
		ctx, cancel := context.WithCancel(context.Background())
		count := 0
		var result error
		for _, err := range GenIter(ctx, chain, option) {
			if err != nil {
				result = err
				break
			}
			count++
			cancel()
		}
		cancel()
		if count != 1 || !errors.Is(result, context.Canceled) {
			t.Fatalf("tokens=%d err=%v", count, result)
		}
	}
}

type widthValidator struct{ width int }

func (v widthValidator) N() int                 { return v.width }
func (v widthValidator) Validate([]string) bool { return true }

func TestGenerationRejectsInvalidOptions(t *testing.T) {
	chain := InitChain(1).BuildCompressed([][]string{{"a", "b", "c"}})
	for _, option := range []GenOption[string]{WithMaxTokens[string](-1), WithParallelism[string](-1), WithNGramValidator[string](widthValidator{0}), WithNGramValidator[string](widthValidator{-2})} {
		if out, err := Gen(context.Background(), chain, option); err == nil || out != nil {
			t.Fatalf("invalid option: %v %v", out, err)
		}
	}
}

func TestValidatorOverrideRestoresDefaultWindow(t *testing.T) {
	chain := InitChain(1).BuildCompressed([][]string{{"a", "b", "c"}})
	var windows [][]string
	validator := WithValidator(func(window []string) bool {
		windows = append(windows, slices.Clone(window))
		return true
	})
	out, err := Gen(context.Background(), chain, WithNGramValidator[string](widthValidator{0}), validator)
	if err != nil || !slices.Equal(out, []string{"a", "b", "c"}) {
		t.Fatalf("out=%v err=%v", out, err)
	}
	if len(windows) != 1 || !slices.Equal(windows[0], out) {
		t.Fatalf("default validation windows: %v", windows)
	}
}

func TestThreadedCancellationIsNotEmptySuccess(t *testing.T) {
	chain := InitChain(1).BuildCompressed([][]string{{"a"}})
	for range 10 {
		out, err := Gen(canceledContext(), chain, WithParallelism[string](4))
		if out != nil || !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled call: %v %v", out, err)
		}
	}
}
