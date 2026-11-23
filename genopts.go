package barkov

import (
	"fmt"
	"runtime"
)

// StuckDetector tracks seeds that keep failing so retries can stop early.
// The barkov/stuck subpackage provides the default implementation.
type StuckDetector interface {
	RecordFailure(state string) bool // returns true if now considered stuck
	RecordSuccess(state string)
	IsStuck(state string) bool
}

// SlicePool reuses slices across attempts. Slices may have any capacity and
// come back grown. Pools used with parallel workers must be safe for
// concurrent use. barkov/interned provides a sync.Pool-backed one.
type SlicePool[T any] interface {
	GetState() *[]T
	PutState(*[]T)
	GetGenerated() *[]T
	PutGenerated(*[]T)
}

// NoPool is a SlicePool that allocates fresh slices every time.
type NoPool[T any] struct{}

func (NoPool[T]) GetState() *[]T     { s := make([]T, 0, 16); return &s }
func (NoPool[T]) PutState(*[]T)      {}
func (NoPool[T]) GetGenerated() *[]T { s := make([]T, 0, 128); return &s }
func (NoPool[T]) PutGenerated(*[]T)  {}

// GenOption configures a generation call.
type GenOption[T comparable] func(*genConfig[T])

type genConfig[T comparable] struct {
	seed            []T
	initialState    []T
	validator       func([]T) bool
	validatorWidth  int
	parallelism     int
	stuckCache      StuckDetector
	pool            SlicePool[T]
	outputValidator func([]T) bool
	maxTokens       int
	// Bools sit together so genConfig stays in the 128-byte size class
	noTokenLimit      bool
	hasValidatorWidth bool
}

// DefaultMaxTokens caps walks that have lost their way back to End
const DefaultMaxTokens = 1000

// tokenLimit is 0 for no limit
func (c *genConfig[T]) tokenLimit() int {
	switch {
	case c.noTokenLimit:
		return 0
	case c.maxTokens > 0:
		return c.maxTokens
	}
	return DefaultMaxTokens
}

func (c *genConfig[T]) windowWidth(stateSize int) int {
	if c.hasValidatorWidth {
		return c.validatorWidth
	}
	return stateSize + 2
}

// WithSeed emits seed before the generated tokens and starts the walk from it,
// padded with Sentinels.Begin when shorter than StateSize.
func WithSeed[T comparable](seed []T) GenOption[T] {
	return func(c *genConfig[T]) { c.seed = seed }
}

// WithInitialState starts the walk from tokens instead of the seed, padded the
// same way. Only the last StateSize tokens are used when tokens is longer. They
// are not emitted or counted against WithMaxTokens, and nil keeps the seed.
func WithInitialState[T comparable](tokens []T) GenOption[T] {
	return func(c *genConfig[T]) { c.initialState = tokens }
}

// WithValidator calls v with the last StateSize+2 tokens after each step and
// fails the attempt with ErrSentenceFailedValidation when it returns false.
// Shorter outputs are never checked; WithOutputValidator covers those.
func WithValidator[T comparable](v func([]T) bool) GenOption[T] {
	return func(c *genConfig[T]) {
		c.validator = v
		c.hasValidatorWidth = false
	}
}

// WithOutputValidator calls v once with the output Gen would return and fails
// the attempt with ErrSentenceFailedValidation when it returns false. GenIter
// then buffers the whole output before yielding any of it.
func WithOutputValidator[T comparable](v func([]T) bool) GenOption[T] {
	return func(c *genConfig[T]) { c.outputValidator = v }
}

// WithMaxTokens fails an attempt that emits more than n tokens, seed included,
// with ErrTokenLimit. 0 keeps DefaultMaxTokens.
func WithMaxTokens[T comparable](n int) GenOption[T] {
	return func(c *genConfig[T]) { c.maxTokens, c.noTokenLimit = n, false }
}

// WithoutTokenLimit lets a walk run until End or cancellation, for
// streams consumed through GenIter. Cycle detection still applies.
func WithoutTokenLimit[T comparable]() GenOption[T] {
	return func(c *genConfig[T]) { c.maxTokens, c.noTokenLimit = 0, true }
}

// WithNGramValidator is WithValidator with a window of v.N() tokens, so an
// NGramSet built for a different width than StateSize+2 still lines up.
func WithNGramValidator[T comparable](v WindowValidator[T]) GenOption[T] {
	return func(c *genConfig[T]) {
		c.validator = v.Validate
		c.validatorWidth, c.hasValidatorWidth = v.N(), true
	}
}

// WithThreaded races runtime.NumCPU()*8 attempts and keeps the first success.
func WithThreaded[T comparable]() GenOption[T] {
	return func(c *genConfig[T]) { c.parallelism = runtime.NumCPU() * 8 }
}

// WithParallelism races n attempts. 0 and 1 run a single attempt.
func WithParallelism[T comparable](n int) GenOption[T] {
	return func(c *genConfig[T]) { c.parallelism = n }
}

// WithStuckDetector lets threaded generation fail fast on seeds d calls stuck.
func WithStuckDetector[T comparable](d StuckDetector) GenOption[T] {
	return func(c *genConfig[T]) { c.stuckCache = d }
}

// WithSlicePool reuses p's slices for this call.
func WithSlicePool[T comparable](p SlicePool[T]) GenOption[T] {
	return func(c *genConfig[T]) { c.pool = p }
}

func (c *genConfig[T]) validate() error {
	if c.parallelism < 0 {
		return fmt.Errorf("barkov: parallelism must be nonnegative")
	}
	if c.maxTokens < 0 {
		return fmt.Errorf("barkov: maximum tokens must be nonnegative")
	}
	if c.hasValidatorWidth && c.validatorWidth < 1 {
		return fmt.Errorf("barkov: validator width must be positive")
	}
	return nil
}
