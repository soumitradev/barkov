package text

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"time"
)

type config struct {
	generationConfig
	order           int
	minOrder        int
	minSupport      int
	antiVerbatim    int
	antiVerbatimSet bool
	bidirectional   bool
	rng             *rand.Rand
}

func defaultConfig() config {
	return config{
		generationConfig: generationConfig{timeout: 10 * time.Second, retries: 64},
		order:            4, minOrder: 1, minSupport: 1,
	}
}

// Option configures a Generator at construction time.
type Option func(*config)

// Order sets the context scale, 2..8 (default 4). Generation backs off from
// min(Order+2, 8) tokens, and anti-verbatim defaults to that same window.
func Order(n int) Option { return func(c *config) { c.order = n } }

// MinOrder sets the shortest context allowed, 1..min(Order+2, 8) (default 1).
// Generation fails instead of falling below this floor.
func MinOrder(n int) Option { return func(c *config) { c.minOrder = n } }

// MinSupport requires at least n observations of a context before using it
// (default 1, range 1..65535). Applies to every order including MinOrder.
func MinSupport(n int) Option { return func(c *config) { c.minSupport = n } }

// Timeout sets the per-Generate time budget (default 10s, 0 adds no timeout).
func Timeout(d time.Duration) Option { return func(c *config) { c.timeout = d } }

// Retries sets generation attempts per Generate call before returning the last
// validation failure (default 64). With Threads(t), each retry fans out t
// workers for up to Retries * t attempts.
func Retries(n int) Option { return func(c *config) { c.retries = n } }

// Threads fans each attempt out across n goroutines (default 0, sequential).
// Values above 1 are rejected when a fixed RNG is set.
func Threads(n int) Option { return func(c *config) { c.threads = n } }

// AntiVerbatim sets the n-gram copy-rejection window (default Order+2).
// 0 disables both n-gram steering and whole-message checks.
func AntiVerbatim(n int) Option {
	return func(c *config) {
		c.antiVerbatim = n
		c.antiVerbatimSet = true
	}
}

// MaxTokens caps generated token length (0 means unlimited). Longer attempts retry.
func MaxTokens(n int) Option { return func(c *config) { c.maxTokens = n } }

// Bidirectional builds a reverse engine too, enabling SeedMiddle.
func Bidirectional() Option { return func(c *config) { c.bidirectional = true } }

// RNG fixes the random source for deterministic output. Middle-out generation
// uses the RNG sequentially for reproducible results. A fixed RNG is not safe
// for concurrent Generate calls or Threads > 1.
func RNG(r *rand.Rand) Option { return func(c *config) { c.rng = r } }

// GenOption configures a single Generate call.
type GenOption func(*genConfig)

type genConfig struct {
	generationConfig
	seed, seedMiddle []string
	verbatimFallback bool
	escape           float64
	noSteering       bool
	seedEnding       bool
	spliceCheck      bool
	noFragments      bool
	trace            *Trace
}

type generationConfig struct {
	maxTokens        int
	timeout          time.Duration
	retries, threads int
}

func (c generationConfig) validate(fixedRNG bool) error {
	if c.retries < 1 {
		return fmt.Errorf("text: retries must be >= 1 (got %d)", c.retries)
	}
	if c.threads < 0 {
		return fmt.Errorf("text: threads must be >= 0 (got %d)", c.threads)
	}
	if c.maxTokens < 0 {
		return fmt.Errorf("text: maximum tokens must be >= 0 (got %d)", c.maxTokens)
	}
	if c.timeout < 0 {
		return fmt.Errorf("text: timeout must be >= 0 (got %s)", c.timeout)
	}
	if fixedRNG && c.threads > 1 {
		return errors.New("text: a fixed RNG cannot be shared by parallel workers; use Threads(0) or Threads(1)")
	}
	return nil
}

// Seed starts generation from words found in the vocabulary (exact, then
// lowercased). If none match, generation runs unseeded.
func Seed(words []string) GenOption {
	return func(gc *genConfig) { gc.seed = words }
}

// SeedMiddle generates around words using Bidirectional(). Short seeds get
// observed history without lowering MinOrder or MinSupport. Output must continue
// past the seed (unless WithSeedEnding is set), and a backward-only attempt
// switches remaining retries to forward-only within the original budgets.
// All-unknown seeds fall back to unseeded.
func SeedMiddle(words []string) GenOption {
	return func(gc *genConfig) { gc.seedMiddle = words }
}

// WithMaxTokens overrides MaxTokens for this call.
func WithMaxTokens(n int) GenOption {
	return func(gc *genConfig) { gc.maxTokens = n }
}

// WithTimeout overrides Timeout for this call. d must be nonnegative.
func WithTimeout(d time.Duration) GenOption {
	return func(gc *genConfig) { gc.timeout = d }
}

// WithRetries overrides Retries for this call. n must be positive.
func WithRetries(n int) GenOption {
	return func(gc *genConfig) { gc.retries = n }
}

// WithThreads overrides Threads for this call.
func WithThreads(n int) GenOption {
	return func(gc *genConfig) { gc.threads = n }
}

// WithVerbatimFallback retries an exhausted validation failure without copy
// prevention, keeping seed, support, length and timeout constraints. The
// fallback gets its own Retries budget and changes neither the model nor
// concurrent calls.
func WithVerbatimFallback() GenOption {
	return func(gc *genConfig) { gc.verbatimFallback = true }
}

// WithEscape backs off to the next shorter context with probability
// scale*T/(N+T) (PPM method C, T distinct followers, N observations).
// Scale 0 keeps longest-context sampling. MinOrder still applies.
func WithEscape(scale float64) GenOption {
	return func(gc *genConfig) { gc.escape = scale }
}

// WithoutCopySteering disables n-gram steering while keeping whole-message rejection.
func WithoutCopySteering() GenOption {
	return func(gc *genConfig) { gc.noSteering = true }
}

// WithSeedEnding keeps a valid backward completion when SeedMiddle stops at the seed.
func WithSeedEnding() GenOption {
	return func(gc *genConfig) { gc.seedEnding = true }
}

// WithSpliceValidation makes SeedMiddle reject joined outputs that copy a run
// across the seed and resample both halves instead of keeping only forward.
func WithSpliceValidation() GenOption {
	return func(gc *genConfig) { gc.spliceCheck = true }
}

// WithoutFragments rejects outputs that occur contiguously inside one corpus
// message (checked up to the model's longest context).
func WithoutFragments() GenOption {
	return func(gc *genConfig) { gc.noFragments = true }
}
