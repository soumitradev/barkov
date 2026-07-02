// Package text is the sugared, string-first layer over barkov: corpus of
// tokenized messages in, generated sentences out, with the fastest engine
// and anti-verbatim protection wired up for you.
//
// It is to the core packages what a facade is to a toolkit. The core gives
// you generics, pluggable encoders, and a dozen functional options; text
// pins the token type to string, picks interned.BuildBackoff as the
// engine (variable-order contexts with verbatim steering built into the
// walk), and absorbs the retry-and-timeout ceremony that every real
// consumer ends up writing by hand. When you need control, drop down:
// Chain() and Vocab() hand you the underlying core objects.
//
// text imports only the standard library and barkov's own zero-dependency
// packages, so adopting it never pulls an external module into your build.
package text

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"strings"
	"time"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/hashers/fnv"
	"github.com/soumitradev/barkov/v2/interned"
)

// avDefault is the sentinel config value meaning "anti-verbatim window not
// set; default it to Order()+2". A user-supplied 0 disables anti-verbatim;
// a positive value fixes the window.
const avDefault = -1

type config struct {
	order        int
	timeout      time.Duration
	retries      int
	threads      int
	antiVerbatim int
	rng          *rand.Rand
}

func defaultConfig() config {
	return config{
		order:        4,
		timeout:      10 * time.Second,
		retries:      64,
		threads:      0,
		antiVerbatim: avDefault,
		rng:          nil,
	}
}

// Option configures a Generator at construction time.
type Option func(*config)

// Order sets the context scale, 2..8. Default 4. Generation uses
// variable-length contexts up to Order+2 tokens (longest first, backing
// off where the corpus is thin), and anti-verbatim protection defaults
// to window Order+2.
func Order(n int) Option { return func(c *config) { c.order = n } }

// Timeout is the per-Generate time budget applied to the context. Default
// 10s; 0 means no added timeout (the caller's context still applies).
func Timeout(d time.Duration) Option { return func(c *config) { c.timeout = d } }

// Retries is the number of generation attempts per Generate call before it
// gives up with the last validation failure. Default 64. With Threads(t)
// each retry fans out t workers, so one Generate call can make up to
// Retries × t generation attempts.
//
// Under the steered engine most verbatim rejection happens during the
// walk itself, so far fewer retries are consumed than under a
// generate-reject-retry loop; the default stays as headroom for steering
// dead-ends and the whole-message check.
func Retries(n int) Option { return func(c *config) { c.retries = n } }

// Threads fans each attempt out across n goroutines (WithParallelism) and
// takes the first result that clears the validators. Default 0 (sequential).
// Combined with Retries(r), one Generate call can make up to r × n
// generation attempts. Do not combine with RNG: parallel workers share
// one random source.
func Threads(n int) Option { return func(c *config) { c.threads = n } }

// AntiVerbatim sets the n-gram window for verbatim-copy rejection. Default
// Order()+2. 0 disables anti-verbatim entirely (both the sliding n-gram
// check and the whole-message check).
func AntiVerbatim(n int) Option { return func(c *config) { c.antiVerbatim = n } }

// RNG fixes the random source for deterministic output. A Generator with a
// fixed RNG is no longer safe for concurrent Generate calls, and should not
// be combined with Threads.
func RNG(r *rand.Rand) Option { return func(c *config) { c.rng = r } }

// Generator is a ready-to-use, string-first text generator. Unless RNG was
// set, its Generate/Sentence methods are safe to call concurrently.
type Generator struct {
	chain   barkov.GenerativeChain[interned.TokenID]
	vocab   *interned.Vocabulary
	timeout time.Duration
	retries int
	threads int

	// messages holds fnv digests of every complete corpus message, so a
	// short output that reproduces a whole message is caught even when it
	// is too short for any n-gram window. nil when anti-verbatim is disabled.
	// (The sliding n-gram check itself is steered away inside the engine's
	// walk; see interned.BuildBackoff.)
	messages map[uint64]struct{}
}

// New builds a Generator from a pre-tokenized corpus (each inner slice is
// one message's tokens — text does no tokenization itself). It builds the
// vocabulary, interns the corpus, and constructs the engine internally:
// a variable-order backoff chain that samples where the corpus genuinely
// branches and steers around verbatim reproductions during the walk. It
// returns an error on an empty or unusable corpus or an invalid option.
func New(corpus [][]string, opts ...Option) (*Generator, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	if cfg.order < 2 || cfg.order > 8 {
		return nil, fmt.Errorf("text: Order(%d) out of range 2..8", cfg.order)
	}
	if cfg.retries < 1 {
		return nil, fmt.Errorf("text: Retries(%d) must be >= 1", cfg.retries)
	}
	if cfg.threads < 0 {
		return nil, fmt.Errorf("text: Threads(%d) must be >= 0", cfg.threads)
	}
	if cfg.antiVerbatim < 0 && cfg.antiVerbatim != avDefault {
		return nil, fmt.Errorf("text: AntiVerbatim(%d) must be >= 0", cfg.antiVerbatim)
	}
	if !hasContent(corpus) {
		return nil, errors.New("text: corpus has no usable messages")
	}

	// The engine's context stack reaches to Order+2 (capped at the
	// engine's max of 8) so the steering window can cover the same
	// Order+2 n-grams the anti-verbatim contract has always promised.
	maxOrder := min(cfg.order+2, 8)
	maxVerbatim := -1 // AntiVerbatim(0): steering off
	if cfg.antiVerbatim != 0 {
		window := cfg.antiVerbatim
		if window == avDefault {
			window = cfg.order + 2
		}
		// Steering at a window narrower than requested is strictly
		// stronger: no verbatim R-gram implies no verbatim (R+k)-gram.
		maxVerbatim = max(2, min(window, maxOrder))
	}

	vocab := interned.NewVocabulary()
	encoded := vocab.InternCorpus(corpus)
	chain := interned.BuildBackoff(interned.BackoffConfig{
		MaxOrder:    maxOrder,
		MaxVerbatim: maxVerbatim,
	}, encoded)
	if cfg.rng != nil {
		s, ok := chain.(barkov.RNGSettable)
		if !ok {
			return nil, errors.New("text: engine does not support RNG")
		}
		s.SetRNG(cfg.rng)
	}

	g := &Generator{
		chain:   chain,
		vocab:   vocab,
		timeout: cfg.timeout,
		retries: cfg.retries,
		threads: cfg.threads,
	}

	if cfg.antiVerbatim != 0 {
		g.messages = buildMessageSet(encoded)
	}

	return g, nil
}

func hasContent(corpus [][]string) bool {
	for _, msg := range corpus {
		if len(msg) > 0 {
			return true
		}
	}
	return false
}

// buildMessageSet digests every complete corpus message so exact
// reproductions can be rejected regardless of length.
func buildMessageSet(encoded [][]interned.TokenID) map[uint64]struct{} {
	set := make(map[uint64]struct{}, len(encoded))
	enc := interned.PackedEncoder{}
	var buf []byte
	for _, msg := range encoded {
		buf = enc.AppendEncoded(buf[:0], msg)
		set[fnv.FNV{}.Hash(buf)] = struct{}{}
	}
	return set
}

// GenOption configures a single Generate call.
type GenOption func(*genConfig)

type genConfig struct {
	seed []string
}

// Seed biases generation to start from the given words. Words not in the
// corpus vocabulary are ignored; if none remain, generation is unseeded.
// The surviving seed words are part of the returned output.
func Seed(words []string) GenOption {
	return func(gc *genConfig) { gc.seed = words }
}

// Generate returns one generated message as tokens. Seed words that survive
// vocabulary lookup lead the output. It retries on anti-verbatim rejection
// up to the configured Retries, and returns context and barkov errors
// unchanged.
func (g *Generator) Generate(ctx context.Context, opts ...GenOption) ([]string, error) {
	var gc genConfig
	for _, opt := range opts {
		opt(&gc)
	}

	seed := g.internSeed(gc.seed)

	if g.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, g.timeout)
		defer cancel()
	}

	genOpts := g.genOptions(seed)

	var lastErr error
	for attempt := 0; attempt < g.retries; attempt++ {
		out, err := barkov.Gen(ctx, g.chain, genOpts...)
		if err == nil {
			return g.vocab.DecodeTokens(out), nil
		}
		// Only anti-verbatim rejection is retryable. Context errors and
		// any other barkov error propagate unchanged.
		if !errors.Is(err, barkov.ErrSentenceFailedValidation) {
			return nil, err
		}
		lastErr = err
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
	}
	return nil, lastErr
}

// Sentence is Generate joined with single spaces.
func (g *Generator) Sentence(ctx context.Context, opts ...GenOption) (string, error) {
	out, err := g.Generate(ctx, opts...)
	if err != nil {
		return "", err
	}
	return strings.Join(out, " "), nil
}

// internSeed maps seed words to TokenIDs, dropping any not in the
// vocabulary. Returns nil (unseeded) when nothing survives.
func (g *Generator) internSeed(words []string) []interned.TokenID {
	if len(words) == 0 {
		return nil
	}
	seed := make([]interned.TokenID, 0, len(words))
	for _, w := range words {
		if id, ok := g.vocab.Lookup(w); ok {
			seed = append(seed, id)
		}
	}
	if len(seed) == 0 {
		return nil
	}
	return seed
}

// genOptions assembles the core generation options for one attempt. The
// sliding anti-verbatim check needs no option here: the backoff engine
// steers around verbatim continuations inside the walk itself.
func (g *Generator) genOptions(seed []interned.TokenID) []barkov.GenOption[interned.TokenID] {
	opts := make([]barkov.GenOption[interned.TokenID], 0, 3)
	if len(seed) > 0 {
		opts = append(opts, barkov.WithSeed(seed))
	}
	if g.messages != nil {
		opts = append(opts, barkov.WithOutputValidator(g.notWholeMessage))
	}
	if g.threads > 0 {
		opts = append(opts, barkov.WithParallelism[interned.TokenID](g.threads))
	}
	return opts
}

// notWholeMessage reports whether out is NOT an exact reproduction of a
// complete corpus message. Safe for concurrent use: it reads the
// read-only digest set and allocates only stack/local state.
func (g *Generator) notWholeMessage(out []interned.TokenID) bool {
	var buf [256]byte
	b := interned.PackedEncoder{}.AppendEncoded(buf[:0], out)
	_, found := g.messages[fnv.FNV{}.Hash(b)]
	return !found
}

// Chain returns the underlying engine for callers who need the full core
// API (raw barkov.Gen, custom validators, state inspection). The tokens it
// speaks are interned.TokenID; use Vocab to translate.
func (g *Generator) Chain() barkov.GenerativeChain[interned.TokenID] { return g.chain }

// Vocab returns the vocabulary that maps between strings and the TokenIDs
// the Chain uses.
func (g *Generator) Vocab() *interned.Vocabulary { return g.vocab }
