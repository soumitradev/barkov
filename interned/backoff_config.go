package interned

import "fmt"

// BackoffConfig configures BuildBackoff. The zero value gives defaults.
type BackoffConfig struct {
	// MaxOrder is the longest context tried, 2..8. Default 6.
	MaxOrder int
	// MinOrder is the shortest context allowed during generation, 1..MaxOrder.
	// Default 1. Lower-order tables remain available for verbatim checks.
	MinOrder int
	// MinSupport is the minimum observation count (ChoicesIndex.Obs) a
	// context needs to be used at any order, 1..65535. Default 1.
	MinSupport int
	// MaxVerbatim is the longest run of consecutive emitted tokens
	// allowed to reproduce a corpus n-gram verbatim. Steering filters
	// candidates that would extend a run to this length. 0 = default
	// (MaxOrder); -1 = disabled; else 2..MaxOrder.
	MaxVerbatim int
}

func (cfg BackoffConfig) withDefaults() BackoffConfig {
	if cfg.MaxOrder == 0 {
		cfg.MaxOrder = 6
	}
	if cfg.MinOrder == 0 {
		cfg.MinOrder = 1
	}
	if cfg.MinSupport == 0 {
		cfg.MinSupport = 1
	}
	if cfg.MaxVerbatim == 0 {
		cfg.MaxVerbatim = cfg.MaxOrder
	}
	return cfg
}

func (cfg BackoffConfig) validate() error {
	switch {
	case cfg.MaxOrder < 2 || cfg.MaxOrder > 8:
		return fmt.Errorf("MaxOrder %d outside supported range 2..8", cfg.MaxOrder)
	case cfg.MinOrder < 1 || cfg.MinOrder > cfg.MaxOrder:
		return fmt.Errorf("MinOrder %d outside supported range 1..%d", cfg.MinOrder, cfg.MaxOrder)
	case cfg.MinSupport < 1 || cfg.MinSupport > 65535:
		return fmt.Errorf("MinSupport %d outside supported range 1..65535", cfg.MinSupport)
	case cfg.MaxVerbatim != -1 && (cfg.MaxVerbatim < 2 || cfg.MaxVerbatim > cfg.MaxOrder):
		return fmt.Errorf("MaxVerbatim %d must be -1 (disabled) or in 2..MaxOrder (%d)", cfg.MaxVerbatim, cfg.MaxOrder)
	}
	return nil
}
