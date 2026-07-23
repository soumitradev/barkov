// Package barkov is a markov chain text generator.
//
// Most users should start with the sibling text package instead: text.New
// builds a generator from a tokenized corpus and produces sentences that
// don't reproduce the corpus verbatim. Use this package directly when you
// need a custom token type, your own validator, or direct access to the
// chain.
package barkov
