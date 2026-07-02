package barkov

import (
	"os/exec"
	"strings"
	"testing"
)

// TestDepsStdlibOnly is a durable invariant check for Rule R4: the core
// packages of this module must not pull in any external dependencies.
// Only hashers/xxh3 and hashers/xxhash64 are allowed to import external
// modules (they exist specifically to wrap zeebo/xxh3 and cespare/xxhash).
//
// NOTE: after J.6 adds the ./text package, it must be added to this list.
func TestDepsStdlibOnly(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go binary not on PATH; skipping dependency closure check")
	}

	pkgs := []string{
		".",
		"./interned",
		"./nhash",
		"./stuck",
		"./hashers",
		"./hashers/fnv",
		"./text",
	}

	const modulePrefix = "github.com/soumitradev/barkov"

	for _, pkg := range pkgs {
		t.Run(pkg, func(t *testing.T) {
			out, err := exec.Command("go", "list", "-deps", pkg).CombinedOutput()
			if err != nil {
				t.Fatalf("go list -deps %s failed: %v\n%s", pkg, err, out)
			}

			for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
				line = strings.TrimSpace(line)
				if line == "" {
					continue
				}
				if strings.HasPrefix(line, modulePrefix) {
					continue
				}

				// Anything that isn't part of this module must be a stdlib
				// package: stdlib import paths never have a dot in their
				// first path segment (e.g. "fmt", "math/rand/v2"), whereas
				// external modules always do (e.g. "github.com/...").
				firstSegment := line
				if idx := strings.Index(line, "/"); idx != -1 {
					firstSegment = line[:idx]
				}
				if strings.Contains(firstSegment, ".") {
					t.Errorf("package %s leaks external dependency: %s", pkg, line)
				}
			}
		})
	}
}
