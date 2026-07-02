package interned

import "testing"

// TestBuildIndexedCoreObs checks that Obs on the built indexedCore reflects
// total corpus observations, both for the Count==1 inline-follower path
// (where the running cumdist total is rolled back and never written) and
// for the ordinary Count>1 path.
func TestBuildIndexedCoreObs(t *testing.T) {
	vocab := NewVocabulary()
	corpus := [][]string{
		{"a", "b", "c"},
		{"a", "b", "c"},
		{"a", "b", "d"},
	}
	encoded := vocab.InternCorpus(corpus)
	core := buildIndexedCore[[2]TokenID](encoded)

	sentinels := DefaultSentinels()
	begin := sentinels.Begin
	a, ok := vocab.Lookup("a")
	if !ok {
		t.Fatalf("vocab missing token %q", "a")
	}
	b, ok := vocab.Lookup("b")
	if !ok {
		t.Fatalf("vocab missing token %q", "b")
	}

	// (BEGIN, BEGIN) -> "a" always: Count==1 inline path, but observed 3
	// times across the corpus.
	beginKey := [2]TokenID{begin, begin}
	beginIdx, ok := core.Model.Get(beginKey)
	if !ok {
		t.Fatalf("begin state missing from indexed model")
	}
	if beginIdx.Count != 1 {
		t.Fatalf("begin state Count = %d, want 1 (inline path precondition)", beginIdx.Count)
	}
	if beginIdx.Obs != 3 {
		t.Errorf("begin state Obs = %d, want 3", beginIdx.Obs)
	}

	// (a, b) -> {c: 2, d: 1}: Count==2 ordinary path, Obs == 3.
	abKey := [2]TokenID{a, b}
	abIdx, ok := core.Model.Get(abKey)
	if !ok {
		t.Fatalf("(a b) state missing from indexed model")
	}
	if abIdx.Count != 2 {
		t.Fatalf("(a b) Count = %d, want 2", abIdx.Count)
	}
	if abIdx.Obs != 3 {
		t.Errorf("(a b) Obs = %d, want 3", abIdx.Obs)
	}
}
