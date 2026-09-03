package interned

import "testing"

func TestMessageSet(t *testing.T) {
	v := NewVocabulary()
	corpus := v.InternCorpus([][]string{
		{"hello", "world"},
		{"quick", "brown", "fox"},
	})

	ms := NewMessageSet(corpus)
	if ms.Size() != 2 {
		t.Fatalf("expected size 2, got %d", ms.Size())
	}

	helloWorld := corpus[0]
	quickFox := corpus[1]
	other := []TokenID{v.Intern("hello"), v.Intern("there")}

	if !ms.Contains(helloWorld) {
		t.Error("expected Contains(helloWorld) to be true")
	}
	if !ms.Contains(quickFox) {
		t.Error("expected Contains(quickFox) to be true")
	}
	if ms.Contains(other) {
		t.Error("expected Contains(other) to be false")
	}

	validator := ms.NotWholeMessage()
	if validator(helloWorld) {
		t.Error("expected validator(helloWorld) to be false (rejected)")
	}
	if !validator(other) {
		t.Error("expected validator(other) to be true (accepted)")
	}
}
