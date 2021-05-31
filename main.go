// Provides very basic markov utilities
package main

import "fmt"

func main() {
	myModel := NewLineText("./corpus.txt")

	// Test run
	for i := 0; i < 20; i++ {
		fmt.Println(myModel.GenerateSentence(20, "Man", "eats"))
		fmt.Println(myModel.GenerateSentence(20, "Man"))
		fmt.Println(myModel.GenerateSentence(20))
		fmt.Println()
	}
}
