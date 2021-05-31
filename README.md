# Barkov

A shitty markov implementation in Go.

A demo is included, of Onion and r/NotTheOnion headlines.

## Usage:

```go
package main

import "github.com/soumitradev/barkov"

func main() {
	myModel := barkov.NewLineText("./corpus.txt")

	// Test run
	for i := 0; i < 20; i++ {
		fmt.Println(myModel.GenerateSentence(20, "Man", "eats"))
		fmt.Println(myModel.GenerateSentence(20, "Man"))
		fmt.Println(myModel.GenerateSentence(20))
		fmt.Println()
	}
}
```
