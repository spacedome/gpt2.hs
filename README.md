# Large Lambda Model

Implements the forward pass of GPT-2 in Haskell.

I chose not to use a tensor library like `hasktorch` (which seems semi-abandoned?) or one of the array combinator DSLs like `accelerate`, and implemented this with the openblas bindings in `hmatrix`.
It performs better than I expected, though the scaling factors in the quadratic slowdown of attention with respect to context length is noticeably way worse than pytorch (which uses fast attention).
The main performance issue is actually loading the model, as I'm not sure how to mmap a file directly into the `Storable` backed Vectors, so there's some inefficient lazy bytestring traversal that takes about a minute.
Once the model is loaded, it easily gets 1 token/s on an old thinkpad.

The tokenizer is no fun at all so this is decode only and doesn't handle unicode very well.


### Instructions

Download the open source weights and run the model with

```bash
./download_model.sh
cabal run
```

