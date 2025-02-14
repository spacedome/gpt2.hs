# Large Lambda Model

Implements the forward pass of GPT-2 in Haskell.

I chose not to use a tensor library like `hasktorch` (which seems semi-abandoned?) or one of the array combinator DSLs like `accelerate`, and implemented this with the openblas bindings in `hmatrix`.
It performs better than I expected, though the scaling factors in the quadratic slowdown of attention with respect to context length is noticeably way worse than pytorch (which uses fast attention).
Once the model is loaded, it easily gets 1 token/s on an old thinkpad (until the context starts to grow).

The tokenizer is no fun at all so this is decode only and doesn't handle unicode very well (the fact that the vocab json uses unicode keys might be causing an issue with Aeson...)


### References
The best references for this are Karpathy's [NanoGPT](https://github.com/karpathy/nanoGPT) and [llm.c](https://github.com/karpathy/llm.c), as well as Brendan Bycroft's [LLM Visualizer](https://bbycroft.net/llm).

There is a handy interface for the tokenizer [online](https://tiktokenizer.vercel.app/?model=gpt2).

### Instructions

Download the open source weights and run the model with

```bash
./download_model.sh
cabal run
```

