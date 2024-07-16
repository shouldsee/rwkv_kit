Roadmap

- Notations
  - v:hidden state
  - y:downstream tokens
  - x:upstream tokens
  - m:model parameters

- Motivations:
  - Why this project:
    - Little engineering diagnostic tools are available for controlling P(y|x,m). This project seeks to leverage the information bottleneck in RNN to distill more understanding and controllability for complex sequences models like large language models (LLM).
  - Why RWKV instead of transformers?
    - RWKV uses a **fixed-size** hidden state, whereas the size grows linearly with upstream token count in transformers.
  - Why RWKV6 instead of RWKV4?
    - RWKV6 performs much better on longer context than RWKV4  

- High level targets:
  - To promote a methodology in understanding the properties of a given RNN language model.
    - Need to abstract the engineering process, and find the important steps/ingridients
    - current eng includes finetuning, prompt tuning, prefix tuning, with collected or synthesized data.

- Targets:
  - understand the property of P(y|v)
    - estimate the entropy of the distribution? 
  - understand how varying upstream entities would change P(v|x)
    - what is the suitable model for the holistic relation between two objects y = r(x)?
  - entity patching
    - can I patch the states so that the model returns information about specific objects/concepts?
  - Information propagation plot:
    - this plot is first saw in pengbo's analysis of RWKV, where multiple hidden states are compared to a specific token vector to describe the trend of similarity.
    - this plot is generically applicable to different architectures.
