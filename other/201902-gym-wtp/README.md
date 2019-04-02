# gym-wtp

[openai/gym](https://github.com/openai/gym) environments and reinforcement learning with [rlworkgroup/garage](https://github.com/rlworkgroup/garage/)  to simulate and the neural network (agent) to learn

- water treatment plant design, 
- water treatment plant operation, 
- and water treatment plant combined design + operation

The idea from this is to use reinforcement learning to exemplify teaching an automated agent to design/operate a simulation of a real-life example, in this case a water treatment plant.

Check the jupyter notebooks in the folder for more info and full implementations.

My related reddit post is [AI in ChemE](https://www.reddit.com/r/ChemicalEngineering/comments/arek93/ai_in_cheme/)


The agent is trained with reinforcement learning to automate the above tasks.

The models in the notebooks were implemented in the same order listed above, which is the same as increasing complexity.

The 3rd "combo design + operation" model is still WIP and doesn't work yet as of 2019-02-16

The idea triggering the combo is "how to teach a system designer to add a bypass to the system?" The bypass is not an item that brings value to system design. Its purpose is demonstrated only during operation when an unforeseen blockage happens. Hence, only the "operator" agent would understand the benefit of the bypass, and not the "designer" agent. If an agent was both "operator" and "designer" at once, then it would see the value of the bypass.

The other idea trigerring the combo is "how to teach a system designer to add redundancy in the system?" The same idea from above follows.
