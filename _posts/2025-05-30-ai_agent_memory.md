---
layout: post
title: AI Agent Memory
date: 2025-05-30 18:14:10
description: Demystifying the different types of AI agent memory
tags: AI Research Agents
categories: data-science
typograms: true
---
 
<p style="text-align: center;">
    <em>"What we call the present is given shape by an accumulation of memories."</em><br>
    — Haruki Murakami
</p>

## What is AI Agent Memory
* An agent's "memory" is data that is not provided by the user in their prompt, but is retrieved and appended to the reasoning process via runtime calls. 
* Agent memory encompasses a diverse set of references and can include everything from past user interactions, previous agent actions, external knowledge bases, system prompts, guardrails, etc.
* The additional context and knowledge provided by memory helps the agent to better conceptualize the request, plan, and then answer the user or take an action.


<img src="https://github.com/duncancalvert/duncancalvert.github.io/blob/master/assets/img/agent_memory_post/agent_memory.png"/>


<em>Source: Cognitive Architectures for Language Agents</em>


---
## Long-Term Agent Memory Types
1. **Episodic:** this type of memory contains past agent interactions and agent action logs. For example, if you asked a chatbot to "repeat the last action that it took in a previous session," it could use episodic memory to complete this request.
2. **Semantic:** this type of memory contains any knowledge the agent should have about itself, or any grounding information stored in knowledge bases that the agent has access to. For example, a vector store in a RAG application is semantic memory.
3. **Procedural:** this type of memory contains system information like the system prompt, metadata on available tools, guardrails, etc. It is usually stored and versioned in Git or prompt registry tools.

---
## Short-Term Agent Memory
* Any of the above long-term memory types that is pulled during runtime is called "short-term memory" or "working memory".
* This short-term memory is added to the user prompt and passed to the LLM with the aim of boosting performance
* Any intermediate reasoning steps/action history of the current session is also considered short-term memory when in-use.


### References
* [Cognitive Architectures for Language Agents](https://arxiv.org/pdf/2309.02427)


