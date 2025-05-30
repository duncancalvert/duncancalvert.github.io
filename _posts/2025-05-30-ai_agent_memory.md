---
layout: post
title: AI Agent Memory
date: 2025-05-30 18:14:10
description: Demystifying the different types of AI agent memory
tags: AI Research Agents
categories: data-science
typograms: true
---

## What is AI Agent Memory
* An agent's "memory" is data that is not provided via the user prompt, but is appended to the prompt and passed to an LLM. Memory is diverse and encompass past chatbot interactions, past actions, external knowledge sources, system prompts, guardrails, etc.
* This memory helps the agent to better conceptualize the request, plan, and then answer the user or take an action.

---
## Long-Term Agent Memory Types
1. **Episodic:** this memory type contains past agent interactions and agent action logs. For example, if you asked a chatbot to "repeat the last action that it took in a previous session," it could use episodic memory to complete this request.
2. **Semantic:** this memory type contains any knowledge the agent should have about itself, or any grounding information stored in knowledge bases that the agent has access to. For example, a vector store in a RAG application is semantic memory.
3. **Procedural:** this memory type contains system information like the system prompt, metadata on available tools, guardrails, etc. It is usually stored and versioned in Git or prompt registry tools.

---
## Short-Term Agent Memory
* Any of the above long-term memory types that is pulled during runtime is called "short-term memory" or "working memory".
* This short-term memory is added to the user prompt and passed to the LLM with the aim of boosting performance
* Any intermediate reasoning steps/action history of the current session is also considered short-term memory when in-use.

