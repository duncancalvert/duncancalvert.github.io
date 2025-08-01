---
layout: post
title: AI Agent Memory
date: 2025-05-30 18:14:10
description: Demystifying the different types of AI agent memory
tags: AI Research Agents
categories: data-science
typograms: true
thumbnail:
---

<br>
<br>
<p style="text-align: center;">
    <em>"What we call the present is given shape by an accumulation of memories."</em><br>
    — Haruki Murakami
</p>
<br>
<br>

## Why Do Agents Need Memory

Imagine trying to hold a conversation with someone who forgets everything you’ve said the moment you stop talking. That’s essentially what AI agents are without memory, perpetually amnesiac, doomed to reinvent the wheel with every interaction.

Memory gives agents context: what you asked before, what actions they've taken, and what worked (or failed) in the past. It's the difference between your trusted lieutenance and right hand man and a goldfish with Wi-Fi.

Just like humans, agents use memory to build up knowledge, learn from past mistakes, and recognize familiar faces. This helps them better personalize experiences, conduct long-term planning, and avoids repeat mistakes. Without memory, an agent can’t improve or adapt; it’s like trying to navigate a city with no map and no recollection of where you’ve been. Memory turns reactive automatons into proactive thinkers.

## What is AI Agent Memory

- An agent's "memory" is data that is not provided by the user in their prompt, but is retrieved and appended to the reasoning process via runtime calls.
- Agent memory encompasses a diverse set of references and can include everything from past user interactions, previous agent actions, external knowledge bases, system prompts, guardrails, etc.
- The additional context and knowledge provided by memory helps the agent to better conceptualize the request, plan, and then answer the user or take an action.

<br>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts_agent_memory/agent_memory.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Source: Cognitive Architectures for Language Agents
</div>
<br>

---

## Long-Term Agent Memory Types

1. **Episodic:** this type of memory contains past agent interactions and agent action logs. For example, if you asked a chatbot to "repeat the last action that it took in a previous session," it could use episodic memory to complete this request.
2. **Semantic:** this type of memory contains any knowledge the agent should have about itself, or any grounding information stored in knowledge bases that the agent has access to. For example, a vector store in a RAG application is semantic memory.
3. **Procedural:** this type of memory contains system information like the system prompt, metadata on available tools, guardrails, etc. It is usually stored and versioned in Git or prompt registry tools.

---

## Short-Term Agent Memory

- Any of the above long-term memory types that is pulled during runtime is called "short-term memory" or "working memory".
- This short-term memory is added to the user prompt and passed to the LLM with the aim of boosting performance
- Any intermediate reasoning steps/action history of the current session is also considered short-term memory when in-use.

<br>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts_agent_memory/ai_agent_gif_cropped.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Source: SwirlAI
</div>
<br>

---

### References

- [Cognitive Architectures for Language Agents](https://arxiv.org/pdf/2309.02427)
- [SwirlAI](https://www.newsletter.swirlai.com/)
