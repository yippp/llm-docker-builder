---
name: grill-me
description: Grill Me. Interview me to stress-test a plan, decision, or idea.
disable-model-invocation: true
---

# Grill Me

Interview the user until you reach a shared understanding of their plan, decision, or idea. Represent the work as a **design tree**: each decision branches into the decisions that depend on it.

## Interview in rounds

The **frontier** is the set of decisions whose prerequisites are settled. Ask every question on the current frontier in one round. Do not ask a question whose answer depends on another unanswered question; put it in a later round.

Number each question, give a recommendation, then wait for the user's answers before continuing. Each answer reshapes the tree: settled decisions expose the next frontier. Recompute it after every round.

Use this format:

```text
❓ **Q1** - **<question title>**: <question and relevant context>

➡️ <recommended answer and brief reason>

---

❓ **Q2** - **<question title>**: <question and relevant context>

➡️ <recommended answer and brief reason>
```

Ask focused questions with clear choices when practical. Include the trade-offs that matter to the decision; do not make the user decide facts that can be checked.

## Establish facts yourself

Look up facts in the repository or environment before asking about them. If a substantial fact-finding task needs a separate context, delegate that bounded research and continue with other independent frontier questions. Treat unfinished research as an unresolved prerequisite; do not ask downstream questions until it returns.

## Stop condition

The interview is complete when every branch has been considered, no decision remains silently assumed, and the user confirms that you have reached a shared understanding. Do not implement or otherwise act on the plan before that confirmation.
