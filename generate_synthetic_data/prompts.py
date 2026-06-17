TONE_EXPAND_PROMPT = """You are generating training data for an AI assistant.

COMPANY CONTEXT:
{context}

Here are examples showing the exact tone, style, and depth we want:
{examples}

Generate {count} NEW examples in EXACTLY the same style.
- Match the voice: direct, specific, senior-engineer level
- Always show trade-offs or caveats
- Use numbered steps for multi-part answers
- End with a specific next action when possible
- Cover different topics from the examples (don't repeat the same scenarios)

Return ONLY a JSON array of objects: [{{"instruction": "...", "input": "...", "output": "..."}}]
"""

STRUCTURED_EXPAND_PROMPT = """Generate {count} more examples for this task:

TASK: {instruction}

SCHEMA (always output this exact structure):
{schema}

EXISTING EXAMPLES (match this exact output format):
{examples}

The inputs should be realistic and varied.
Return ONLY a JSON array: [{{"input": "...", "output": "valid JSON string"}}]
The "output" value must itself be valid JSON (as a string).
"""

REASONING_EXPAND_PROMPT = """Generate {count} more examples showing step-by-step technical reasoning.

CONTEXT:
{context}

THESE EXAMPLES show the reasoning style we want — numbered steps, explicit
trade-offs, a clear bottom-line recommendation:
{examples}

Generate new examples on DIFFERENT technical scenarios.
The output must show the reasoning process, not just the answer.
Use the same structure: numbered steps + explicit trade-offs + bottom-line recommendation.

Return ONLY a JSON array: [{{"instruction": "...", "input": "...", "output": "..."}}]
"""

REFUSAL_EXPAND_PROMPT = """Generate {count} more refusal training examples.

CONTEXT:
{context}

THESE EXAMPLES show how the assistant handles things it should refuse —
with a polite redirect, not a flat "no":
{examples}

Generate new examples covering DIFFERENT refusal scenarios.
Each refusal should:
  1. Acknowledge what the user wants
  2. Explain briefly why you can't do it
  3. Offer a helpful alternative or next step

Cover different categories: destructive operations, missing information,
out-of-scope requests, decisions needing human judgment, sensitive data.

Return ONLY a JSON array: [{{"instruction": "...", "input": "...", "output": "..."}}]
"""

SYNTHETIC_PROMPT = """Generate {count} training examples about this topic:
TOPIC: {topic}

CONTEXT:
{context}

STYLE: Match this example's voice and depth:
{style_example}

Generate realistic questions a backend engineer would actually ask,
with specific, actionable answers. No generic advice.

Return ONLY a JSON array: [{{"instruction": "...", "input": "...", "output": "..."}}]
"""
