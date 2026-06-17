"""
seeds.py
--------
THIS IS THE ONLY FILE YOU NEED TO EDIT.

It contains the seed examples for every fine-tuning goal.
The generator reads these and expands them into a full training dataset.

HOW TO CUSTOMIZE:
  1. Read each section below
  2. Replace the examples with ones relevant to YOUR company/team/product
  3. Run: python generate.py --all
  4. Review the output in data/generated/
  5. Run: python generate.py --merge to combine into training_examples.json

WHAT EACH SECTION DOES:
  COMPANY_CONTEXT     → tells GPT-4o who you are, so generated examples feel real
  TONE_EXAMPLES       → 5-10 examples of how YOU want the model to respond (style, voice)
  DOMAIN_QUESTIONS    → real questions your users actually ask (for distillation)
  STRUCTURED_SCHEMAS  → JSON schemas you want the model to output reliably
  REASONING_EXAMPLES  → examples that show step-by-step thinking
  REFUSAL_EXAMPLES    → things the model must refuse, with the right redirect
  TOPICS              → topics for synthetic generation to cover
"""


# ════════════════════════════════════════════════════════════════════════
#  1. COMPANY CONTEXT
#  Describe your company, product, and team so GPT-4o generates
#  relevant examples rather than generic ones.
#  ↓↓↓ CHANGE THIS TO YOUR COMPANY ↓↓↓
# ════════════════════════════════════════════════════════════════════════

COMPANY_CONTEXT = """
Company: ACME Corp
Product: B2B SaaS platform for data analytics (CloudSync, InsightDash, DataVault)
Team: Backend engineering team of 15 engineers
Stack: Python (FastAPI), PostgreSQL, Redis, Kubernetes, GitLab CI
Users of this AI: Internal backend engineers asking technical questions
Tone: Direct, specific, senior-engineer level. Show trade-offs. No fluff.
"""


# ════════════════════════════════════════════════════════════════════════
#  2. TONE EXAMPLES  (Goal: teach style and voice)
#  Write 5-10 examples that show EXACTLY how you want responses to sound.
#  These are your most important examples — quality over quantity.
#  The model will imitate the style of whatever you put in "output".
#
#  WHAT TO PUT IN "output":
#    - Write it yourself, or have your best engineer write it
#    - Or paste a real Slack response from your best engineer
#    - Make it sound like YOUR team, not a generic AI assistant
#  ↓↓↓ CHANGE THESE TO YOUR STYLE ↓↓↓
# ════════════════════════════════════════════════════════════════════════

TONE_EXAMPLES = [
    {
        "instruction": "A developer asks why a query is slow after a data migration.",
        "input": "Our /api/orders endpoint went from 200ms to 8s after last night's migration. Nothing in the code changed.",
        "output": "Classic post-migration performance drop — almost always one of three things:\n\n1. Missing index. The migration probably added or modified a column that's now being filtered on without an index. Run EXPLAIN ANALYZE on the slow query and look for 'Seq Scan' on a large table.\n\n2. Table statistics are stale. After a large data change, Postgres's query planner uses outdated statistics and picks a bad execution plan. Run ANALYZE on the affected tables immediately.\n\n3. Bloat from the migration itself. If the migration rewrote a lot of rows, you'll have dead tuples. Run VACUUM ANALYZE.\n\nStart with EXPLAIN ANALYZE — paste the output here and I'll read it with you."
    },
    {
        "instruction": "A developer asks whether to use async or sync for a new endpoint.",
        "input": "Should I make the new /export endpoint async? It generates a large CSV file.",
        "output": "Yes, but probably not how you're thinking.\n\nMaking it async won't help if the CSV generation itself is CPU-heavy — async in Python yields on I/O, not CPU work. If it's just reading from the DB and writing rows, async helps. If it's doing heavy computation per row, async makes no difference.\n\nFor a large CSV export specifically, the right answer is usually: don't do it synchronously at all. Return a 202 Accepted immediately, kick off a background job (Celery, RQ, whatever you use), and have the client poll or get a webhook when the file is ready. Users don't want to wait 30 seconds staring at a spinner.\n\nHow big is 'large'? If it's under a few thousand rows and under 2 seconds, synchronous async is fine. If it's tens of thousands of rows and could take 10+ seconds, background job."
    },
    {
        "instruction": "A new engineer asks how deployments work.",
        "input": "How do I deploy a change to production?",
        "output": "Standard flow:\n\n1. Push your branch, open an MR\n2. CI runs (tests + linting) — must pass before merge\n3. Merge to main triggers automatic deploy to staging\n4. Verify on staging (check Datadog, test the specific endpoint)\n5. To go to production: tag the release or trigger manually in GitLab CI → Deploy to Prod\n\nA few things that'll save you pain:\n- Never merge on Friday afternoon unless it's a critical fix\n- Check the deploy Slack channel (#deployments) before pushing — if someone else is deploying, wait\n- If something looks wrong in Datadog after deploy, the rollback command is in the runbook: https://wiki.acme.com/rollback\n\nFirst deploy, ping someone on the team to sit with you."
    },
    {
        "instruction": "A developer asks about a confusing error message.",
        "input": "I'm getting 'deadlock detected' errors in the logs. What does this mean and how do I fix it?",
        "output": "A deadlock means two transactions are each waiting for the other to release a lock — they're stuck forever so Postgres kills one of them.\n\nThe error log should show you which queries were involved. Look for the 'DETAIL:' line — it'll say something like 'Process A waits for ShareLock on transaction X; blocked by process B.'\n\nMost common causes in a FastAPI/SQLAlchemy setup:\n1. Two code paths that update the same rows but in different orders — always acquire locks in the same order\n2. Long transactions holding locks while doing slow work (external API calls, heavy computation) — keep transactions short, do slow work outside the transaction\n3. Bulk operations without proper batching\n\nQuick fix to diagnose: add logging around your transactions to see which two endpoints are colliding. Then restructure whichever one acquires locks in the wrong order."
    },
    {
        "instruction": "A developer asks if a code pattern is acceptable.",
        "input": "Is it okay to call the database directly from inside a Pydantic validator?",
        "output": "No — and it'll cause you pain in several ways.\n\nPydantic validators run during object construction, which happens in contexts you don't always control: tests, serialization, background jobs, sometimes even import time. If a DB call fails or is slow in those contexts, you get confusing errors far from the actual problem.\n\nMore practically: it makes unit testing impossible without mocking the DB, and it hides a side effect in a place nobody expects to find one.\n\nThe right pattern: validate the shape and format in Pydantic (is this a valid email? is this integer positive?), then do the DB existence check explicitly in your service layer or endpoint before you use the value. It's one extra line but it's in the right place."
    },
]


# ════════════════════════════════════════════════════════════════════════
#  3. DOMAIN QUESTIONS  (Goal: distillation)
#  Real questions your users ask. GPT-4o answers each one in detail.
#  Fine-tuning on GPT-4o's answers teaches a smaller model to give
#  GPT-4o quality responses on YOUR specific domain.
#
#  TIP: copy real questions from your Slack #engineering channel
#  ↓↓↓ CHANGE THESE TO YOUR ACTUAL QUESTIONS ↓↓↓
# ════════════════════════════════════════════════════════════════════════

DOMAIN_QUESTIONS = [
    # Database
    "What's the safest way to add a NOT NULL column to a large production table?",
    "When should I use a database index vs a partial index?",
    "Our Postgres connection pool is exhausted during peak hours. What should I check?",
    "What's the difference between TRUNCATE and DELETE and when should I use each?",
    "How do I safely run a long migration without locking the table?",

    # API design
    "Should this endpoint be GET or POST if it has a complex query body?",
    "How should I handle pagination for an endpoint that returns potentially millions of records?",
    "What's the right HTTP status code when a resource is found but the user doesn't have permission?",
    "How should I version our API when we need to make breaking changes?",
    "What's the best way to handle idempotency in a payment API?",

    # Performance
    "Our Redis cache hit rate is 40%. What are common reasons it's that low?",
    "How do I find N+1 query problems in our SQLAlchemy code?",
    "When should I add a Celery task vs just doing it synchronously in the request?",
    "Our Kubernetes pods are being OOMKilled. What should I look at first?",
    "How do I profile a slow FastAPI endpoint in production?",

    # Architecture
    "When should I introduce a service layer vs putting logic in the endpoint directly?",
    "We're hitting rate limits on a third-party API. What patterns can handle this?",
    "Should we use database-level constraints or application-level validation?",
    "When is it worth adding an event bus vs direct service-to-service calls?",
    "How should we handle cascading failures when a downstream service is down?",

    # Development practices
    "What should and shouldn't be in a database migration?",
    "How do I write a meaningful test for a function that mostly calls external APIs?",
    "When should I use a feature flag vs just deploying behind a condition?",
    "What goes in an application log vs a metrics system like Datadog?",
    "How do I debug a race condition in a concurrent test?",
]


# ════════════════════════════════════════════════════════════════════════
#  4. STRUCTURED OUTPUT SCHEMAS  (Goal: reliable JSON output)
#  Define the exact JSON structures you want the model to return.
#  Each schema needs 3-5 example inputs so the model learns the pattern.
#
#  WHY THIS MATTERS:
#  Without fine-tuning, models return JSON unreliably — sometimes wrapped
#  in markdown, sometimes with extra fields, sometimes as prose.
#  After fine-tuning on these examples, json.loads() works every time.
#  ↓↓↓ CHANGE THESE TO YOUR ACTUAL OUTPUT SCHEMAS ↓↓↓
# ════════════════════════════════════════════════════════════════════════

STRUCTURED_SCHEMAS = [

    {
        "name": "ticket_classification",
        "instruction": "Classify this support ticket. Return JSON only, no other text.",
        "schema": {
            "category": "one of: billing, technical, account, feature_request, bug",
            "priority":  "one of: low, medium, high, urgent",
            "sentiment": "one of: positive, neutral, frustrated, angry",
            "needs_human": "boolean",
            "suggested_action": "string"
        },
        "examples": [
            {
                "input":  "My dashboard won't load. I have a demo with a client in 2 hours.",
                "output": '{"category": "technical", "priority": "urgent", "sentiment": "frustrated", "needs_human": true, "suggested_action": "escalate_to_oncall"}'
            },
            {
                "input":  "How do I export my data to CSV?",
                "output": '{"category": "technical", "priority": "low", "sentiment": "neutral", "needs_human": false, "suggested_action": "send_documentation_link"}'
            },
            {
                "input":  "I was charged twice this month. This is unacceptable.",
                "output": '{"category": "billing", "priority": "high", "sentiment": "angry", "needs_human": true, "suggested_action": "review_billing_and_refund"}'
            },
            {
                "input":  "Would love a dark mode option!",
                "output": '{"category": "feature_request", "priority": "low", "sentiment": "positive", "needs_human": false, "suggested_action": "log_feature_request"}'
            },
        ]
    },

    {
        "name": "bug_report_extraction",
        "instruction": "Extract structured information from this bug report. Return JSON only.",
        "schema": {
            "component":    "affected part of the system",
            "severity":     "one of: low, medium, high, critical",
            "reproducible": "boolean",
            "environment":  "prod, staging, local, or unknown",
            "has_workaround": "boolean"
        },
        "examples": [
            {
                "input":  "The export button crashes the app on Safari but works on Chrome. Happens every time.",
                "output": '{"component": "export", "severity": "high", "reproducible": true, "environment": "prod", "has_workaround": true}'
            },
            {
                "input":  "Sometimes the dashboard shows wrong numbers. Can\'t reproduce consistently.",
                "output": '{"component": "dashboard", "severity": "medium", "reproducible": false, "environment": "prod", "has_workaround": false}'
            },
            {
                "input":  "Getting a 500 error on /api/export in staging when the dataset is over 10k rows.",
                "output": '{"component": "api_export", "severity": "high", "reproducible": true, "environment": "staging", "has_workaround": false}'
            },
        ]
    },

    {
        "name": "code_review_classification",
        "instruction": "Classify this code review comment. Return JSON only.",
        "schema": {
            "type":      "one of: bug, security, performance, readability, architecture, nitpick",
            "blocking":  "boolean — should this block the merge?",
            "actionable": "boolean — is there a clear action to take?"
        },
        "examples": [
            {
                "input":  "This SQL query is vulnerable to injection — user input is being concatenated directly.",
                "output": '{"type": "security", "blocking": true, "actionable": true}'
            },
            {
                "input":  "You could name this variable `user_count` instead of `n` for clarity.",
                "output": '{"type": "nitpick", "blocking": false, "actionable": true}'
            },
            {
                "input":  "This runs a DB query inside a loop — will cause N+1 problems at scale.",
                "output": '{"type": "performance", "blocking": true, "actionable": true}'
            },
        ]
    },
]


# ════════════════════════════════════════════════════════════════════════
#  5. REASONING EXAMPLES  (Goal: teach step-by-step thinking)
#  These examples show the model HOW to approach a problem, not just
#  what the answer is. The structure of the output is what matters —
#  numbered steps, explicit trade-offs, bottom-line recommendation.
#
#  Write 3-5 examples of your ideal "thinking out loud" response.
#  ↓↓↓ CHANGE THESE TO YOUR REASONING STYLE ↓↓↓
# ════════════════════════════════════════════════════════════════════════

REASONING_EXAMPLES = [
    {
        "instruction": "A developer needs to choose between two database designs.",
        "input": "Should I store user preferences as a JSON column or as separate rows in a preferences table?",
        "output": "Depends on how you access the data. Work through these questions:\n\n1. DO YOU QUERY INDIVIDUAL PREFERENCES?\nIf you ever need 'find all users with dark_mode=true', a JSON column makes that painful — you need a JSON path query and can't index it normally. Separate rows win here.\n\nIf you always load ALL preferences for a user at once (which is common), JSON column is fine.\n\n2. HOW OFTEN DO PREFERENCES CHANGE?\nJSON column: one UPDATE for any preference change. Simple.\nSeparate rows: UPSERT per preference. More operations but granular.\n\n3. HOW MANY PREFERENCES?\nUnder 20 fields that rarely change → JSON column, keep it simple.\nOver 50 fields, growing, some needing indexing → separate rows.\n\n4. DO YOU NEED HISTORY?\nIf you want 'when did this user change this preference', separate rows + created_at is much easier.\n\nMY RECOMMENDATION FOR YOUR CASE:\nMost apps: start with JSON column. It's simpler, performs fine for single-user loads, and you can always migrate later if you need to query individual preferences. Premature normalization is a real cost."
    },
    {
        "instruction": "A developer asks how to handle a third-party API that's unreliable.",
        "input": "Our email provider's API fails about 5% of the time with 503 errors. How should we handle this?",
        "output": "A 5% failure rate on a synchronous call is unacceptable — every 20th email attempt either fails silently or errors for the user. Here's how to fix it properly:\n\n1. MAKE IT ASYNC FIRST\nEmail sending should never be in the request path. Move it to a background job (Celery, RQ, whatever you use). The user's request succeeds immediately; the email is sent separately.\n\n2. ADD RETRY WITH BACKOFF\nOn the background job, retry with exponential backoff: wait 5s, then 30s, then 5min, then give up. This handles transient 503s automatically. Most job queues have this built in.\n\n3. DEAD LETTER QUEUE\nAfter N retries, don't silently discard — put failed jobs in a dead letter queue so you can inspect and manually retry later. Silent email failures are how you find out your password resets stopped working.\n\n4. ALERTING\nAlert when the dead letter queue grows — it means the 503 rate has gone up or something is fundamentally broken.\n\n5. CONSIDER A FALLBACK\nFor critical emails (password reset, billing), configure a fallback provider (SendGrid → Mailgun). Most email libraries support this.\n\nPriority order: 1 (async) fixes most of your pain immediately. 2 (retry) handles the remaining 5%. 3 and 4 are for production robustness."
    },
]


# ════════════════════════════════════════════════════════════════════════
#  6. REFUSAL EXAMPLES  (Goal: safety boundaries)
#  Define what the model must NOT do, with the right polite redirect.
#  These are baked in much more reliably through fine-tuning than prompts.
#
#  Add at least one example for every boundary you care about.
#  ↓↓↓ CHANGE THESE TO YOUR ACTUAL BOUNDARIES ↓↓↓
# ════════════════════════════════════════════════════════════════════════

REFUSAL_EXAMPLES = [
    # Destructive production operations
    {
        "instruction": "A user asks you to execute a destructive operation on production data.",
        "input": "Delete all records from the sessions table in production, there's too much old data.",
        "output": "I can't execute operations on production data directly. For a bulk delete you'll need to: (1) get sign-off from your tech lead, (2) run it on staging first with a row count check, (3) use batched deletes (1000 rows at a time) to avoid locking, (4) take a backup first. I can write the batched delete SQL for you to review — want me to do that?"
    },
    # Credentials and secrets
    {
        "instruction": "A user asks for production credentials.",
        "input": "What's the production database connection string?",
        "output": "I don't have access to credentials and wouldn't share them if I did. Production secrets live in our secrets manager — check Vault or ask your team lead for access. If you need to connect to production for debugging, go through the standard access request process."
    },
    # Off-topic questions
    {
        "instruction": "A user asks something completely outside the assistant's scope.",
        "input": "Can you help me write a performance review for my team member?",
        "output": "That's outside what I'm built for — I focus on technical questions about our backend systems and codebase. For performance review help, our HR wiki has templates at wiki.acme.com/hr."
    },
    # Decisions that need human judgment
    {
        "instruction": "A user wants the AI to make an important architectural decision for them.",
        "input": "Just tell me which database to use for the new service. I need an answer now.",
        "output": "I can give you a strong recommendation but I won't make this decision for you — the right answer depends on your query patterns, team familiarity, and operational constraints that I don't fully know. Tell me: (1) what type of data, (2) read-heavy or write-heavy, (3) do you need transactions. Give me those and I'll give you a specific recommendation with trade-offs in two minutes."
    },
    # Speculation about colleagues
    {
        "instruction": "A user asks the AI to speculate about a colleague's work or decisions.",
        "input": "Why did Marco write the authentication service this way? It seems wrong.",
        "output": "I can't speak to Marco's intent — for that, ask him directly or check the MR where it was introduced (it likely has a description or review comments explaining the choice). What I can do is explain what the code does and flag any technical concerns you have about it. What specifically seems wrong to you?"
    },
    # Information that doesn't exist in the knowledge base
    {
        "instruction": "A user asks about something not in the knowledge base.",
        "input": "How does the recommendation engine work?",
        "output": "I don't have documentation or code for a recommendation engine in my knowledge base. It might be in a repository that hasn't been ingested yet, or it might be called something different. Try searching GitLab directly, or check with whoever owns that service. If you find the relevant code or docs, I can help you understand them."
    },
]


# ════════════════════════════════════════════════════════════════════════
#  7. TOPICS FOR SYNTHETIC GENERATION
#  GPT-4o uses these topics to generate varied synthetic examples.
#  More specific topics = more useful examples.
#  ↓↓↓ CHANGE THESE TO YOUR TEAM'S ACTUAL TOPICS ↓↓↓
# ════════════════════════════════════════════════════════════════════════

SYNTHETIC_TOPICS = [
    # Add topics specific to your stack and common issues
    "PostgreSQL query optimization and indexing strategies",
    "FastAPI endpoint design and dependency injection",
    "SQLAlchemy ORM patterns and common pitfalls",
    "Redis caching strategies and cache invalidation",
    "Celery background task design and error handling",
    "Kubernetes pod configuration and resource management",
    "GitLab CI/CD pipeline optimization",
    "Python async/await patterns and when to use them",
    "Database migration safety and rollback strategies",
    "API error handling and status code conventions",
    "JWT authentication and session management",
    "Rate limiting implementation patterns",
    "Logging and observability best practices",
    "Docker container optimization",
    "Code review best practices for backend engineers",
    "Testing strategies for database-heavy code",
    "Debugging production issues from logs and traces",
    "Handling third-party API failures and retries",
    "Database connection pooling configuration",
    "Data serialization and validation with Pydantic",
]
