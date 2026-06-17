"""
agents/agent.py — agentic loop with tools.
The agent decides which tools to call based on the user's message.
Wraps the RAG pipeline as one of its tools.
"""
import json, os, sys, uuid
from datetime import datetime, date
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openai import OpenAI
from inference.pipeline import ask as rag_ask
import config


# ── Tool definitions (what the LLM sees) ─────────────────────────────────────

TOOLS = [
    {"type":"function","function":{
        "name": "search_knowledge",
        "description": "Search internal documentation, code, and policies. Use for any policy or technical question.",
        "parameters": {"type":"object","properties":{
            "query": {"type":"string"},
            "type_filter": {"type":"string","enum":["code","issue","wiki","document"],
                            "description":"Optional: filter by source type"}
        },"required":["query"]},
    }},
    {"type":"function","function":{
        "name": "lookup_order",
        "description": "Look up a customer order by ID. Always call this before discussing a specific order.",
        "parameters": {"type":"object","properties":{
            "order_id": {"type":"string"}
        },"required":["order_id"]},
    }},
    {"type":"function","function":{
        "name": "check_return_eligibility",
        "description": "Check if an order is within the 30-day return window.",
        "parameters": {"type":"object","properties":{
            "order_id": {"type":"string"}
        },"required":["order_id"]},
    }},
    {"type":"function","function":{
        "name": "calculate_refund",
        "description": "Calculate the refund amount for an order.",
        "parameters": {"type":"object","properties":{
            "order_id":           {"type":"string"},
            "refund_type":        {"type":"string","enum":["full","partial"]},
            "partial_percentage": {"type":"number"},
        },"required":["order_id","refund_type"]},
    }},
    {"type":"function","function":{
        "name": "create_ticket",
        "description": "Create a support ticket for escalation. Use when issue cannot be resolved directly.",
        "parameters": {"type":"object","properties":{
            "customer_email": {"type":"string"},
            "subject":        {"type":"string"},
            "description":    {"type":"string"},
            "priority":       {"type":"string","enum":["low","medium","high","urgent"]},
            "order_id":       {"type":"string"},
        },"required":["customer_email","subject","description","priority"]},
    }},
]


# ── Tool implementations ──────────────────────────────────────────────────────

def search_knowledge(query: str, type_filter: str = None) -> dict:
    result = rag_ask(query, type_filter=type_filter)
    return {"answer": result["answer"],
            "sources": [s["source"] for s in result["sources"][:3]]}

def lookup_order(order_id: str) -> dict:
    if not os.path.exists(config.ORDERS_PATH):
        return {"error": "Order database not found"}
    with open(config.ORDERS_PATH) as f:
        orders = json.load(f)
    order = orders.get(order_id.upper())
    return order if order else {"error": f"Order {order_id} not found"}

def check_return_eligibility(order_id: str) -> dict:
    order = lookup_order(order_id)
    if "error" in order:
        return order
    if order["status"] == "cancelled":
        return {"eligible": False, "reason": "Already cancelled/refunded"}
    if order["status"] in ("processing","shipped"):
        return {"eligible": True, "reason": "Not yet delivered — can cancel", "action":"cancel"}
    delivery = order.get("delivery_date")
    if not delivery:
        return {"eligible": False, "reason": "No delivery date on record"}
    days = (date.today() - datetime.strptime(delivery,"%Y-%m-%d").date()).days
    window = 30
    if days <= window:
        return {"eligible": True, "days_since": days, "days_remaining": window - days,
                "reason": f"Delivered {days} days ago — within {window}-day window"}
    return {"eligible": False, "days_since": days,
            "reason": f"Delivered {days} days ago — outside {window}-day window"}

def calculate_refund(order_id: str, refund_type: str, partial_percentage: float = None) -> dict:
    order = lookup_order(order_id)
    if "error" in order:
        return order
    total  = order.get("total",0)
    amount = total if refund_type == "full" else round(total * ((partial_percentage or 50)/100), 2)
    return {"order_id": order_id, "order_total": total, "refund_amount": amount,
            "refund_type": refund_type, "note": "Processes in 3-5 business days"}

def create_ticket(customer_email: str, subject: str, description: str,
                  priority: str, order_id: str = None) -> dict:
    os.makedirs(os.path.dirname(config.TICKETS_PATH), exist_ok=True)
    tickets = {}
    if os.path.exists(config.TICKETS_PATH):
        with open(config.TICKETS_PATH) as f:
            tickets = json.load(f)
    tid = f"TKT-{str(uuid.uuid4())[:8].upper()}"
    sla = {"low":72,"medium":24,"high":8,"urgent":2}[priority]
    tickets[tid] = {"ticket_id":tid,"customer_email":customer_email,"subject":subject,
                    "description":description,"priority":priority,"order_id":order_id,
                    "status":"open","created_at":datetime.now().isoformat(),"sla_hours":sla}
    with open(config.TICKETS_PATH,"w") as f:
        json.dump(tickets, f, indent=2)
    return {"ticket_id":tid,"status":"created","sla":f"Response within {sla}h",
            "message":f"Ticket {tid} created — customer will be emailed at {customer_email}"}

TOOL_MAP = {
    "search_knowledge":       search_knowledge,
    "lookup_order":           lookup_order,
    "check_return_eligibility": check_return_eligibility,
    "calculate_refund":       calculate_refund,
    "create_ticket":          create_ticket,
}

def _execute(name: str, args: dict) -> str:
    fn = TOOL_MAP.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return json.dumps(fn(**args), default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Agentic loop ──────────────────────────────────────────────────────────────

def run(user_message: str,
        history: list[dict] = None,
        session_id: str = None,
        verbose: bool = True) -> dict:
    """
    Run the agent for one user message.
    Pass history back on each turn for multi-turn conversations.
    """
    session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    messages   = (history or []) + [{"role":"user","content":user_message}]

    if not any(m["role"]=="system" for m in messages):
        messages = [{"role":"system","content":config.AGENT_SYSTEM_PROMPT}] + messages

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    if config.MODEL_BACKEND == "vllm":
        client = OpenAI(base_url=config.VLLM_CHAT_URL, api_key="x")

    model      = config.OPENAI_CHAT_MODEL if config.MODEL_BACKEND=="openai" else config.VLLM_CHAT_MODEL
    tool_log   = []
    turns      = 0

    if verbose:
        print(f"\n  User: {user_message}")

    while turns < config.AGENT_MAX_TURNS:
        turns += 1
        resp   = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS,
            tool_choice="auto", temperature=config.AGENT_TEMP,
        )
        msg    = resp.choices[0].message
        finish = resp.choices[0].finish_reason

        if finish == "stop" or not msg.tool_calls:
            answer = msg.content or ""
            messages.append({"role":"assistant","content":answer})
            if verbose:
                print(f"\n  Agent: {answer}")
            _log(session_id, {"answer":answer,"tool_calls":tool_log,"turns":turns})
            return {"answer":answer,"tool_calls":tool_log,"turns":turns,"history":messages}

        messages.append(msg)
        if verbose:
            print(f"\n  [Turn {turns}] Tools: {[tc.function.name for tc in msg.tool_calls]}")

        for tc in msg.tool_calls:
            name   = tc.function.name
            args   = json.loads(tc.function.arguments)
            result = _execute(name, args)
            if verbose:
                r = json.loads(result)
                print(f"    → {name}({args})")
                print(f"    ← {str(r)[:120]}")
            messages.append({"role":"tool","tool_call_id":tc.id,"content":result})
            tool_log.append({"tool":name,"args":args,"result":json.loads(result)})

    fallback = "I've hit a complexity limit. Let me create a ticket so a human can help."
    return {"answer":fallback,"tool_calls":tool_log,"turns":turns,"history":messages}


def _log(session_id: str, data: dict):
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    with open(os.path.join(config.LOGS_DIR, f"{session_id}.jsonl"),"a") as f:
        f.write(json.dumps({**data,"ts":datetime.now().isoformat()}) + "\n")
