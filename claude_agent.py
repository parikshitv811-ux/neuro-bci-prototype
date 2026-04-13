"""
BCI Level 2 — Claude AI Agent Integration
==========================================
Maps decoded EEG intent → structured Claude prompt → OS command
Handles multi-step task planning, personalization context, and
confirmation loop before execution.
"""

import json, time, re
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque

# ─────────────────────────────────────────────────────────────
# CONTEXT STORE  (replaces a real vector DB for the prototype)
# ─────────────────────────────────────────────────────────────
@dataclass
class UserContext:
    """Persisted user-specific context fed into every LLM prompt."""
    name: str = "User"
    frequent_contacts: list = field(default_factory=lambda: [
        "alice@example.com", "bob@example.com", "team@example.com"
    ])
    recent_apps: list = field(default_factory=lambda: [
        "terminal", "browser", "vscode", "email_client"
    ])
    recent_actions: deque = field(default_factory=lambda: deque(maxlen=10))
    preferences: dict = field(default_factory=lambda: {
        "email_signature": "Best regards",
        "browser": "firefox",
        "default_email": "user@example.com",
    })

    def add_action(self, action: str):
        self.recent_actions.appendleft({"action": action, "ts": time.time()})

    def summary(self) -> str:
        recents = ", ".join(self.recent_apps[:3])
        contacts = ", ".join(self.frequent_contacts[:3])
        last = list(self.recent_actions)[:3]
        last_str = "; ".join(a["action"] for a in last) if last else "none"
        return (
            f"User: {self.name}\n"
            f"Recent apps: {recents}\n"
            f"Frequent contacts: {contacts}\n"
            f"Last actions: {last_str}\n"
            f"Preferences: {json.dumps(self.preferences)}"
        )


# ─────────────────────────────────────────────────────────────
# COARSE INTENT → STRUCTURED COMMAND  (via Claude API)
# ─────────────────────────────────────────────────────────────
class ClaudeIntentAgent:
    """
    Sends coarse EEG intent label to Claude and receives a
    structured JSON action plan (steps + OS commands).

    In production: uses anthropic.Anthropic() client.
    In simulation: returns deterministic mock responses.
    """

    SYSTEM_PROMPT = """You are an AI agent integrated into a Brain-Computer Interface (BCI) system.
You receive a coarse intent label decoded from EEG signals and the user's context.
Your job is to:
1. Infer the most likely specific action the user wants to perform.
2. Return a structured JSON action plan — nothing else.

Output format (strict JSON, no markdown):
{
  "intent_label": "<original coarse label>",
  "interpreted_action": "<natural language description>",
  "steps": [
    {"step": 1, "description": "<what to do>", "os_command": "<command type>", "params": {}}
  ],
  "confirmation_required": true/false,
  "confirmation_message": "<human-readable confirmation prompt>"
}

Rules:
- Keep steps minimal and safe.
- Set confirmation_required=true for destructive or irreversible actions (send email, delete).
- os_command must be one of: open_app, type_text, click, scroll, hotkey, send_email, idle.
- Respond ONLY with valid JSON.
"""

    # Mock responses for simulation (no API key needed for demo)
    MOCK_RESPONSES = {
        "communication": {
            "intent_label": "communication",
            "interpreted_action": "Compose email to most recent contact",
            "steps": [
                {"step": 1, "description": "Open email client", "os_command": "open_app",
                 "params": {"app": "email_client"}},
                {"step": 2, "description": "Create new email to alice@example.com",
                 "os_command": "send_email",
                 "params": {"to": "alice@example.com", "subject": "Quick message",
                             "body": "Hi Alice, "}},
            ],
            "confirmation_required": True,
            "confirmation_message": "Send email to alice@example.com?"
        },
        "open_app": {
            "intent_label": "open_app",
            "interpreted_action": "Open terminal (most recently used dev tool)",
            "steps": [
                {"step": 1, "description": "Open terminal", "os_command": "hotkey",
                 "params": {"keys": ["ctrl", "alt", "t"]}}
            ],
            "confirmation_required": False,
            "confirmation_message": ""
        },
        "scroll_down": {
            "intent_label": "scroll_down",
            "interpreted_action": "Scroll down in active window",
            "steps": [
                {"step": 1, "description": "Scroll down 3 units", "os_command": "scroll",
                 "params": {"direction": "down", "amount": 3}}
            ],
            "confirmation_required": False,
            "confirmation_message": ""
        },
        "scroll_up": {
            "intent_label": "scroll_up",
            "interpreted_action": "Scroll up in active window",
            "steps": [
                {"step": 1, "description": "Scroll up 3 units", "os_command": "scroll",
                 "params": {"direction": "up", "amount": 3}}
            ],
            "confirmation_required": False,
            "confirmation_message": ""
        },
        "click": {
            "intent_label": "click",
            "interpreted_action": "Left click at current cursor position",
            "steps": [
                {"step": 1, "description": "Left click", "os_command": "click",
                 "params": {"button": "left"}}
            ],
            "confirmation_required": False,
            "confirmation_message": ""
        },
        "idle": {
            "intent_label": "idle",
            "interpreted_action": "No action — user is at rest",
            "steps": [],
            "confirmation_required": False,
            "confirmation_message": ""
        }
    }

    def __init__(self, context: UserContext, use_real_api: bool = False,
                 api_key: Optional[str] = None):
        self.context = context
        self.use_real_api = use_real_api
        self.api_key = api_key
        self._client = None

        if use_real_api:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                print("[ClaudeAgent] anthropic package not found — falling back to mock mode.")
                self.use_real_api = False

    def resolve(self, intent_label: str) -> dict:
        """Expand coarse intent label into a structured action plan."""
        if self.use_real_api and self._client:
            return self._resolve_via_api(intent_label)
        return self._resolve_mock(intent_label)

    def _resolve_via_api(self, intent_label: str) -> dict:
        user_message = (
            f"Coarse EEG intent decoded: '{intent_label}'\n\n"
            f"User context:\n{self.context.summary()}"
        )
        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}]
            )
            raw = response.content[0].text.strip()
            # Strip any accidental markdown fences
            raw = re.sub(r"```json|```", "", raw).strip()
            return json.loads(raw)
        except Exception as e:
            print(f"[ClaudeAgent] API error: {e} — using mock fallback")
            return self._resolve_mock(intent_label)

    def _resolve_mock(self, intent_label: str) -> dict:
        """Return deterministic mock response for testing without API key."""
        plan = self.MOCK_RESPONSES.get(intent_label, self.MOCK_RESPONSES["idle"]).copy()
        # Inject live context into email step
        if intent_label == "communication" and self.context.frequent_contacts:
            top_contact = self.context.frequent_contacts[0]
            for step in plan["steps"]:
                if step["os_command"] == "send_email":
                    step["params"]["to"] = top_contact
            plan["confirmation_message"] = f"Send email to {top_contact}?"
        return plan


# ─────────────────────────────────────────────────────────────
# CONFIRMATION ENGINE  (second-signal or timeout)
# ─────────────────────────────────────────────────────────────
class ConfirmationEngine:
    """
    Waits for a confirmation signal from the user (second EEG trigger,
    eye-blink, or timeout) before executing irreversible actions.

    Modes:
        'auto'    — always confirm (for testing)
        'deny'    — always deny (for testing)
        'blink'   — simulate eye-blink confirmation (EEG alpha burst)
        'timeout' — auto-confirm after N seconds (assistive use)
    """

    def __init__(self, mode: str = "auto", timeout_s: float = 3.0):
        self.mode = mode
        self.timeout = timeout_s

    def request_confirmation(self, message: str) -> bool:
        print(f"\n  [CONFIRMATION] {message}")
        if self.mode == "auto":
            print("  [CONFIRMATION] Auto-confirmed (test mode)")
            return True
        elif self.mode == "deny":
            print("  [CONFIRMATION] Auto-denied (test mode)")
            return False
        elif self.mode == "timeout":
            print(f"  [CONFIRMATION] Waiting {self.timeout}s for abort signal...")
            time.sleep(min(self.timeout, 0.1))  # shortened for demo
            print("  [CONFIRMATION] No abort received — confirmed by timeout")
            return True
        elif self.mode == "blink":
            # In production: listen for alpha-burst (8-13 Hz power spike)
            # Here we simulate it
            print("  [CONFIRMATION] Waiting for eye-blink signal (simulated)...")
            time.sleep(0.05)
            print("  [CONFIRMATION] Blink detected — confirmed")
            return True
        return False
