import sys
from datetime import datetime
from typing import Any, Dict, List, Optional


class TokenTracker:
    """
    Real-time token consumption and savings auditor for the Meta Muse + SAM 3.1 pipeline.
    Maintains structured trace logs formatted with [MUSE-TRACE] for both console and web UI.
    """

    def __init__(self, existing_audit: Optional[Dict[str, Any]] = None) -> None:
        audit = existing_audit or {}
        self.total_input_tokens: int = int(audit.get("total_input_tokens", 0))
        self.total_image_tokens: int = int(audit.get("total_image_tokens", 0))
        self.total_output_tokens: int = int(audit.get("total_output_tokens", 0))
        self.grand_total_tokens: int = int(audit.get("grand_total_tokens", 0))
        self.estimated_tokens_saved: int = int(audit.get("estimated_tokens_saved", 0))
        self.logs: List[Dict[str, Any]] = list(audit.get("logs", []))

    def record_event(
        self,
        phase: int,
        model: str,
        operation: str,
        input_tokens: int = 0,
        image_tokens: int = 0,
        output_tokens: int = 0,
        tokens_saved: int = 0,
        status: str = "ZERO_TOKEN_LOCAL",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        step_total = input_tokens + image_tokens + output_tokens

        self.total_input_tokens += input_tokens
        self.total_image_tokens += image_tokens
        self.total_output_tokens += output_tokens
        self.grand_total_tokens = (
            self.total_input_tokens + self.total_image_tokens + self.total_output_tokens
        )
        self.estimated_tokens_saved += tokens_saved

        trace_line = (
            f"[MUSE-TRACE] [{timestamp}] [FASE-{phase}] [{model}] "
            f"{operation} | Gastados: {step_total} tok "
            f"(in={input_tokens}, img={image_tokens}, out={output_tokens}) | "
            f"Ahorrados: +{tokens_saved} tok | Estado: {status}"
        )
        try:
            print(trace_line, flush=True)
        except UnicodeEncodeError:
            # La consola de Windows (cp1252) no imprime algunos símbolos que pueden venir en nombres de productos.
            enc = getattr(sys.stdout, "encoding", None) or "ascii"
            print(trace_line.encode(enc, errors="replace").decode(enc), flush=True)

        entry = {
            "timestamp": timestamp,
            "phase": phase,
            "model": model,
            "operation": operation,
            "input_tokens": input_tokens,
            "image_tokens": image_tokens,
            "output_tokens": output_tokens,
            "step_total_tokens": step_total,
            "tokens_saved": tokens_saved,
            "status": status,
            "trace_line": trace_line,
            "details": details or {},
        }
        self.logs.append(entry)
        from backend.core import progress  # import diferido: progress no depende del tracker

        progress.note_event(phase, model, operation, status, self.grand_total_tokens, tokens_saved)
        return entry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_image_tokens": self.total_image_tokens,
            "total_output_tokens": self.total_output_tokens,
            "grand_total_tokens": self.grand_total_tokens,
            "estimated_tokens_saved": self.estimated_tokens_saved,
            "logs": self.logs,
        }

    def sync_to_session(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        session_state["token_audit"] = self.to_dict()
        return session_state

