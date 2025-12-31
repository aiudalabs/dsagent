"""Human-in-the-Loop (HITL) gateway for managing agent pauses and human feedback."""

from __future__ import annotations

import threading
from typing import Optional, TYPE_CHECKING

from dsagent.schema.models import (
    HITLMode,
    HITLAction,
    HumanFeedback,
    PlanState,
)

if TYPE_CHECKING:
    from dsagent.schema.models import ExecutionResult


class HITLGateway:
    """Gateway for Human-in-the-Loop interactions.

    Manages pause/resume flow when human intervention is required.
    Uses threading events to block execution until human provides feedback.

    Example:
        gateway = HITLGateway(mode=HITLMode.PLAN_ONLY)

        # In agent loop
        if gateway.should_pause_for_plan():
            gateway.request_plan_approval(plan)
            feedback = gateway.wait_for_feedback()  # Blocks until human responds

        # From UI/API
        gateway.provide_feedback(HumanFeedback(action=HITLAction.APPROVE))
    """

    def __init__(self, mode: HITLMode = HITLMode.NONE, timeout: float = 300.0) -> None:
        """Initialize HITL gateway.

        Args:
            mode: HITL mode controlling when to pause
            timeout: Max seconds to wait for human feedback (default 5 min)
        """
        self.mode = mode
        self.timeout = timeout

        # Threading primitives
        self._feedback_event = threading.Event()
        self._feedback: Optional[HumanFeedback] = None
        self._lock = threading.Lock()

        # Current state
        self._awaiting_type: Optional[str] = None
        self._pending_plan: Optional[PlanState] = None
        self._pending_code: Optional[str] = None
        self._pending_error: Optional[str] = None
        self._pending_answer: Optional[str] = None

        # Abort flag
        self._aborted = False

    @property
    def is_enabled(self) -> bool:
        """Check if HITL is enabled (mode is not NONE)."""
        return self.mode != HITLMode.NONE

    @property
    def is_awaiting_feedback(self) -> bool:
        """Check if gateway is waiting for human feedback."""
        return self._awaiting_type is not None

    @property
    def is_aborted(self) -> bool:
        """Check if execution was aborted by human."""
        return self._aborted

    def should_pause_for_plan(self) -> bool:
        """Check if should pause for plan approval based on mode."""
        return self.mode in (HITLMode.PLAN_ONLY, HITLMode.PLAN_AND_ANSWER, HITLMode.FULL)

    def should_pause_for_code(self) -> bool:
        """Check if should pause before code execution based on mode."""
        return self.mode == HITLMode.FULL

    def should_pause_on_error(self) -> bool:
        """Check if should pause on code execution error based on mode."""
        return self.mode in (HITLMode.ON_ERROR, HITLMode.FULL)

    def should_pause_for_answer(self) -> bool:
        """Check if should pause before final answer based on mode."""
        return self.mode in (HITLMode.PLAN_AND_ANSWER, HITLMode.FULL)

    def request_plan_approval(self, plan: PlanState) -> None:
        """Request human approval for a plan.

        Args:
            plan: The plan to approve
        """
        with self._lock:
            self._awaiting_type = "plan"
            self._pending_plan = plan
            self._feedback = None
            self._feedback_event.clear()

    def request_code_approval(self, code: str) -> None:
        """Request human approval before executing code.

        Args:
            code: The code to approve
        """
        with self._lock:
            self._awaiting_type = "code"
            self._pending_code = code
            self._feedback = None
            self._feedback_event.clear()

    def request_error_guidance(self, code: str, error: str) -> None:
        """Request human guidance after code execution error.

        Args:
            code: The code that failed
            error: The error message
        """
        with self._lock:
            self._awaiting_type = "error"
            self._pending_code = code
            self._pending_error = error
            self._feedback = None
            self._feedback_event.clear()

    def request_answer_approval(self, answer: str) -> None:
        """Request human approval for final answer.

        Args:
            answer: The answer to approve
        """
        with self._lock:
            self._awaiting_type = "answer"
            self._pending_answer = answer
            self._feedback = None
            self._feedback_event.clear()

    def wait_for_feedback(self, timeout: Optional[float] = None) -> Optional[HumanFeedback]:
        """Wait for human feedback (blocking).

        Args:
            timeout: Max seconds to wait (uses default if None)

        Returns:
            Human feedback or None if timed out
        """
        wait_timeout = timeout or self.timeout
        received = self._feedback_event.wait(timeout=wait_timeout)

        with self._lock:
            if received and self._feedback:
                feedback = self._feedback
                self._clear_pending()
                return feedback
            else:
                # Timeout - treat as rejection
                self._clear_pending()
                return HumanFeedback(
                    action=HITLAction.REJECT,
                    message="Timeout waiting for human feedback",
                )

    def provide_feedback(self, feedback: HumanFeedback) -> None:
        """Provide feedback from human (unblocks wait_for_feedback).

        Args:
            feedback: The human's feedback
        """
        with self._lock:
            self._feedback = feedback
            if feedback.action == HITLAction.REJECT:
                self._aborted = True
            self._feedback_event.set()

    def approve(self, message: Optional[str] = None) -> None:
        """Convenience method to approve current pending item.

        Args:
            message: Optional message from human
        """
        self.provide_feedback(HumanFeedback(
            action=HITLAction.APPROVE,
            message=message,
        ))

    def reject(self, message: Optional[str] = None) -> None:
        """Convenience method to reject and abort.

        Args:
            message: Optional rejection reason
        """
        self.provide_feedback(HumanFeedback(
            action=HITLAction.REJECT,
            message=message,
        ))

    def modify_plan(self, new_plan: str, message: Optional[str] = None) -> None:
        """Convenience method to provide modified plan.

        Args:
            new_plan: The modified plan text
            message: Optional message
        """
        self.provide_feedback(HumanFeedback(
            action=HITLAction.MODIFY,
            message=message,
            modified_plan=new_plan,
        ))

    def modify_code(self, new_code: str, message: Optional[str] = None) -> None:
        """Convenience method to provide modified code.

        Args:
            new_code: The modified code
            message: Optional message
        """
        self.provide_feedback(HumanFeedback(
            action=HITLAction.MODIFY,
            message=message,
            modified_code=new_code,
        ))

    def retry(self, message: Optional[str] = None) -> None:
        """Convenience method to retry failed operation.

        Args:
            message: Optional message
        """
        self.provide_feedback(HumanFeedback(
            action=HITLAction.RETRY,
            message=message,
        ))

    def skip(self, message: Optional[str] = None) -> None:
        """Convenience method to skip current step.

        Args:
            message: Optional message
        """
        self.provide_feedback(HumanFeedback(
            action=HITLAction.SKIP,
            message=message,
        ))

    def send_feedback(self, message: str) -> None:
        """Send textual feedback to inject into conversation.

        Args:
            message: Feedback message to send to agent
        """
        self.provide_feedback(HumanFeedback(
            action=HITLAction.FEEDBACK,
            message=message,
        ))

    def get_pending_state(self) -> dict:
        """Get current pending state for UI display.

        Returns:
            Dict with awaiting_type and pending items
        """
        with self._lock:
            return {
                "awaiting_type": self._awaiting_type,
                "pending_plan": self._pending_plan,
                "pending_code": self._pending_code,
                "pending_error": self._pending_error,
                "pending_answer": self._pending_answer,
            }

    def reset(self) -> None:
        """Reset gateway to initial state."""
        with self._lock:
            self._clear_pending()
            self._aborted = False

    def _clear_pending(self) -> None:
        """Clear pending state (internal, must hold lock)."""
        self._awaiting_type = None
        self._pending_plan = None
        self._pending_code = None
        self._pending_error = None
        self._pending_answer = None
        self._feedback = None
