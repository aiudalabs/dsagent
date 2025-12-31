"""Tests for Human-in-the-Loop (HITL) functionality."""

import pytest
import threading
import time

from dsagent.schema.models import (
    HITLMode,
    HITLAction,
    HumanFeedback,
    PlanState,
    PlanStep,
)
from dsagent.core.hitl import HITLGateway


class TestHITLMode:
    """Tests for HITLMode enum."""

    def test_hitl_modes_exist(self):
        """Test that all HITL modes are defined."""
        assert HITLMode.NONE == "none"
        assert HITLMode.PLAN_ONLY == "plan_only"
        assert HITLMode.ON_ERROR == "on_error"
        assert HITLMode.PLAN_AND_ANSWER == "plan_and_answer"
        assert HITLMode.FULL == "full"

    def test_hitl_action_types(self):
        """Test that all HITL actions are defined."""
        assert HITLAction.APPROVE == "approve"
        assert HITLAction.REJECT == "reject"
        assert HITLAction.MODIFY == "modify"
        assert HITLAction.RETRY == "retry"
        assert HITLAction.SKIP == "skip"
        assert HITLAction.FEEDBACK == "feedback"


class TestHumanFeedback:
    """Tests for HumanFeedback model."""

    def test_create_approve_feedback(self):
        """Test creating approve feedback."""
        feedback = HumanFeedback(action=HITLAction.APPROVE)
        assert feedback.action == HITLAction.APPROVE
        assert feedback.message is None

    def test_create_feedback_with_message(self):
        """Test creating feedback with message."""
        feedback = HumanFeedback(
            action=HITLAction.FEEDBACK,
            message="Try a different approach",
        )
        assert feedback.action == HITLAction.FEEDBACK
        assert feedback.message == "Try a different approach"

    def test_create_modify_feedback(self):
        """Test creating modify feedback with plan."""
        feedback = HumanFeedback(
            action=HITLAction.MODIFY,
            modified_plan="1. [ ] New step\n2. [ ] Another step",
        )
        assert feedback.action == HITLAction.MODIFY
        assert "New step" in feedback.modified_plan


class TestHITLGateway:
    """Tests for HITLGateway class."""

    def test_gateway_disabled_by_default(self):
        """Test that gateway with NONE mode is disabled."""
        gateway = HITLGateway(mode=HITLMode.NONE)
        assert not gateway.is_enabled
        assert not gateway.should_pause_for_plan()
        assert not gateway.should_pause_for_code()
        assert not gateway.should_pause_on_error()
        assert not gateway.should_pause_for_answer()

    def test_plan_only_mode(self):
        """Test PLAN_ONLY mode pauses only for plan."""
        gateway = HITLGateway(mode=HITLMode.PLAN_ONLY)
        assert gateway.is_enabled
        assert gateway.should_pause_for_plan()
        assert not gateway.should_pause_for_code()
        assert not gateway.should_pause_on_error()
        assert not gateway.should_pause_for_answer()

    def test_on_error_mode(self):
        """Test ON_ERROR mode pauses only on errors."""
        gateway = HITLGateway(mode=HITLMode.ON_ERROR)
        assert gateway.is_enabled
        assert not gateway.should_pause_for_plan()
        assert not gateway.should_pause_for_code()
        assert gateway.should_pause_on_error()
        assert not gateway.should_pause_for_answer()

    def test_plan_and_answer_mode(self):
        """Test PLAN_AND_ANSWER mode."""
        gateway = HITLGateway(mode=HITLMode.PLAN_AND_ANSWER)
        assert gateway.is_enabled
        assert gateway.should_pause_for_plan()
        assert not gateway.should_pause_for_code()
        assert not gateway.should_pause_on_error()
        assert gateway.should_pause_for_answer()

    def test_full_mode(self):
        """Test FULL mode pauses at all points."""
        gateway = HITLGateway(mode=HITLMode.FULL)
        assert gateway.is_enabled
        assert gateway.should_pause_for_plan()
        assert gateway.should_pause_for_code()
        assert gateway.should_pause_on_error()
        assert gateway.should_pause_for_answer()

    def test_request_plan_approval(self):
        """Test requesting plan approval."""
        gateway = HITLGateway(mode=HITLMode.PLAN_ONLY)
        plan = PlanState(
            steps=[
                PlanStep(number=1, description="Load data"),
                PlanStep(number=2, description="Analyze"),
            ],
            raw_text="1. [ ] Load data\n2. [ ] Analyze",
        )

        gateway.request_plan_approval(plan)

        assert gateway.is_awaiting_feedback
        state = gateway.get_pending_state()
        assert state["awaiting_type"] == "plan"
        assert state["pending_plan"] == plan

    def test_provide_feedback_approve(self):
        """Test providing approve feedback."""
        gateway = HITLGateway(mode=HITLMode.PLAN_ONLY, timeout=1.0)
        plan = PlanState(steps=[], raw_text="test")

        # Start request in background
        gateway.request_plan_approval(plan)

        # Provide feedback from another thread
        def provide():
            time.sleep(0.1)
            gateway.approve("Looks good!")

        thread = threading.Thread(target=provide)
        thread.start()

        # Wait for feedback
        feedback = gateway.wait_for_feedback()
        thread.join()

        assert feedback is not None
        assert feedback.action == HITLAction.APPROVE
        assert feedback.message == "Looks good!"
        assert not gateway.is_awaiting_feedback

    def test_provide_feedback_reject(self):
        """Test providing reject feedback sets aborted flag."""
        gateway = HITLGateway(mode=HITLMode.PLAN_ONLY, timeout=1.0)
        plan = PlanState(steps=[], raw_text="test")

        gateway.request_plan_approval(plan)

        def provide():
            time.sleep(0.1)
            gateway.reject("Bad plan")

        thread = threading.Thread(target=provide)
        thread.start()

        feedback = gateway.wait_for_feedback()
        thread.join()

        assert feedback is not None
        assert feedback.action == HITLAction.REJECT
        assert gateway.is_aborted

    def test_provide_modified_plan(self):
        """Test providing modified plan."""
        gateway = HITLGateway(mode=HITLMode.PLAN_ONLY, timeout=1.0)
        plan = PlanState(steps=[], raw_text="test")

        gateway.request_plan_approval(plan)

        def provide():
            time.sleep(0.1)
            gateway.modify_plan("1. [ ] Better step")

        thread = threading.Thread(target=provide)
        thread.start()

        feedback = gateway.wait_for_feedback()
        thread.join()

        assert feedback is not None
        assert feedback.action == HITLAction.MODIFY
        assert feedback.modified_plan == "1. [ ] Better step"

    def test_timeout_returns_reject(self):
        """Test that timeout returns reject feedback."""
        gateway = HITLGateway(mode=HITLMode.PLAN_ONLY, timeout=0.1)
        plan = PlanState(steps=[], raw_text="test")

        gateway.request_plan_approval(plan)

        # Don't provide feedback, let it timeout
        feedback = gateway.wait_for_feedback()

        assert feedback is not None
        assert feedback.action == HITLAction.REJECT
        assert "Timeout" in feedback.message

    def test_request_code_approval(self):
        """Test requesting code approval."""
        gateway = HITLGateway(mode=HITLMode.FULL)
        code = "print('hello')"

        gateway.request_code_approval(code)

        state = gateway.get_pending_state()
        assert state["awaiting_type"] == "code"
        assert state["pending_code"] == code

    def test_request_error_guidance(self):
        """Test requesting error guidance."""
        gateway = HITLGateway(mode=HITLMode.ON_ERROR)
        code = "1/0"
        error = "ZeroDivisionError"

        gateway.request_error_guidance(code, error)

        state = gateway.get_pending_state()
        assert state["awaiting_type"] == "error"
        assert state["pending_code"] == code
        assert state["pending_error"] == error

    def test_request_answer_approval(self):
        """Test requesting answer approval."""
        gateway = HITLGateway(mode=HITLMode.PLAN_AND_ANSWER)
        answer = "The analysis shows..."

        gateway.request_answer_approval(answer)

        state = gateway.get_pending_state()
        assert state["awaiting_type"] == "answer"
        assert state["pending_answer"] == answer

    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        gateway = HITLGateway(mode=HITLMode.PLAN_ONLY)
        plan = PlanState(steps=[], raw_text="test")

        gateway.request_plan_approval(plan)
        gateway.reject("test")

        assert gateway.is_aborted

        gateway.reset()

        assert not gateway.is_aborted
        assert not gateway.is_awaiting_feedback
        state = gateway.get_pending_state()
        assert state["awaiting_type"] is None

    def test_skip_action(self):
        """Test skip action."""
        gateway = HITLGateway(mode=HITLMode.ON_ERROR, timeout=1.0)

        gateway.request_error_guidance("code", "error")

        def provide():
            time.sleep(0.1)
            gateway.skip("Just skip it")

        thread = threading.Thread(target=provide)
        thread.start()

        feedback = gateway.wait_for_feedback()
        thread.join()

        assert feedback.action == HITLAction.SKIP
        assert feedback.message == "Just skip it"

    def test_send_feedback_message(self):
        """Test sending feedback message."""
        gateway = HITLGateway(mode=HITLMode.ON_ERROR, timeout=1.0)

        gateway.request_error_guidance("code", "error")

        def provide():
            time.sleep(0.1)
            gateway.send_feedback("Try using pandas instead")

        thread = threading.Thread(target=provide)
        thread.start()

        feedback = gateway.wait_for_feedback()
        thread.join()

        assert feedback.action == HITLAction.FEEDBACK
        assert feedback.message == "Try using pandas instead"
