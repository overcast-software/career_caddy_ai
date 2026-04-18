"""Tests for the Agent Wizard delegation rule in the main chat prompt.

After step B (agent delegation), the main chat prompt only contains a
short rule telling the agent to call `ask_onboarding_wizard` and pass
the user's message verbatim. The detailed behavior rules (review loop,
stop flow, profile-basics handling, reconcile policy) live in the
sub-agent's prompt — see test_onboarding_agent.py.
"""

from mcp_servers.chat_server import (
    _build_system_prompt,
    _render_onboarding,
    _should_inject_aw,
)


SAMPLE_PROFILE = (
    "First name: Jane\nLast name: (blank)\nEmail: jane@example.com"
)


class TestRenderOnboarding:
    """Kept for parity — used anywhere in the chat_server code path."""

    def test_all_incomplete(self):
        lines, next_step = _render_onboarding(
            {"wizard_enabled": True, "resume_imported": False}
        )
        assert "wizard_enabled: yes" in lines
        assert "resume_imported: no" in lines
        assert next_step == "Fill in name + email"

    def test_skips_completed_for_next_step(self):
        lines, next_step = _render_onboarding(
            {
                "wizard_enabled": True,
                "profile_basics": True,
                "resume_imported": False,
            }
        )
        assert "profile_basics: yes" in lines
        assert next_step == "Import a resume"

    def test_wizard_disabled_short_circuits_next_step(self):
        _, next_step = _render_onboarding({"wizard_enabled": False})
        assert "disabled" in next_step

    def test_all_complete(self):
        payload = {
            "wizard_enabled": True,
            "profile_basics": True,
            "resume_imported": True,
            "resume_reviewed": True,
            "first_job_post": True,
            "first_score": True,
            "first_cover_letter": True,
        }
        _, next_step = _render_onboarding(payload)
        assert next_step == "none — all setup complete"


class TestDelegationRuleInjection:
    def test_no_onboarding_leaves_prompt_unchanged(self):
        prompt = _build_system_prompt(SAMPLE_PROFILE)
        assert "ask_onboarding_wizard" not in prompt

    def test_onboarding_with_pending_step_injects_delegation_rule(self):
        prompt = _build_system_prompt(
            SAMPLE_PROFILE,
            onboarding={"wizard_enabled": True, "resume_imported": False},
        )
        assert "ask_onboarding_wizard" in prompt
        assert "VERBATIM" in prompt
        assert "navigate" in prompt.lower()

    def test_delegation_rule_does_NOT_include_behavior_details(self):
        """Behavior rules (review loop, stop flow, reconcile policy) are
        in the sub-agent's prompt — not here. Main chat should be slim."""
        prompt = _build_system_prompt(
            SAMPLE_PROFILE,
            onboarding={"wizard_enabled": True, "resume_imported": True},
        )
        # Short delegation rule only — these details belong to the sub-agent.
        assert "Does this look right" not in prompt
        assert "COST-SAVING" not in prompt
        assert "wizard_enabled: yes" not in prompt  # no snapshot injection

    def test_onboarding_disabled_skips_delegation_rule(self):
        """Gate: if the user opted out, don't even tell the main agent
        about the delegation tool — prevents accidental invocation."""
        prompt = _build_system_prompt(
            SAMPLE_PROFILE,
            onboarding={"wizard_enabled": False, "resume_imported": False},
        )
        assert "ask_onboarding_wizard" not in prompt

    def test_all_steps_complete_skips_delegation_rule(self):
        prompt = _build_system_prompt(
            SAMPLE_PROFILE,
            onboarding={
                "wizard_enabled": True,
                "profile_basics": True,
                "resume_imported": True,
                "resume_reviewed": True,
                "first_job_post": True,
                "first_score": True,
                "first_cover_letter": True,
            },
        )
        assert "ask_onboarding_wizard" not in prompt

    def test_should_inject_aw_gate(self):
        assert _should_inject_aw(None) is False
        assert _should_inject_aw({}) is False
        assert _should_inject_aw({"wizard_enabled": False}) is False
        all_done = {"wizard_enabled": True, **{k: True for k in (
            "profile_basics", "resume_imported", "resume_reviewed",
            "first_job_post", "first_score", "first_cover_letter",
        )}}
        assert _should_inject_aw(all_done) is False
        assert _should_inject_aw(
            {"wizard_enabled": True, "resume_imported": False}
        ) is True
