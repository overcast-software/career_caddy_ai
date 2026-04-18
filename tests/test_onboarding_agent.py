"""Tests for the Agent Wizard sub-agent's prompt + registration.

The sub-agent is invoked via agent delegation (agent-as-tool) by the main
chat. These tests exercise the prompt builder and registration — no LLM
calls, no HTTP. Behavior under LLM inference is covered manually via the
e2e walkthrough.
"""

import pytest

from agents.agent_factory import _AGENT_REGISTRY, register_defaults
from agents.onboarding_agent import (
    build_onboarding_prompt,
    register_onboarding_agent,
)
from lib.toolsets import SCOPES


SAMPLE_PROFILE = (
    "First name: Remi\n"
    "Last name: (blank)\n"
    "Email: remi@example.com\n"
    "Phone: (blank)"
)


class TestOnboardingScope:
    def test_scope_has_exactly_the_five_tools(self):
        assert SCOPES["onboarding"] == {
            "reconcile_onboarding",
            "edit_profile_onboarding",
            "show_resume",
            "edit_resume",
            "import_resume_from_url",
        }

    def test_scope_excludes_user_account_writes(self):
        """AW sub-agent must not have a tool that edits user fields."""
        assert "edit_user_profile" not in SCOPES["onboarding"]

    def test_scope_excludes_cover_letter_tools(self):
        """Cover letters aren't in the onboarding checklist; keep surface lean."""
        assert "show_cover_letter" not in SCOPES["onboarding"]
        assert "edit_cover_letter" not in SCOPES["onboarding"]

    def test_scope_excludes_general_crud(self):
        assert "get_job_posts" not in SCOPES["onboarding"]
        assert "create_company" not in SCOPES["onboarding"]


class TestRegistration:
    def test_onboarding_agent_registered_after_defaults(self):
        register_defaults()
        assert "onboarding" in _AGENT_REGISTRY

    def test_register_onboarding_agent_is_idempotent(self):
        register_onboarding_agent()
        register_onboarding_agent()  # second call shouldn't explode
        assert "onboarding" in _AGENT_REGISTRY


class TestBuildOnboardingPrompt:
    def test_renders_snapshot_block(self):
        prompt = build_onboarding_prompt(
            user_profile=SAMPLE_PROFILE,
            onboarding={"wizard_enabled": True, "resume_imported": False},
        )
        assert "Onboarding snapshot" in prompt
        assert "wizard_enabled: yes" in prompt
        assert "resume_imported: no" in prompt

    def test_names_the_next_incomplete_step(self):
        prompt = build_onboarding_prompt(
            user_profile=SAMPLE_PROFILE,
            onboarding={"wizard_enabled": True, "profile_basics": True},
        )
        assert "Next incomplete step:" in prompt
        # With profile_basics done, resume import comes next.
        assert "Import a resume" in prompt

    def test_includes_user_profile_with_blank_markers(self):
        """The sub-agent needs to see which fields are '(blank)' to name
        missing profile basics without guessing."""
        prompt = build_onboarding_prompt(
            user_profile=SAMPLE_PROFILE,
            onboarding={"wizard_enabled": True},
        )
        assert "Known user profile" in prompt
        assert "Last name: (blank)" in prompt
        assert "Email: remi@example.com" in prompt

    def test_adds_page_context_when_provided(self):
        prompt = build_onboarding_prompt(
            user_profile=SAMPLE_PROFILE,
            onboarding={"wizard_enabled": True},
            page_context={"route": "settings.profile.index", "url": "/settings/profile"},
        )
        assert "Current page" in prompt
        assert "/settings/profile" in prompt

    def test_omits_page_context_when_absent(self):
        prompt = build_onboarding_prompt(
            user_profile=SAMPLE_PROFILE,
            onboarding={"wizard_enabled": True},
        )
        assert "Current page" not in prompt

    def test_prompt_contains_off_limits_rule_for_user_fields(self):
        prompt = build_onboarding_prompt(
            user_profile=SAMPLE_PROFILE,
            onboarding={"wizard_enabled": True},
        )
        assert "OFF-LIMITS" in prompt
        assert "first_name" in prompt
        assert "last_name" in prompt

    def test_prompt_contains_review_loop(self):
        prompt = build_onboarding_prompt(
            user_profile=SAMPLE_PROFILE,
            onboarding={"wizard_enabled": True},
        )
        assert "show_resume" in prompt
        assert "Does this look right" in prompt
        assert "resume_reviewed" in prompt

    def test_prompt_contains_stop_helping_flow(self):
        prompt = build_onboarding_prompt(
            user_profile=SAMPLE_PROFILE,
            onboarding={"wizard_enabled": True},
        )
        assert "stop" in prompt.lower()
        assert "edit_profile_onboarding" in prompt
        assert "/settings/profile/edit" in prompt

    def test_prompt_enforces_one_step_at_a_time(self):
        prompt = build_onboarding_prompt(
            user_profile=SAMPLE_PROFILE,
            onboarding={"wizard_enabled": True},
        )
        assert "ONE step at a time" in prompt

    def test_prompt_forbids_refusing_to_share_user_own_data(self):
        """Regression guard — gpt-4o-mini loved to refuse personal-info
        questions. The sub-agent prompt explicitly defangs that."""
        prompt = build_onboarding_prompt(
            user_profile=SAMPLE_PROFILE,
            onboarding={"wizard_enabled": True},
        )
        assert "Never refuse" in prompt or "never refuse" in prompt
        assert "don't have access" in prompt.lower() or "do not say" in prompt.lower()


class TestDelegationToolsetFactory:
    """The delegation toolset is built on demand; smoke-check it constructs."""

    def test_factory_returns_toolset_with_ask_onboarding_wizard(self):
        pytest.importorskip("pydantic_ai")
        from lib.toolsets import onboarding_delegation_toolset

        ts = onboarding_delegation_toolset()
        assert "ask_onboarding_wizard" in ts.tools
        # This is the whole point — one tool, nothing else.
        assert len(ts.tools) == 1
