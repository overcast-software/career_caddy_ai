"""Tests for the Agent Wizard tools registered in lib.api_tools."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from lib.api_tools import (
    ApiClient,
    edit_cover_letter,
    edit_profile_onboarding,
    edit_resume,
    reconcile_onboarding,
    show_cover_letter,
    show_resume,
)


def _ok(data, status=200):
    return json.dumps(
        {"success": True, "data": data, "error": None, "status_code": status},
        indent=2,
    )


@pytest.fixture
def api():
    return ApiClient("http://api:8000", "jh_test")


class TestShowResume:
    @pytest.mark.asyncio
    async def test_returns_raw_markdown(self, api):
        expected = "## Jane Doe\n## SRE Resume\n"
        with patch.object(ApiClient, "get_text", new=AsyncMock(return_value=expected)) as mock:
            result = await show_resume(api, resume_id=42)
        assert result == expected
        mock.assert_awaited_once_with("/api/v1/resumes/42/markdown/")


class TestEditResume:
    @pytest.mark.asyncio
    async def test_patches_resume_with_supplied_fields(self, api):
        with patch.object(
            ApiClient,
            "patch",
            new=AsyncMock(return_value=_ok({"data": {"type": "resume", "id": "42"}})),
        ) as mock:
            await edit_resume(api, resume_id=42, title="Senior SRE", favorite=True)

        mock.assert_awaited_once()
        path, payload = mock.await_args.args
        assert path == "/api/v1/resumes/42/"
        attrs = payload["data"]["attributes"]
        assert attrs == {"title": "Senior SRE", "favorite": True}

    @pytest.mark.asyncio
    async def test_rejects_empty_update(self, api):
        result = json.loads(await edit_resume(api, resume_id=1))
        assert result["success"] is False
        assert "at least one field" in result["error"]


class TestShowCoverLetter:
    @pytest.mark.asyncio
    async def test_returns_raw_markdown(self, api):
        expected = "# Cover Letter\nCreated: 2026-04-18\n\nDear hiring manager,"
        with patch.object(ApiClient, "get_text", new=AsyncMock(return_value=expected)) as mock:
            result = await show_cover_letter(api, cover_letter_id=7)
        assert result == expected
        mock.assert_awaited_once_with("/api/v1/cover-letters/7/markdown/")


class TestEditCoverLetter:
    @pytest.mark.asyncio
    async def test_patches_cover_letter(self, api):
        with patch.object(
            ApiClient,
            "patch",
            new=AsyncMock(return_value=_ok({"data": {"type": "cover-letter", "id": "3"}})),
        ) as mock:
            await edit_cover_letter(
                api, cover_letter_id=3, content="Revised body.", favorite=True
            )

        path, payload = mock.await_args.args
        assert path == "/api/v1/cover-letters/3/"
        assert payload["data"]["id"] == "3"
        assert payload["data"]["attributes"] == {
            "content": "Revised body.",
            "favorite": True,
        }

    @pytest.mark.asyncio
    async def test_rejects_empty_update(self, api):
        result = json.loads(await edit_cover_letter(api, cover_letter_id=3))
        assert result["success"] is False


class TestReconcileOnboarding:
    @pytest.mark.asyncio
    async def test_posts_empty_body_to_reconcile_endpoint(self, api):
        with patch.object(
            ApiClient,
            "post",
            new=AsyncMock(return_value=_ok({"wizard_enabled": True, "resume_imported": True})),
        ) as mock:
            result = json.loads(await reconcile_onboarding(api))
        mock.assert_awaited_once()
        path, payload = mock.await_args.args
        assert path == "/api/v1/onboarding/reconcile/"
        assert payload == {}
        assert result["success"] is True
        assert result["data"]["resume_imported"] is True


class TestEditProfileOnboarding:
    @pytest.mark.asyncio
    async def test_resolves_me_then_patches_user(self, api):
        me_response = _ok(
            {"data": {"type": "user", "id": "11", "attributes": {"onboarding": {}}}}
        )
        patch_response = _ok(
            {
                "data": {
                    "type": "user",
                    "id": "11",
                    "attributes": {"onboarding": {"resume_reviewed": True}},
                }
            }
        )
        with patch.object(ApiClient, "get", new=AsyncMock(return_value=me_response)) as mock_get, \
             patch.object(ApiClient, "patch", new=AsyncMock(return_value=patch_response)) as mock_patch:
            result = json.loads(
                await edit_profile_onboarding(api, {"resume_reviewed": True})
            )

        mock_get.assert_awaited_once_with("/api/v1/me/")
        path, payload = mock_patch.await_args.args
        assert path == "/api/v1/users/11/"
        assert payload["data"]["attributes"]["onboarding"] == {"resume_reviewed": True}
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_rejects_empty_patch(self, api):
        result = json.loads(await edit_profile_onboarding(api, {}))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_rejects_non_dict_patch(self, api):
        result = json.loads(await edit_profile_onboarding(api, None))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_propagates_me_lookup_failure(self, api):
        failure = json.dumps(
            {
                "success": False,
                "data": None,
                "error": "401 - unauthorized",
                "status_code": 401,
            },
            indent=2,
        )
        with patch.object(ApiClient, "get", new=AsyncMock(return_value=failure)), \
             patch.object(ApiClient, "patch", new=AsyncMock()) as mock_patch:
            result = json.loads(await edit_profile_onboarding(api, {"resume_reviewed": True}))
        assert result["success"] is False
        mock_patch.assert_not_called()
