"""
Pydantic-AI agent for interacting with the job hunting API.
Handles authentication and provides methods to interact with various endpoints.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports to work when called via stdio
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import os
import json
import logging
import logfire
from typing import Literal, Optional
from datetime import datetime
from fastmcp import FastMCP
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStdio
from lib.utils import scrubbing_callback
from lib.browser.credentials import Credentials  # type: ignore
from lib.models.career_caddy import (  # type: ignore
    funtimes,
    APICredentials,
    APIContext,
    JobPostCreate,
    APIResponse,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logfire.configure(
    service_name="career_caddy_server",
    scrubbing=False,
    console=False,
)


class CompanyData(BaseModel):
    """Data model for a company."""

    name: str = Field(..., min_length=1, max_length=200, description="Company name")
    description: Optional[str] = Field(None, description="Company description")
    website: Optional[str] = Field(None, description="Company website URL")
    industry: Optional[str] = Field(
        None, max_length=100, description="Company industry"
    )
    size: Optional[str] = Field(None, description="Company size")
    location: Optional[str] = Field(
        None, max_length=100, description="Company location"
    )


# Create screenshots directory if it doesn't exist
SCREENSHOTS_DIR = Path("screenshots")
SCREENSHOTS_DIR.mkdir(exist_ok=True)

# Load credentials
try:
    credentials = Credentials.load_credentials()
    logfire.info(f"Loaded credentials for {len(credentials.domains)} domains")
except FileNotFoundError as e:
    logfire.warn(f"No credentials file found: {e}")
    credentials = Credentials(domains={})
except Exception as e:
    logfire.error(f"Error loading credentials: {e}")
    credentials = Credentials(domains={})


server = FastMCP("career-caddy-server")

# Initialize screenshot agent only if OPENAI_API_KEY is available


@server.tool()
async def create_company(
    name: str,
    description: Optional[
        str
    ] = """
    This is the company for which the job post is for.
    Most resources are related to this, measures are taken to keep duplicate company's.
    """,
    website: Optional[str] = None,
    industry: Optional[str] = None,
    size: Optional[str] = None,
    location: Optional[str] = None,
) -> str:
    """Create a new company."""
    try:
        # Validate input data
        company_data = CompanyData(
            name=name,
            description=description,
            website=website,
            industry=industry,
            size=size,
            location=location,
        )

        credentials = APICredentials()
        client = httpx.AsyncClient()
        client.headers.update({"Authorization": f"Bearer {credentials.api_token}"})

        url = urljoin(credentials.base_url, "/api/v1/companies/")

        payload = {
            "data": {
                "type": "company",
                "attributes": company_data.model_dump(exclude_none=True),
            }
        }

        response = await client.post(url, json=payload)
        await client.aclose()

        if response.status_code in [200, 201]:
            result = APIResponse(
                success=True, data=response.json(), status_code=response.status_code
            )
        else:
            result = APIResponse(
                success=False,
                error=f"Failed to create company: {response.status_code} - {response.text}",
                status_code=response.status_code,
            )

        return json.dumps(result.model_dump(), indent=2)

    except ValueError as e:
        result = APIResponse(success=False, error=f"Validation error: {str(e)}")
        return json.dumps(result.model_dump(), indent=2)
    except Exception as e:
        result = APIResponse(success=False, error=f"Error creating company: {str(e)}")
        return json.dumps(result.model_dump(), indent=2)


async def create_job_post(
    title: str,
    company_id: int,
    description: Optional[str] = None,
    location: Optional[str] = None,
    salary_min: Optional[int] = None,
    salary_max: Optional[int] = None,
    employment_type: Optional[str] = None,
    remote_ok: bool = False,
    link: Optional[str] = None,
    posted_date: Optional[str] = None,
) -> str:
    """Internal helper — creates a job post with a known Career Caddy company_id.
    Not exposed as an MCP tool. Call create_job_post_with_company_check instead."""
    try:
        # Validate that the company_id actually exists before attempting to create
        company_check = json.loads(await get_companies(id=company_id))
        if not company_check.get("success"):
            result = APIResponse(
                success=False,
                error=(
                    f"company_id={company_id} does not exist in Career Caddy. "
                    "Use create_job_post_with_company_check (with company_name) instead — "
                    "it will look up or create the company automatically."
                ),
                status_code=400,
            )
            return json.dumps(result.model_dump(), indent=2)

        # Deduplicate by link before creating
        if link:
            existing = json.loads(await find_job_post_by_link(link))
            if existing.get("success"):
                posts = existing.get("data", {}).get("data", [])
                if posts:
                    existing_id = posts[0].get("id")
                    result = APIResponse(
                        success=False,
                        error=f"Duplicate: job post with this link already exists (id={existing_id})",
                        status_code=409,
                        data={"duplicate": True, "existing_id": existing_id},
                    )
                    return json.dumps(result.model_dump(), indent=2)

        # Validate input data
        job_post_data = JobPostCreate(
            title=title,
            description=description,
            company_id=company_id,
            location=location,
            salary_min=salary_min,
            salary_max=salary_max,
            employment_type=employment_type,
            remote_ok=remote_ok,
            link=link,
            posted_date=posted_date,
        )

        app_credentials = APICredentials()
        client = httpx.AsyncClient()
        client.headers.update({"Authorization": f"Bearer {app_credentials.api_token}"})

        url = urljoin(app_credentials.base_url, "/api/v1/job-posts/")

        # Prepare payload in JSON:API format
        attributes = job_post_data.model_dump(exclude={"company_id"}, exclude_none=True)

        payload = {
            "data": {
                "type": "job-post",
                "attributes": attributes,
                "relationships": {
                    "company": {"data": {"type": "company", "id": str(company_id)}}
                },
            }
        }

        response = await client.post(url, json=payload)
        await client.aclose()

        if response.status_code in [200, 201]:
            result = APIResponse(
                success=True, data=response.json(), status_code=response.status_code
            )
        else:
            # Truncate large error bodies (e.g. Django HTML debug pages) to avoid
            # blowing the LLM context window.
            error_text = (
                response.text[:500] if len(response.text) > 500 else response.text
            )
            result = APIResponse(
                success=False,
                error=f"Failed to create job post: {response.status_code} - {error_text}",
                status_code=response.status_code,
            )

        return json.dumps(result.model_dump(), indent=2)

    except ValueError as e:
        result = APIResponse(success=False, error=f"Validation error: {str(e)}")
        return json.dumps(result.model_dump(), indent=2)
    except Exception as e:
        result = APIResponse(success=False, error=f"Error creating job post: {str(e)}")
        return json.dumps(result.model_dump(), indent=2)


@server.tool()
async def find_job_post_by_link(link: str) -> str:
    """Find a job post by its original posting URL
    if the response is 200 and data is [] that means there is no job-post  for that given link
    ."""
    try:
        app_credentials = APICredentials()
        client = httpx.AsyncClient()
        client.headers.update({"Authorization": f"Bearer {app_credentials.api_token}"})

        url = urljoin(app_credentials.base_url, "/api/v1/job-posts/")
        response = await client.get(url, params={"filter[link]": link})
        await client.aclose()

        if response.status_code == 200:
            result = APIResponse(
                success=True, data=response.json(), status_code=response.status_code
            )
        else:
            result = APIResponse(
                success=False,
                error=f"Failed to find job post by link: {response.status_code}",
                status_code=response.status_code,
            )

        return json.dumps(result.model_dump(), indent=2)

    except Exception as e:
        result = APIResponse(
            success=False, error=f"Error finding job post by link: {str(e)}"
        )
        return json.dumps(result.model_dump(), indent=2)


@server.tool()
async def create_job_application(
    job_post_id: int,
    status: str = "applied",
    notes: Optional[str] = None,
    applied_at: Optional[str] = None,
) -> str:
    """Create a new job application linked to an existing job post.

    Use this when recording that you have applied (or intend to apply) to a job.
    job_post_id is the integer ID of the job post (e.g. from find_job_post_by_link).
    status should be one of: applied, interviewing, offered, rejected, withdrawn.
    applied_at: ISO date string (e.g. '2026-03-23'). Do NOT use 'applied_date'.
    """
    if job_post_id <= 0:
        result = APIResponse(
            success=False,
            error=f"Invalid job_post_id={job_post_id}. You must look up the real job post ID first (e.g. via find_job_post_by_link or get_job_posts).",
        )
        return json.dumps(result.model_dump(), indent=2)

    try:
        app_credentials = APICredentials()
        client = httpx.AsyncClient()
        client.headers.update({"Authorization": f"Bearer {app_credentials.api_token}"})

        url = urljoin(app_credentials.base_url, "/api/v1/job-applications/")

        attributes: dict = {"status": status}
        if notes is not None:
            attributes["notes"] = notes
        if applied_at is not None:
            attributes["applied_at"] = applied_at

        payload = {
            "data": {
                "type": "job-application",
                "attributes": attributes,
                "relationships": {
                    "job-post": {"data": {"type": "job-post", "id": str(job_post_id)}}
                },
            }
        }

        response = await client.post(url, json=payload)
        await client.aclose()

        if response.status_code in [200, 201]:
            result = APIResponse(
                success=True, data=response.json(), status_code=response.status_code
            )
        else:
            error_text = (
                response.text[:500] if len(response.text) > 500 else response.text
            )
            result = APIResponse(
                success=False,
                error=f"Failed to create job application: {response.status_code} - {error_text}",
                status_code=response.status_code,
            )

        return json.dumps(result.model_dump(), indent=2)

    except Exception as e:
        result = APIResponse(
            success=False, error=f"Error creating job application: {str(e)}"
        )
        return json.dumps(result.model_dump(), indent=2)


@server.tool()
async def get_applications_for_job_post(job_post_id: int) -> str:
    """Fetch all job applications linked to a specific job post.

    Use this to find the application ID when you need to update an existing application.
    Returns the list of applications (each with its own integer 'id') for the given job post.
    returning data: [] means that there is not job applications for that job post
    """
    try:
        app_credentials = APICredentials()
        client = httpx.AsyncClient()
        client.headers.update({"Authorization": f"Bearer {app_credentials.api_token}"})

        url = urljoin(
            app_credentials.base_url,
            f"/api/v1/job-posts/{job_post_id}/job-applications/",
        )
        response = await client.get(url)
        await client.aclose()

        if response.status_code == 200:
            result = APIResponse(
                success=True, data=response.json(), status_code=response.status_code
            )
        else:
            result = APIResponse(
                success=False,
                error=f"Failed to fetch applications for job post {job_post_id}: {response.status_code}",
                status_code=response.status_code,
            )

        return json.dumps(result.model_dump(), indent=2)

    except Exception as e:
        result = APIResponse(
            success=False, error=f"Error fetching applications for job post: {str(e)}"
        )
        return json.dumps(result.model_dump(), indent=2)


_APPLICATION_SORT_FIELDS = Literal[
    "id", "applied_at", "status", "job_post_id", "company_id", "notes"
]


@server.tool()
async def get_job_applications(
    id: Optional[int] = None,
    sort: Optional[_APPLICATION_SORT_FIELDS] = None,
    order: Optional[Literal["asc", "desc"]] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch job applications. Pass id to retrieve a single application; omit for a list.

    Args:
        id: Application ID. If provided, fetches that single application — all other params ignored.
        sort: Field to sort by. Must be one of: id, applied_at, status, job_post_id, company_id, notes.
              Use 'id' to get most recent. Do NOT use 'created_at' — it is not a valid field.
        order: 'asc' or 'desc'.
        page: Page number (1-based).
        per_page: Results per page.
    """
    try:
        app_credentials = APICredentials()
        client = httpx.AsyncClient()
        client.headers.update({"Authorization": f"Bearer {app_credentials.api_token}"})

        if id is not None:
            url = urljoin(app_credentials.base_url, f"/api/v1/job-applications/{id}/")
            response = await client.get(url)
        else:
            params = {}
            if sort is not None:
                params["sort"] = sort
            if order is not None:
                params["order"] = order
            if page is not None:
                params["page"] = page
            if per_page is not None:
                params["per_page"] = per_page
            url = urljoin(app_credentials.base_url, "/api/v1/job-applications/")
            response = await client.get(url, params=params)

        await client.aclose()

        if response.status_code == 200:
            result = APIResponse(
                success=True, data=response.json(), status_code=response.status_code
            )
        else:
            result = APIResponse(
                success=False,
                error=f"Failed to fetch job applications: {response.status_code}",
                status_code=response.status_code,
            )

        return json.dumps(result.model_dump(), indent=2)

    except Exception as e:
        result = APIResponse(
            success=False, error=f"Error fetching job applications: {str(e)}"
        )
        return json.dumps(result.model_dump(), indent=2)


@server.tool()
async def update_job_application(
    application_id: int,
    status: Optional[str] = None,
    notes: Optional[str] = None,
    applied_at: Optional[str] = None,
    company_id: Optional[int] = None,
) -> str:
    """Update a job application's status, notes, or company association.

    application_id is the application's own ID, NOT the job post ID or company ID.
    To associate with a company, pass company_id (use find_company_by_name to look it up).
    """
    try:
        app_credentials = APICredentials()
        client = httpx.AsyncClient()
        client.headers.update({"Authorization": f"Bearer {app_credentials.api_token}"})

        url = urljoin(
            app_credentials.base_url, f"/api/v1/job-applications/{application_id}/"
        )

        # Build attributes dict with only provided values
        attributes = {}
        if status is not None:
            attributes["status"] = status
        if notes is not None:
            attributes["notes"] = notes
        if applied_at is not None:
            attributes["applied_at"] = applied_at

        if not attributes and company_id is None:
            result = APIResponse(
                success=False,
                error="No fields provided to update",
            )
            return json.dumps(result.model_dump(), indent=2)

        payload: dict = {
            "data": {
                "type": "job-application",
                "id": str(application_id),
                "attributes": attributes,
            }
        }

        if company_id is not None:
            payload["data"]["relationships"] = {
                "company": {"data": {"type": "company", "id": str(company_id)}}
            }

        response = await client.patch(url, json=payload)
        await client.aclose()

        if response.status_code == 200:
            result = APIResponse(
                success=True, data=response.json(), status_code=response.status_code
            )
        else:
            result = APIResponse(
                success=False,
                error=f"Failed to update job application: {response.status_code} - {response.text}",
                status_code=response.status_code,
            )

        return json.dumps(result.model_dump(), indent=2)

    except Exception as e:
        result = APIResponse(
            success=False, error=f"Error updating job application: {str(e)}"
        )
        return json.dumps(result.model_dump(), indent=2)


@server.tool()
async def get_companies(id: Optional[int] = None) -> str:
    """Fetch companies. Pass id to retrieve a single company; omit for the full list.

    Args:
        id: Company ID. If provided, fetches that single company.
    """
    try:
        app_credentials = APICredentials()
        client = httpx.AsyncClient()
        client.headers.update({"Authorization": f"Bearer {app_credentials.api_token}"})

        if id is not None:
            url = urljoin(app_credentials.base_url, f"/api/v1/companies/{id}/")
        else:
            url = urljoin(app_credentials.base_url, "/api/v1/companies/")

        response = await client.get(url)
        await client.aclose()

        if response.status_code == 200:
            result = APIResponse(
                success=True, data=response.json(), status_code=response.status_code
            )
        else:
            result = APIResponse(
                success=False,
                error=f"Failed to fetch companies: {response.status_code}",
                status_code=response.status_code,
            )

        return json.dumps(result.model_dump(), indent=2)

    except Exception as e:
        result = APIResponse(success=False, error=f"Error fetching companies: {str(e)}")
        return json.dumps(result.model_dump(), indent=2)


@server.tool()
async def find_company_by_name(company_name: str) -> str:
    """Find a company by name (case-insensitive search)."""
    try:
        app_credentials = APICredentials()
        client = httpx.AsyncClient()
        client.headers.update({"Authorization": f"Bearer {app_credentials.api_token}"})

        url = urljoin(app_credentials.base_url, "/api/v1/companies/")
        response = await client.get(url, params={"filter[query]": company_name})
        await client.aclose()

        if response.status_code == 200:
            companies_data = response.json()
            companies = companies_data.get("data", [])

            if companies:
                result = APIResponse(
                    success=True,
                    data={"companies": companies, "count": len(companies)},
                    status_code=200,
                )
            else:
                result = APIResponse(
                    success=False,
                    error=f"No companies found matching '{company_name}'",
                    status_code=404,
                )
        else:
            result = APIResponse(
                success=False,
                error=f"Failed to fetch companies: {response.status_code}",
                status_code=response.status_code,
            )

        return json.dumps(result.model_dump(), indent=2)

    except Exception as e:
        result = APIResponse(
            success=False, error=f"Error searching for company: {str(e)}"
        )
        return json.dumps(result.model_dump(), indent=2)


@server.tool()
async def create_job_post_with_company_check(
    title: str,
    company_name: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    salary_min: Optional[int] = None,
    salary_max: Optional[int] = None,
    employment_type: Optional[str] = None,
    remote_ok: bool = False,
    url: Optional[str] = None,
    link: Optional[str] = None,
    posted_date: Optional[str] = None,
    # Company creation fields (if company doesn't exist)
    company_description: Optional[str] = None,
    company_website: Optional[str] = None,
    company_industry: Optional[str] = None,
    company_size: Optional[str] = None,
    company_location: Optional[str] = None,
) -> str:
    """Create a job post, creating the company first if it doesn't exist."""
    url = url or link  # accept 'link' as an alias for 'url'
    if not company_name:
        result = APIResponse(
            success=False, error="company_name is required to create a job post"
        )
        return json.dumps(result.model_dump(), indent=2)
    try:
        # Check for duplicate job post by URL before doing anything else
        if url:
            existing = json.loads(await find_job_post_by_link(url))
            if existing.get("success"):
                posts = existing.get("data", {}).get("data", [])
                if posts:
                    existing_id = posts[0].get("id")
                    result = APIResponse(
                        success=False,
                        error=f"Duplicate: job post with this link already exists (id={existing_id})",
                        status_code=409,
                        data={"duplicate": True, "existing_id": existing_id},
                    )
                    return json.dumps(result.model_dump(), indent=2)

        # Search for existing company
        company_search_result = await find_company_by_name(company_name)
        company_search_data = json.loads(company_search_result)

        company_id = None

        if company_search_data.get("success"):
            # Company found, use the first match
            companies = company_search_data.get("data", {}).get("companies", [])
            if companies:
                company_id = int(companies[0].get("id"))
                logfire.info(
                    f"Found existing company: {company_name} (ID: {company_id})"
                )

        # If company not found, create it
        if company_id is None:
            logfire.info(f"Company '{company_name}' not found, creating new company...")
            create_company_result = await create_company(
                name=company_name,
                description=company_description,
                website=company_website,
                industry=company_industry,
                size=company_size,
                location=company_location,
            )

            create_company_data = json.loads(create_company_result)
            if create_company_data.get("success"):
                company_data = create_company_data.get("data", {}).get("data", {})
                company_id = int(company_data.get("id"))
                logfire.info(f"Created new company: {company_name} (ID: {company_id})")
            else:
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Failed to create company: {create_company_data.get('error')}",
                    },
                    indent=2,
                )

        # Now create the job post with the company_id
        job_post_result = await create_job_post(
            title=title,
            description=description,
            company_id=company_id,
            location=location,
            salary_min=salary_min,
            salary_max=salary_max,
            employment_type=employment_type,
            remote_ok=remote_ok,
            link=url,
            posted_date=posted_date,
        )

        return job_post_result

    except Exception as e:
        result = APIResponse(
            success=False, error=f"Error creating job post with company check: {str(e)}"
        )
        return json.dumps(result.model_dump(), indent=2)


@server.tool()
async def search_job_posts(
    query: Optional[str] = None,
    title: Optional[str] = None,
    company: Optional[str] = None,
    company_id: Optional[int] = None,
    sort: Optional[str] = None,
    page_size: Optional[int] = None,
) -> str:
    """Search job posts by keyword, title, company name, or company ID.

    Args:
        query: Free-text search across title, description, and company name (OR match).
        title: Filter by title only (case-insensitive contains).
        company: Filter by company name only (case-insensitive contains).
        company_id: Filter by exact company ID.
        sort: Sort field, e.g. '-created_at' (prefix '-' for descending).
        page_size: Number of results to return.
    """
    try:
        app_credentials = APICredentials()
        client = httpx.AsyncClient()
        client.headers.update({"Authorization": f"Bearer {app_credentials.api_token}"})

        params = {}
        if query is not None:
            params["filter[query]"] = query
        if title is not None:
            params["filter[title]"] = title
        if company is not None:
            params["filter[company]"] = company
        if company_id is not None:
            params["filter[company_id]"] = company_id
        if sort is not None:
            params["sort"] = sort
        if page_size is not None:
            params["page[size]"] = page_size

        url = urljoin(app_credentials.base_url, "/api/v1/job-posts/")
        response = await client.get(url, params=params)
        await client.aclose()

        if response.status_code == 200:
            result = APIResponse(
                success=True, data=response.json(), status_code=response.status_code
            )
        else:
            result = APIResponse(
                success=False,
                error=f"Failed to search job posts: {response.status_code}",
                status_code=response.status_code,
            )

        return json.dumps(result.model_dump(), indent=2)

    except Exception as e:
        result = APIResponse(success=False, error=f"Error searching job posts: {str(e)}")
        return json.dumps(result.model_dump(), indent=2)


@server.tool()
async def search_companies(
    query: Optional[str] = None,
    page_size: Optional[int] = None,
) -> str:
    """Search companies by name or display_name (case-insensitive OR match).

    Args:
        query: Free-text search across name and display_name.
        page_size: Number of results to return.
    """
    try:
        app_credentials = APICredentials()
        client = httpx.AsyncClient()
        client.headers.update({"Authorization": f"Bearer {app_credentials.api_token}"})

        params = {}
        if query is not None:
            params["filter[query]"] = query
        if page_size is not None:
            params["page[size]"] = page_size

        url = urljoin(app_credentials.base_url, "/api/v1/companies/")
        response = await client.get(url, params=params)
        await client.aclose()

        if response.status_code == 200:
            result = APIResponse(
                success=True, data=response.json(), status_code=response.status_code
            )
        else:
            result = APIResponse(
                success=False,
                error=f"Failed to search companies: {response.status_code}",
                status_code=response.status_code,
            )

        return json.dumps(result.model_dump(), indent=2)

    except Exception as e:
        result = APIResponse(success=False, error=f"Error searching companies: {str(e)}")
        return json.dumps(result.model_dump(), indent=2)


@server.tool()
async def get_job_posts(
    id: Optional[int] = None,
    sort: Optional[str] = None,
    order: Optional[str] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch job posts. Pass id to retrieve a single post; omit for a paginated list.

    Args:
        id: Job post ID. If provided, fetches that single post — all other params ignored.
        sort: Field to sort by (e.g. 'created_at', 'posted_date', 'title').
        order: 'asc' or 'desc'.
        page: 1-based page number.
        per_page: Results per page.
    """
    try:
        app_credentials = APICredentials()
        client = httpx.AsyncClient()
        client.headers.update({"Authorization": f"Bearer {app_credentials.api_token}"})

        if id is not None:
            url = urljoin(app_credentials.base_url, f"/api/v1/job-posts/{id}/")
            response = await client.get(url)
        else:
            params = {}
            if sort is not None:
                params["sort"] = sort
            if order is not None:
                params["order"] = order
            if page is not None:
                params["page"] = page
            if per_page is not None:
                params["per_page"] = per_page
            url = urljoin(app_credentials.base_url, "/api/v1/job-posts/")
            response = await client.get(url, params=params)

        await client.aclose()

        if response.status_code == 200:
            result = APIResponse(
                success=True, data=response.json(), status_code=response.status_code
            )
        else:
            result = APIResponse(
                success=False,
                error=f"Failed to fetch job posts: {response.status_code}",
                status_code=response.status_code,
            )

        return json.dumps(result.model_dump(), indent=2)

    except Exception as e:
        result = APIResponse(success=False, error=f"Error fetching job posts: {str(e)}")
        return json.dumps(result.model_dump(), indent=2)


@server.tool()
async def update_job_post(
    job_post_id: int,
    title: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    salary_min: Optional[int] = None,
    salary_max: Optional[int] = None,
    employment_type: Optional[str] = None,
    remote_ok: Optional[bool] = None,
    link: Optional[str] = None,
    posted_date: Optional[str] = None,
    company_id: Optional[int] = None,
) -> str:
    """Update an existing job post's attributes or company relationship.

    To change the company associated with a job post, pass company_id (the integer ID).
    Use find_company_by_name to look up the company ID first.
    All fields are optional — only provided fields are updated.
    """
    try:
        app_credentials = APICredentials()
        client = httpx.AsyncClient()
        client.headers.update({"Authorization": f"Bearer {app_credentials.api_token}"})

        url = urljoin(app_credentials.base_url, f"/api/v1/job-posts/{job_post_id}/")

        attributes = {}
        if title is not None:
            attributes["title"] = title
        if description is not None:
            attributes["description"] = description
        if location is not None:
            attributes["location"] = location
        if salary_min is not None:
            attributes["salary_min"] = salary_min
        if salary_max is not None:
            attributes["salary_max"] = salary_max
        if employment_type is not None:
            attributes["employment_type"] = employment_type
        if remote_ok is not None:
            attributes["remote_ok"] = remote_ok
        if link is not None:
            attributes["link"] = link
        if posted_date is not None:
            attributes["posted_date"] = posted_date

        if not attributes and company_id is None:
            result = APIResponse(success=False, error="No fields provided to update")
            return json.dumps(result.model_dump(), indent=2)

        payload: dict = {
            "data": {
                "type": "job-post",
                "id": str(job_post_id),
                "attributes": attributes,
            }
        }

        if company_id is not None:
            payload["data"]["relationships"] = {
                "company": {"data": {"type": "company", "id": str(company_id)}}
            }

        response = await client.patch(url, json=payload)
        await client.aclose()

        if response.status_code == 200:
            result = APIResponse(
                success=True, data=response.json(), status_code=response.status_code
            )
        else:
            result = APIResponse(
                success=False,
                error=f"Failed to update job post: {response.status_code} - {response.text}",
                status_code=response.status_code,
            )

        return json.dumps(result.model_dump(), indent=2)

    except Exception as e:
        result = APIResponse(success=False, error=f"Error updating job post: {str(e)}")
        return json.dumps(result.model_dump(), indent=2)


@server.tool(
    description="Fetch the user's personal career profile: resume, skills, experience, education, certifications, and cover letters. Use this to score jobs or answer questions about the user's background. This is NOT job posts."
)
async def get_career_data() -> str:
    app_credentials = APICredentials()
    client = httpx.AsyncClient()
    client.headers.update({"Authorization": f"Bearer {app_credentials.api_token}"})
    url = urljoin(app_credentials.base_url, "/api/v1/career-data")
    print("*" * 88)
    response = await client.get(url)
    print(response)
    print("*" * 88)
    await client.aclose()

    if response.status_code == 200:
        result = APIResponse(
            success=True, data=response.json(), status_code=response.status_code
        )
    else:
        result = APIResponse(
            success=False,
            error=f"Failed to fetch career data: {response.status_code}",
            status_code=response.status_code,
        )

    return json.dumps(result.model_dump(), indent=2)


async def cleanup_client(ctx: APIContext):
    """Clean up the HTTP client."""
    if ctx.client:
        await ctx.client.aclose()


if __name__ == "__main__":
    import sys

    logfire.info("Starting Career Caddy MCP Server...")
    try:
        server.run()
    except Exception as e:
        logfire.error(f"Failed to start MCP server: {e}")
        sys.exit(1)
