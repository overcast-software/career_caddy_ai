"""PydanticAI agent for extracting structured job post data from raw text/markdown."""

import os
import logging
from typing import Optional
from pydantic_ai import Agent
from lib.models.job_models import JobPostData

logger = logging.getLogger(__name__)

_EXTRACTION_MODEL = os.getenv("JOB_EXTRACTOR_MODEL", "openai:gpt-4o-mini")

_SYSTEM_PROMPT = """
You are a precise job posting data extractor. Given raw job posting text or markdown,
extract and return structured data. Be thorough — fill every field you can find.

Guidelines:
- title: The job title exactly as stated
- company_name: The hiring company (not a recruiter/job board). Use the hostname if unclear.
- description: Full job description including requirements, responsibilities, and qualifications
- location: City/state/country. Use "Remote" if fully remote.
- remote_ok: True if the role is remote or hybrid-remote
- salary_min/salary_max: Annual figures in integers. Convert hourly rates (×2080). Null if not stated.
- employment_type: "full-time", "part-time", "contract", "internship", or null
- link: The canonical URL of the posting (provided separately, do not invent one)
- posted_date: ISO format (YYYY-MM-DD) if a posting date is mentioned, else null
- company_description/company_website/company_industry/company_size/company_location:
  Fill from any "about the company" section in the content

Do not hallucinate data that is not present. Leave fields null if not mentioned.
"""

_extractor_agent: Optional[Agent] = None


def _get_extractor_agent() -> Agent:
    global _extractor_agent
    if _extractor_agent is None:
        _extractor_agent = Agent(
            model=_EXTRACTION_MODEL,
            name="job_extractor",
            output_type=JobPostData,
            system_prompt=_SYSTEM_PROMPT,
        )
    return _extractor_agent


async def extract_job_from_content(job_content: str, url: Optional[str] = None) -> JobPostData:
    """Extract structured JobPostData from raw job posting text/markdown.

    Args:
        job_content: Raw text or markdown of the job posting
        url: The source URL of the posting (used as the link field)

    Returns:
        Populated JobPostData instance
    """
    agent = _get_extractor_agent()
    prompt = job_content
    if url:
        prompt = f"Source URL: {url}\n\n{job_content}"
    logger.info("extract_job_from_content: running extraction content_len=%s url=%s", len(job_content), url)
    result = await agent.run(prompt)
    job_data: JobPostData = result.output
    if url and not job_data.link:
        job_data = job_data.model_copy(update={"link": url})
    logger.info("extract_job_from_content: extracted title=%r company=%r", job_data.title, job_data.company_name)
    return job_data
