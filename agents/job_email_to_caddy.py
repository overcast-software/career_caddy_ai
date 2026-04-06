# job email to caddy
#
import os
import logfire
import logging
import json
import argparse
from pydantic import BaseModel
from lib.models.job_models import JobPostData
from agents.career_caddy_agent import add_job_post
from agents.job_extractor_agent import extract_job_from_content
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.usage import UsageLimits
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logfire.configure(service_name="job_email_to_caddy")
logfire.instrument_pydantic_ai()


class JobOpportunity(BaseModel):
    """A job opportunity found in emails"""

    url: str
    title: str


# Inline email-job agent — searches job_post-tagged emails, extracts URLs.
# Uses email_server.py as a stdio MCP subprocess so no external service is needed.
_email_mcp = MCPServerStdio("python", args=["mcp_servers/email_server.py"], env=os.environ.copy())

email_job_agent = Agent(
    "openai:gpt-4o-mini",
    name="email_job_agent",
    output_type=list[JobOpportunity],
    toolsets=[_email_mcp],
    system_prompt=(
        "Search for emails tagged 'job_post'. "
        "For each email found, read it and extract the job title and one primary job posting URL. "
        "Return a list of JobOpportunity objects. "
        "Only include URLs that point to actual job postings — skip unsubscribe links and tracking pixels."
    ),
)


async def scrape_url_and_add_to_caddy(url: str):
    """Scrape a job URL and add it to career caddy."""
    logger.info(f"Scraping job URL: {url}")
    return await _scrape_url_and_add_to_caddy(url)


async def _scrape_url_and_add_to_caddy(url: str):
    # Spawn browser_server as a stdio subprocess — no running docker service required.
    # scrape_page is a one-shot tool: create tab → navigate → snapshot → close.
    browser_mcp = MCPServerStdio(
        "python", args=["mcp_servers/browser_server.py"], env=os.environ.copy()
    )
    scraper_agent = Agent(
        "openai:gpt-4o-mini",
        name="browser_scraper",
        toolsets=[browser_mcp],
        system_prompt="Use the scrape_page tool to retrieve all visible text from the given URL. Return the raw text.",
    )

    with logfire.span("browser.scrape_job", url=url):
        scrape_result = await scraper_agent.run(
            f"Scrape this URL and return all visible text: {url}",
            usage_limits=UsageLimits(request_limit=5),
        )

    raw_text = str(scrape_result.output or "")
    logger.info(f"Browser scrape output length: {len(raw_text)}")

    # Parse raw text into structured JobPostData via the dedicated extractor agent
    with logfire.span("browser.parse_job_data", url=url):
        job_data = await extract_job_from_content(raw_text, url=url)

    logger.info(f"Extracted job data: {job_data.title} at {job_data.company_name}")

    # Add to career caddy
    with logfire.span("caddy.add_job_post", title=job_data.title, company=job_data.company_name, url=url):
        caddy_result = await add_job_post(job_data)
    print("\n=== Added Job Post to Career Caddy ===")
    print(f"Title: {job_data.title}")
    print(f"Company: {job_data.company_name}")
    print(f"Location: {job_data.location}")
    print(f"URL: {job_data.url}")
    print(f"\nCareer Caddy Response:")
    print(json.dumps(caddy_result, indent=2))

    return caddy_result


async def main():
    """Main workflow: Find job emails, extract data from URLs, add to career caddy."""
    parser = argparse.ArgumentParser(
        description="Job email to caddy - extract job data and add to career caddy"
    )
    parser.add_argument(
        "--url", type=str, help="Directly scrape a job URL and add to career caddy"
    )
    args = parser.parse_args()

    if args.url:
        # Direct URL scraping mode
        await scrape_url_and_add_to_caddy(args.url)
        return

    # Step 1: Find job opportunities in emails using the email_job_agent
    logger.info("Step 1: Searching for job opportunities in emails...")

    with logfire.span("pipeline.find_job_emails"):
        email_result = await email_job_agent.run(
            "Search for emails tagged 'job_post'. "
            "Extract the job title and URL for each job posting found."
        )

    print("\n=== Job Opportunities Found ===")

    # Access the structured data from the result
    jobs = email_result.output
    print(f"Found {len(jobs)} job opportunities")
    for job in jobs:
        print(f"Title: {job.title}")
        print(f"URL: {job.url}")
        print()

    print(f"\nUsage: {email_result.usage()}")

    # Step 2: Scrape and submit all jobs concurrently
    async def _process(job: JobOpportunity):
        with logfire.span("pipeline.process_job", title=job.title, url=job.url):
            logger.info(f"Processing: {job.title}")
            return await scrape_url_and_add_to_caddy(job.url)

    with logfire.span("pipeline.scrape_and_submit_all", job_count=len(jobs)):
        results = await asyncio.gather(*[_process(job) for job in jobs], return_exceptions=True)

    for job, result in zip(jobs, results):
        if isinstance(result, Exception):
            logger.error(f"Failed {job.title}: {result}")

    print("\n=== Workflow Complete ===")
    print(f"Processed {len(jobs)} job opportunities")


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
