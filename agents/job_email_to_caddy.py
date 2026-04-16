# Scrape a job URL and add it to Career Caddy.
#
# Email pipeline has been moved to career_caddy_automation.
# This file retains the URL scraping mode only.

import os
import uuid
import logfire
import logging
import json
import argparse
from agents.career_caddy_agent import add_job_post
from agents.job_extractor_agent import extract_job_from_content
from agents.agent_factory import get_model, get_model_name, get_agent, register_defaults
from lib.usage_reporter import report_usage
from pydantic_ai.usage import UsageLimits
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logfire.configure(service_name="job_url_to_caddy")
logfire.instrument_pydantic_ai()

register_defaults()


async def scrape_url_and_add_to_caddy(url: str, pipeline_run_id: str | None = None):
    """Scrape a job URL and add it to career caddy."""
    logger.info(f"Scraping job URL: {url}")
    run_id = pipeline_run_id or str(uuid.uuid4())
    return await _scrape_url_and_add_to_caddy(url, pipeline_run_id=run_id)


async def _scrape_url_and_add_to_caddy(url: str, pipeline_run_id: str | None = None):
    api_token = os.environ.get("CC_API_TOKEN", "")

    scraper_model = get_model("browser_scraper")
    scraper_agent = get_agent("browser_scraper")

    with logfire.span("browser.scrape_job", url=url, pipeline_run_id=pipeline_run_id):
        scrape_result = await scraper_agent.run(
            f"Scrape this URL and return all visible text: {url}",
            usage_limits=UsageLimits(request_limit=5),
        )

    if api_token:
        await report_usage(
            api_token=api_token,
            agent_name="browser_scraper",
            model_name=get_model_name(scraper_model),
            usage=scrape_result.usage(),
            trigger="pipeline",
            pipeline_run_id=pipeline_run_id,
        )

    raw_text = str(scrape_result.output or "")
    logger.info(f"Browser scrape output length: {len(raw_text)}")

    with logfire.span("browser.parse_job_data", url=url, pipeline_run_id=pipeline_run_id):
        job_data = await extract_job_from_content(
            raw_text, url=url, api_token=api_token, pipeline_run_id=pipeline_run_id
        )

    logger.info(f"Extracted job data: {job_data.title} at {job_data.company_name}")

    with logfire.span("caddy.add_job_post", title=job_data.title, company=job_data.company_name, url=url, pipeline_run_id=pipeline_run_id):
        caddy_result = await add_job_post(
            job_data, api_token=api_token, pipeline_run_id=pipeline_run_id
        )
    print("\n=== Added Job Post to Career Caddy ===")
    print(f"Title: {job_data.title}")
    print(f"Company: {job_data.company_name}")
    print(f"Location: {job_data.location}")
    print(f"URL: {job_data.url}")
    print("\nCareer Caddy Response:")
    print(json.dumps(caddy_result, indent=2))

    return caddy_result


async def main():
    parser = argparse.ArgumentParser(
        description="Scrape a job URL and add it to Career Caddy"
    )
    parser.add_argument(
        "url", type=str, help="Job posting URL to scrape"
    )
    args = parser.parse_args()

    pipeline_run_id = str(uuid.uuid4())
    await scrape_url_and_add_to_caddy(args.url, pipeline_run_id=pipeline_run_id)


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
