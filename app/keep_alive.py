"""
Health-check ping — verifies the service is live on a configurable schedule.

By default, pings every 12 hours (43200 seconds). Override with the
KEEP_ALIVE_INTERVAL_HOURS env var. Set to 0 to disable entirely.

On Render free tier, the service WILL spin down after 15 min of inactivity
regardless. This ping just ensures the service can still cold-start correctly
when it's woken up by the cron. It is NOT meant to keep it warm 24/7
(doing so burns through free-tier compute hours).
"""

import os
import asyncio
import logging
import httpx

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_HOURS = 12  # Ping every 12 hours


async def keep_alive_loop():
    """
    Background task that periodically pings the service's own /health
    endpoint to verify it is operational.
    """
    # Determine the service's public URL
    service_url = os.getenv("SERVICE_URL", "")

    if not service_url:
        hostname = os.getenv("RENDER_EXTERNAL_HOSTNAME", "")
        if hostname:
            service_url = f"https://{hostname}"

    if not service_url:
        logger.info("ℹ️  No SERVICE_URL or RENDER_EXTERNAL_HOSTNAME set. Health-check ping disabled.")
        logger.info("   Set SERVICE_URL env var to enable (e.g., https://your-app.onrender.com)")
        return

    # Configurable interval (hours) — default 12h, set 0 to disable
    try:
        interval_hours = float(os.getenv("KEEP_ALIVE_INTERVAL_HOURS", str(DEFAULT_INTERVAL_HOURS)))
    except ValueError:
        interval_hours = DEFAULT_INTERVAL_HOURS

    if interval_hours <= 0:
        logger.info("ℹ️  KEEP_ALIVE_INTERVAL_HOURS=0 — Health-check ping disabled.")
        return

    interval_seconds = int(interval_hours * 3600)
    ping_url = f"{service_url}/health"

    logger.info(f"🏓 Health-check ping scheduled: {ping_url} every {interval_hours}h ({interval_seconds}s)")

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(ping_url, timeout=60)
                logger.info(f"🏓 Health-check ping: {response.status_code}")
        except Exception as e:
            logger.warning(f"🏓 Health-check ping failed: {e}")
