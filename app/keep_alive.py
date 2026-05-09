"""
Keep-alive mechanism — prevents Render free tier from spinning down.

Pings the service's own health endpoint every 14 minutes via its public URL.
Requires the RENDER_EXTERNAL_HOSTNAME env var (auto-set by Render)
or a manually set SERVICE_URL env var.
"""

import os
import asyncio
import logging
import httpx

logger = logging.getLogger(__name__)


async def keep_alive_loop():
    """
    Background task that pings the service's own /health endpoint
    every 14 minutes to prevent Render free tier from spinning down.
    """
    # Determine the service's public URL
    service_url = os.getenv("SERVICE_URL", "")

    if not service_url:
        hostname = os.getenv("RENDER_EXTERNAL_HOSTNAME", "")
        if hostname:
            service_url = f"https://{hostname}"

    if not service_url:
        logger.info("ℹ️ No SERVICE_URL or RENDER_EXTERNAL_HOSTNAME set. Keep-alive disabled.")
        logger.info("   Set SERVICE_URL env var to enable (e.g., https://your-app.onrender.com)")
        return

    ping_url = f"{service_url}/health"
    interval = 14 * 60  # 14 minutes in seconds

    logger.info(f"🏓 Keep-alive started. Pinging {ping_url} every {interval // 60} minutes.")

    while True:
        await asyncio.sleep(interval)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(ping_url, timeout=30)
                logger.debug(f"🏓 Keep-alive ping: {response.status_code}")
        except Exception as e:
            logger.warning(f"🏓 Keep-alive ping failed: {e}")
