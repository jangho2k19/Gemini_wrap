#!/usr/bin/env python3
"""
Safe sequential image generator for Gemini Web API.
Designed to prevent abuse flags, rate limits, and IP blocks by enforcing:
  1. Strict one-by-one sequential generation (no concurrent calls).
  2. Randomized human-like pacing (15-25s delay + jitter between requests).
  3. Automatic backoff upon encountering rate limits (1037/1060/1013).
  4. Full cookie persistence & rotation.
  5. Real-time account abuse & quota monitoring.
  6. Resume capability via a progress manifest.
"""

import argparse
import asyncio
import contextlib
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add src to sys.path so we can run directly
if (_src := str(Path(__file__).resolve().parent / "src")) not in sys.path:
    sys.path.insert(0, _src)

from gemini_webapi import GeminiClient, logger, set_log_level
from gemini_webapi.exceptions import APIError, AuthError, TemporarilyBlockedError
from cli import _load_cookies_with_meta, _persist_cookies


def build_client(cookies_json_path: Path):
    json_cookies, _ = _load_cookies_with_meta(str(cookies_json_path))
    psid = json_cookies.get("__Secure-1PSID")
    psidts = json_cookies.get("__Secure-1PSIDTS")

    if not psid:
        raise SystemExit("Missing __Secure-1PSID in cookies file.")

    extra = {
        k: v for k, v in json_cookies.items() if k not in {"__Secure-1PSID", "__Secure-1PSIDTS"}
    }

    client = GeminiClient(
        secure_1psid=psid,
        secure_1psidts=psidts or "",
        cookies=extra or None,
    )
    return client, json_cookies


async def generate_images_safely(
    prompts: list[str],
    output_dir: Path,
    cookies_file: Path,
    min_delay: float = 15.0,
    max_delay: float = 25.0,
    resume: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = output_dir / "manifest.json"

    completed = {}
    if resume and manifest_file.exists():
        try:
            completed = json.loads(manifest_file.read_text(encoding="utf-8"))
            logger.info(f"Loaded existing manifest with {len(completed)} completed items.")
        except Exception:
            completed = {}

    client, json_cookies = build_client(cookies_file)

    logger.info("Initializing GeminiClient session...")
    await client.init(timeout=300, auto_refresh=True)

    if client.account_status.name != "AVAILABLE":
        logger.warning(f"Account status: {client.account_status.name} - {client.account_status.description}")
        if client.account_status.name == "UNAUTHENTICATED":
            await client.close()
            raise SystemExit("Session is unauthenticated. Please re-export cookies.")
    else:
        logger.success(f"Account authenticated. Abuse status: {client.abuse_status or 'clean'}")

    try:
        total = len(prompts)
        for idx, prompt in enumerate(prompts, start=1):
            if prompt in completed and completed[prompt].get("status") == "success":
                logger.info(f"[{idx}/{total}] Skipping already completed: {prompt[:45]}...")
                continue

            logger.info(f"\n==========================================")
            logger.info(f"[{idx}/{total}] Generating: \"{prompt}\"")
            logger.info(f"==========================================")

            retries = 3
            success = False

            while retries > 0 and not success:
                try:
                    response = await client.generate_content(
                        f"Generate an image of: {prompt}",
                        temporary=True
                    )

                    if not response.images:
                        if "Are you signed in?" in response.text and retries > 1:
                            logger.warning("Sign-in refusal detected. Attempting to rotate and refresh cookie...")
                            try:
                                from gemini_webapi.utils.rotate_1psidts import rotate_1psidts
                                new_ts = await rotate_1psidts(client._live_client, base_psid=client._base_psid, verbose=True)
                                if new_ts:
                                    logger.success("Rotated __Secure-1PSIDTS successfully! Retrying prompt...")
                                    retries -= 1
                                    continue
                            except Exception as rot_e:
                                logger.warning(f"Rotation attempt failed: {rot_e}")

                        logger.warning(f"No image was generated for: \"{prompt}\"")
                        logger.info(f"Gemini reply: {response.text[:200]}")
                        completed[prompt] = {
                            "status": "no_image",
                            "timestamp": datetime.now(tz=UTC).isoformat(),
                            "reply": response.text,
                        }
                        break

                    saved_paths = []
                    for img_idx, img in enumerate(response.images):
                        filename = f"img_{idx:03d}_{img_idx}_{int(datetime.now().timestamp())}.png"
                        saved_path = await img.save(
                            path=str(output_dir),
                            filename=filename,
                            verbose=True
                        )
                        saved_paths.append(str(saved_path))
                        logger.success(f"Image successfully saved: {saved_path}")

                    completed[prompt] = {
                        "status": "success",
                        "timestamp": datetime.now(tz=UTC).isoformat(),
                        "images": saved_paths,
                        "chat_id": response.cid,
                    }
                    success = True

                except TemporarilyBlockedError as e:
                    logger.warning(f"Rate limited (1060): {e}")
                    wait_time = 90 + random.uniform(10, 30)
                    logger.info(f"Cooling down for {wait_time:.1f}s before retry...")
                    await asyncio.sleep(wait_time)
                    retries -= 1

                except APIError as e:
                    if ("401" in str(e) or "unauthenticated" in str(e).lower()) and retries > 1:
                        logger.warning(f"Authentication failure ({e}). Attempting cookie refresh...")
                        try:
                            from gemini_webapi.utils.rotate_1psidts import rotate_1psidts
                            await rotate_1psidts(client._live_client, base_psid=client._base_psid, verbose=True)
                            await client.init(timeout=300, auto_refresh=True)
                            retries -= 1
                            continue
                        except Exception:
                            pass
                    logger.warning(f"API Error ({e}): backing off for 30s...")
                    await asyncio.sleep(30 + random.uniform(5, 15))
                    retries -= 1

                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    completed[prompt] = {
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now(tz=UTC).isoformat()
                    }
                    break

            # Persist progress manifest
            manifest_file.write_text(json.dumps(completed, indent=2), encoding="utf-8")

            # Persist updated cookies back to disk
            _persist_cookies(str(cookies_file), json_cookies, client.cookies)

            # Human-like randomized delay before the next prompt
            if idx < total:
                delay = random.uniform(min_delay, max_delay)
                logger.info(f"Pacing: waiting {delay:.1f}s before next image to prevent abuse flags...\n")
                await asyncio.sleep(delay)

    finally:
        _persist_cookies(str(cookies_file), json_cookies, client.cookies)
        await client.close()
        logger.info("Completed! Session closed cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Safely generate images sequentially with Gemini API")
    parser.add_argument("--cookies", default="cookies.json", help="Path to cookies JSON file")
    parser.add_argument("--output", default="generated_images", help="Output directory")
    parser.add_argument("--min-delay", type=float, default=15.0, help="Minimum delay between images (seconds)")
    parser.add_argument("--max-delay", type=float, default=25.0, help="Maximum delay between images (seconds)")
    parser.add_argument("--prompts-file", help="Text file containing one prompt per line")
    parser.add_argument("prompts", nargs="*", help="Image prompts directly from command line")

    args = parser.parse_args()

    all_prompts = []
    if args.prompts_file:
        p_path = Path(args.prompts_file)
        if p_path.exists():
            all_prompts.extend([line.strip() for line in p_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    if args.prompts:
        all_prompts.extend(args.prompts)

    if not all_prompts:
        print("No prompts provided. Provide prompts as arguments or via --prompts-file.")
        print('Example: python safe_image_generator.py "A golden retriever playing in autumn leaves" "A cybernetic city skyline at night"')
        return 1

    cookies_file = Path(args.cookies)
    if not cookies_file.exists():
        print(f"Cookies file not found: {cookies_file}")
        return 1

    asyncio.run(
        generate_images_safely(
            prompts=all_prompts,
            output_dir=Path(args.output),
            cookies_file=cookies_file,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
