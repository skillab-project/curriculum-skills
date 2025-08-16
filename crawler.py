import asyncio
import sys

async def run_apify_crawler(urls: list[str]):
    """
    Asynchronously runs the apify.js crawler as a subprocess.
    Streams stdout and stderr to the console in real-time.
    """
    if not urls:
        print("[CRAWLER_RUNNER] No URLs provided to start the crawl.", file=sys.stderr)
        return

    command = ['node', 'apify.js'] + urls
    print(f"[CRAWLER_RUNNER] Starting crawler with command: {' '.join(command)}")

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    async def read_stream(stream, prefix):
        while True:
            line = await stream.readline()
            if line:
                print(f"{prefix} {line.decode().strip()}", file=sys.stdout if prefix == "[CRAWLER_STDOUT]" else sys.stderr)
            else:
                break

    await asyncio.gather(
        read_stream(process.stdout, "[CRAWLER_STDOUT]"),
        read_stream(process.stderr, "[CRAWLER_STDERR]")
    )

    return_code = await process.wait()
    print(f"[CRAWLER_RUNNER] Crawler process finished with exit code: {return_code}")