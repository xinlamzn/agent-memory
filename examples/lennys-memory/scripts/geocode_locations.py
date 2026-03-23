#!/usr/bin/env python3
"""Geocode Location entities in the memory graph.

This script adds latitude/longitude coordinates to Location entities that
don't have them yet. It uses Nominatim (OpenStreetMap) by default, which
is free but rate-limited to 1 request per second.

Usage:
    python geocode_locations.py [options]

Options:
    --provider nominatim|google    Geocoding provider (default: nominatim)
    --api-key KEY                  API key (required for Google)
    --batch-size N                 Batch size for processing (default: 50)
    --skip-existing                Skip locations that already have coordinates
    -v, --verbose                  Show detailed progress
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add backend src to path for imports
backend_src = Path(__file__).parent.parent / "backend" / "src"
sys.path.insert(0, str(backend_src))

from dotenv import load_dotenv

# Load environment from backend
load_dotenv(Path(__file__).parent.parent / "backend" / ".env")


async def main():
    parser = argparse.ArgumentParser(description="Geocode Location entities in the memory graph")
    parser.add_argument(
        "--provider",
        choices=["nominatim", "google"],
        default="nominatim",
        help="Geocoding provider (default: nominatim)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for geocoding provider (required for Google)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip locations that already have coordinates (default: True)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )
    args = parser.parse_args()

    # Validate Google API key
    if args.provider == "google" and not args.api_key:
        api_key = os.getenv("GOOGLE_GEOCODING_API_KEY")
        if not api_key:
            print("Error: Google geocoding requires an API key.")
            print("Set GOOGLE_GEOCODING_API_KEY environment variable or use --api-key")
            sys.exit(1)
        args.api_key = api_key

    # Import after path setup
    from pydantic import SecretStr

    from neo4j_agent_memory import (
        EmbeddingConfig,
        EmbeddingProvider,
        GeocodingConfig,
        GeocodingProvider,
        MemoryClient,
        MemorySettings,
        MemoryStoreConfig,
    )

    # Get Memory Store connection settings from environment
    memory_store_endpoint = os.getenv("MEMORY_STORE_ENDPOINT", "https://localhost:9200")

    # Configure geocoding - the MemoryClient will automatically create and
    # wire the geocoder to LongTermMemory
    geocoding_config = GeocodingConfig(
        enabled=True,
        provider=GeocodingProvider.GOOGLE
        if args.provider == "google"
        else GeocodingProvider.NOMINATIM,
        api_key=SecretStr(args.api_key) if args.api_key else None,
        cache_results=True,
        user_agent="lennys-memory/1.0",
    )

    settings = MemorySettings(
        backend="memory_store",
        memory_store=MemoryStoreConfig(
            endpoint=memory_store_endpoint,
        ),
        embedding=EmbeddingConfig(
            provider=EmbeddingProvider.BEDROCK,
            model="amazon.titan-embed-text-v2:0",
            dimensions=1024,
            aws_region=os.getenv("AWS_REGION", "us-west-2"),
        ),
        geocoding=geocoding_config,
    )

    print(f"Geocoding Location entities using {args.provider.title()}")
    print(f"Memory Store: {memory_store_endpoint}")
    print()

    async with MemoryClient(settings) as client:

        def on_progress(processed: int, total: int):
            if args.verbose or processed % 10 == 0 or processed == total:
                print(f"  Progress: {processed}/{total} locations processed")

        stats = await client.long_term.geocode_locations(
            batch_size=args.batch_size,
            skip_existing=args.skip_existing,
            on_progress=on_progress,
        )

        print()
        print("Geocoding complete!")
        print(f"  Processed: {stats['processed']} locations")
        print(f"  Geocoded:  {stats['geocoded']} locations")
        print(f"  Skipped:   {stats['skipped']} locations")
        print(f"  Failed:    {stats['failed']} locations")

        if stats["geocoded"] > 0:
            print()
            print("You can now use spatial queries like:")
            print("  await memory.long_term.search_locations_near(lat, lon, radius_km=10)")


if __name__ == "__main__":
    asyncio.run(main())
