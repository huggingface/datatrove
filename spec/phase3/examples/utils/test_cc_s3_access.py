#!/usr/bin/env python3
"""
Test CommonCrawl S3 Access Patterns

Tests two approaches:
1. Using paths_file with anonymous access (no credentials)
2. Using glob patterns with AWS credentials

This will help us understand which approach works for production.
"""

import gzip
from pathlib import Path

import requests


def test_paths_file_approach():
    """Test 1: Download paths file and stream a WARC anonymously."""
    import requests

    print("=" * 80)
    print("Test 1: Paths File + Anonymous S3 Streaming")
    print("=" * 80)

    # Download warc.paths.gz for CC-MAIN-2018-17 (our test crawl)
    crawl_id = "CC-MAIN-2018-17"
    paths_url = f"https://data.commoncrawl.org/crawl-data/{crawl_id}/warc.paths.gz"

    print(f"\n1. Downloading paths file: {paths_url}")
    response = requests.get(paths_url)
    if response.status_code != 200:
        print(f"   ❌ Failed to download: {response.status_code}")
        return False

    # Extract paths
    paths_content = gzip.decompress(response.content).decode('utf-8')
    warc_paths = [line.strip() for line in paths_content.split('\n') if line.strip()]

    print(f"   ✅ Found {len(warc_paths)} WARC files")
    print(f"   First path: {warc_paths[0]}")

    # Save first 10 paths to file for testing
    paths_file = Path("spec/phase3/data/cc_warc_paths.txt")
    paths_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to full HTTPS paths (not S3)
    full_paths = [f"https://data.commoncrawl.org/{path}" for path in warc_paths[:10]]
    paths_file.write_text('\n'.join(full_paths))
    print(f"   ✅ Saved first 10 paths to {paths_file}")

    # Try to stream one WARC file via HTTPS (public access)
    print(f"\n2. Testing HTTPS streaming of first WARC...")
    test_url = f"https://data.commoncrawl.org/{warc_paths[0]}"

    try:
        import requests

        # Stream first chunk
        response = requests.get(test_url, stream=True)
        if response.status_code != 200:
            print(f"   ❌ HTTP {response.status_code}")
            return False

        first_bytes = next(response.iter_content(1024))
        print(f"   ✅ Successfully read {len(first_bytes)} bytes")
        print(f"   File header: {first_bytes[:50]}")
        return True
    except Exception as e:
        print(f"   ❌ Failed to stream: {e}")
        return False


def test_glob_with_credentials():
    """Test 2: Try glob pattern with AWS credentials (if configured)."""
    print("\n" + "=" * 80)
    print("Test 2: Glob Pattern with AWS Credentials")
    print("=" * 80)

    # Check if AWS credentials are configured
    import os
    has_creds = (
        os.environ.get('AWS_ACCESS_KEY_ID') or
        os.environ.get('AWS_PROFILE') or
        Path.home().joinpath('.aws', 'credentials').exists()
    )

    if not has_creds:
        print("\n⚠️  No AWS credentials found. Skipping this test.")
        print("   To test with credentials, either:")
        print("   - Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        print("   - Configure ~/.aws/credentials")
        print("   - Set AWS_PROFILE")
        return None

    print("\n✅ AWS credentials detected. Testing glob pattern...")

    # Try to list files using glob pattern
    try:
        import s3fs

        # Create authenticated S3 filesystem
        fs = s3fs.S3FileSystem(anon=False)

        # Try to list files with glob
        pattern = "commoncrawl/crawl-data/CC-MAIN-2018-17/segments/1524125937193.1/warc/*.warc.gz"
        files = fs.glob(pattern)[:5]  # First 5 files

        if files:
            print(f"   ✅ Successfully listed {len(files)} files with glob pattern")
            print(f"   First file: {files[0]}")
            return True
        else:
            print("   ⚠️  No files found (possible permission issue)")
            return False
    except Exception as e:
        print(f"   ❌ Failed to list files: {e}")
        return False


def main():
    """Run both tests and summarize results."""
    print("\nCommonCrawl S3 Access Testing\n")

    result1 = test_paths_file_approach()
    result2 = test_glob_with_credentials()

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\n1. Paths file + anonymous access: {'✅ WORKS' if result1 else '❌ FAILED'}")
    print(f"2. Glob pattern + credentials:    {'✅ WORKS' if result2 else '⚠️  SKIPPED/FAILED' if result2 is None else '❌ FAILED'}")

    print("\nRecommendation:")
    if result1:
        print("  → Use paths_file approach for production (no credentials needed)")
        print("  → Download warc.paths.gz and use paths_file parameter")

    if result2:
        print("  → Glob patterns work with your AWS credentials")
        print("  → Can use data_folder with glob_pattern if credentials configured")


if __name__ == "__main__":
    main()
