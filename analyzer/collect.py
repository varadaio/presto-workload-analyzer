#!/usr/bin/env python3

# Copyright (c) 2019-2021 Varada, Inc.
# This file is part of Presto Workload Analyzer.
#
# Presto Workload Analyzer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Presto Workload Analyzer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Presto Workload Analyzer.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import gzip
import json
import logbook
import os
import pathlib
import requests
import sys
import time
import traceback
from s3 import s3_provider
import helpers

logbook.StreamHandler(sys.stderr).push_application()
log = logbook.Logger("collect")


def get(url):
    # User header is required by latest Presto versions.
    response = requests.get(url, headers={
        "X-Trino-User": "analyzer",
    })
    if not response.ok:
        log.warn("HTTP {} {} for url: {}", response.status_code, response.reason, url)
        return None
    else:
        return response


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--coordinator", default="http://localhost:8080")
    p.add_argument("-e", "--query-endpoint", default="/v1/query")
    p.add_argument("-o", "--output-dir", default="JSONs", type=pathlib.Path)
    p.add_argument("-d", "--delay", default=0.1, type=float)
    p.add_argument("--loop", default=False, action="store_true")
    p.add_argument("--loop-delay", type=float, default=1.0)
    args = p.parse_args()

    endpoint = "{}{}".format(args.coordinator, args.query_endpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    done_state = {"FINISHED", "FAILED"}
    while True:
        # Download last queries' IDs:
        response = get(endpoint)
        if not response:
            return
        ids = [q["queryId"] for q in response.json() if q["state"] in done_state]
        log.debug("Found {} queries", len(ids))

        # Download new queries only:
        for query_id in sorted(ids):
            output_file = args.output_dir / (query_id + ".json.gz")  # to save storage
            if output_file.exists():  # don't download already downloaded JSONs
                continue

            url = "{}/{}?pretty".format(endpoint, query_id)  # human-readable JSON
            time.sleep(args.delay)  # for rate-limiting
            log.info("Downloading {} -> {}", url, output_file)
            try:
                response = get(url)
                if not response:
                    continue
            except Exception:
                log.exception("Failed to download {}", query_id)
                continue

            with gzip.open(output_file.open("wb"), "wb") as f:
                f.write(response.content)

            s3_file = f"presto_analyzer/{output_file}"

            try:
                s3_provider.upload(output_file, s3_file)

            except Exception as e:
                log.error(f' Exception in Uploading file to S3 {output_file}'
                          f'exception={e}')

            helpers.remove_local_file(output_file)

        if args.loop:
            time.sleep(args.loop_delay)
        else:
            break


if __name__ == "__main__":
    main()
