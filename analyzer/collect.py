#!/usr/bin/env python3

# Copyright (c) 2019-2022 Varada, Inc.
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
import pathlib
import sys
import time
import logbook
import requests
from requests.auth import HTTPBasicAuth

logbook.StreamHandler(sys.stderr).push_application()
log = logbook.Logger("collect")


class Client:
    def __init__(self, username, password, certificate_verification, username_request_header):
        self._username = username
        self._password = password
        self._certificate_verification = certificate_verification
        self._req_headers = self.set_req_headers(username_request_header)

    def set_req_headers(self, request_header):
        if request_header:
            if request_header not in ('X-Trino-User', 'X-Presto-User'):
                log.warning(
                    'Got client-request-header which is not X-Trino-User or X-Presto-User, collecting JSONs might fail')
            return {request_header: "analyzer"}
        else:
            return {"X-Trino-User": "analyzer",
                    "X-Presto-User": "analyzer"}

    def get(self, url):
        if all([self._username, self._password]):
            response = requests.get(url, self._req_headers, auth=HTTPBasicAuth(
                self._username,
                self._password), verify=self._certificate_verification)
        else:
            response = requests.get(url, headers=self._req_headers)

        if not response.ok:
            log.warn("HTTP {} {} for url: {}", response.status_code, response.reason, url)
            return None
        else:
            return response


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--coordinator", default="http://localhost:8080")
    p.add_argument("-e", "--query-endpoint", default="/v1/query")
    p.add_argument("-u", "--username")
    p.add_argument("--username-request-header")
    p.add_argument("-p", "--password")
    p.add_argument("--certificate-verification", default=True, type=str_to_bool)
    p.add_argument("-o", "--output-dir", default="JSONs", type=pathlib.Path)
    p.add_argument("-d", "--delay", default=0.1, type=float)
    p.add_argument("--loop", default=False, action="store_true")
    p.add_argument("--loop-delay", type=float, default=1.0)
    args = p.parse_args()
    client = Client(args.username, args.password, args.certificate_verification, args.username_request_header)
    endpoint = "{}{}".format(args.coordinator, args.query_endpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    done_state = {"FINISHED", "FAILED"}
    while True:
        # Download last queries' IDs:
        response = client.get(endpoint)
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
                response = client.get(url)
                if not response:
                    continue
            except Exception:
                log.exception("Failed to download {}", query_id)
                continue

            with gzip.open(output_file.open("wb"), "wb") as f:
                f.write(response.content)

        if args.loop:
            time.sleep(args.loop_delay)
        else:
            break


if __name__ == "__main__":
    main()
