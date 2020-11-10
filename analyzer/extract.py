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
import pathlib
import sys
import tqdm

logbook.StreamHandler(sys.stderr).push_application()
log = logbook.Logger("extract")

TIME_UNITS = [
    ("ns", 1e-9),
    ("ms", 1e-3),
    ("us", 1e-6),
    ("s", 1),
    ("m", 60),
    ("h", 60 * 60),
    ("d", 60 * 60 * 24),
]

SIZE_UNITS = [
    ("TB", 1024 * 1024 * 1024 * 1024),
    ("GB", 1024 * 1024 * 1024),
    ("MB", 1024 * 1024),
    ("kB", 1024),
    ("B", 1),
]


def parse_units(s, units):
    if s is None:
        return None
    for suffix, factor in units:
        if s.endswith(suffix):
            return float(s[: -len(suffix)]) * factor
    return float(s)


def parse_size(s):
    return parse_units(s, SIZE_UNITS)


def parse_time(s):
    return parse_units(s, TIME_UNITS)


def get_operators(summaries):
    for s in summaries:
        try:
            yield dict(
                node_id=s["planNodeId"],
                type=s["operatorType"],
                input_size=parse_size(s.get("rawInputDataSize") or s.get("inputDataSize"))
                           or parse_size(s["inputDataSize"]),
                output_size=parse_size(s["outputDataSize"]),
                network_size=parse_size(s.get("internalNetworkInputDataSize")),
                input_rows=s.get("rawInputPositions", 0) or s.get("inputPositions", 0),
                output_rows=s["outputPositions"],
                network_rows=s.get("internalNetworkInputPositions"),
                peak_mem=parse_size(s.get("peakTotalMemoryReservation")) if 'peakTotalMemoryReservation' in s else 0,
                input_cpu=parse_time(s["addInputCpu"]),
                output_cpu=parse_time(s["getOutputCpu"]),
                finish_cpu=parse_time(s["finishCpu"]),
                input_wall=parse_time(s["addInputWall"]),
                output_wall=parse_time(s["getOutputWall"]),
                finish_wall=parse_time(s["finishWall"]),
                blocked_wall=parse_time(s["blockedWall"])
                # TODO: wait time
            )
        except KeyError:
            log.exception("missing key for {}", s)
            raise


def iter_plans(stage):
    p = stage.get("plan")
    if p:
        yield {k: p[k] for k in ["id", "root"]}
        for substage in stage["subStages"]:
            yield from iter_plans(substage)


def build_tasks_in_substages(stage):
    substages = []
    for sub in stage.get('subStages', []):
        sub_stage_tasks = []
        for task in sub.get('tasks', []):
            task_stats = {}
            for k in ['totalScheduledTime', 'totalCpuTime', 'totalBlockedTime']:
                task_stats[k] = parse_time(task.get('stats', {}).get(k))
            task_status = {}
            for k in ['taskId', 'state', 'self']:
                task_status[k] = task.get('taskStatus', {}).get(k)
            task_data = dict(
                taskStatus=task_status,
                stats=task_stats)
            sub_stage_tasks.append(task_data)
        substages.append(dict(
            tasks=sub_stage_tasks,
            subStages=build_tasks_in_substages(sub)))
    return substages


def summary(j: dict):
    session = j["session"]
    stats = j["queryStats"]

    if session.get('catalogProperties', {}).get('varada', {}).get('internal_query', '') == 'true':
        # varada internal query - skip it
        return None

    fragments = None
    substages = None
    stage = j.get("outputStage")
    if stage:
        fragments = list(iter_plans(stage))
        substages = build_tasks_in_substages(stage)

    return dict(
        query=j["query"],
        query_id=j["queryId"],
        user=session["user"],
        state=j["state"],
        error_code=j.get("errorCode"),
        update=j.get("updateType"),
        elapsed_time=parse_time(stats["elapsedTime"]),
        cpu_time=parse_time(stats["totalCpuTime"]),
        scheduled_time=parse_time(stats["totalScheduledTime"]),
        blocked_time=parse_time(stats["totalBlockedTime"]),
        input_size=(
                parse_size(stats["rawInputDataSize"])
                or parse_size(stats.get("inputDataSize"))
                or 0
        ),
        output_size=parse_size(stats["outputDataSize"]),
        network_size=parse_size(stats.get("internalNetworkInputDataSize")),
        input_rows=stats["rawInputPositions"],
        output_rows=stats["outputPositions"],
        network_rows=stats.get("internalNetworkInputPositions"),
        peak_mem=parse_size(stats["peakTotalMemoryReservation"]),
        written_size=parse_size(stats.get("rawWrittenDataSize")),
        operators=list(get_operators(stats["operatorSummaries"])),
        inputs=j["inputs"],
        output=j.get("output"),
        fragments=fragments,
        substages=substages
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input-dir", type=pathlib.Path)
    p.add_argument("-l", "--limit", type=int)
    p.add_argument("-q", "--quiet", action="store_true", default=False)
    args = p.parse_args()

    paths = [
        p for pattern in ["*.json", "*.json.gz"] for p in args.input_dir.glob(pattern)
    ]
    log.info("{} JSONs found at {}", len(paths), args.input_dir.absolute())
    paths = sorted(paths)
    if args.limit is not None:
        paths = paths[: args.limit]

    items = [(p, p.stat().st_size) for p in paths]
    total_size = sum(i[1] for i in items)
    compressed_file = args.input_dir / "summary.jsonl.gz"
    with gzip.open(str(compressed_file), "wt") as output:
        with tqdm.tqdm(total=total_size, unit="B", unit_scale=True, disable=args.quiet) as pbar:
            for path, size in items:
                input_file = (
                    gzip.open(str(path), "rt")
                    if path.name.endswith(".gz")
                    else path.open("rt")
                )
                with input_file as f:
                    s = summary(json.load(f))
                    if s:
                        json.dump(s, output)
                        output.write("\n")
                pbar.update(size)

    log.info(
        "Extracted {} JSONs into {} ({:.3f} MB in GZipped JSONL format)",
        len(paths),
        output.name,
        compressed_file.stat().st_size / 1e6,
    )


if __name__ == "__main__":
    main()
