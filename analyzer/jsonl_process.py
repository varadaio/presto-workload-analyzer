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
import ast
import gzip
import json
import logbook
import pathlib
import sys
from copy import deepcopy
from tqdm import tqdm
import itertools
import nested_lookup as nl
from pprint import pformat

logbook.StreamHandler(sys.stderr).push_application()
log = logbook.Logger("json")


def filter_line(stat, filter_dict, or_and=True, include_absent=True):
    """
    stat - dict which represents JSONL line
    filter_dict - simple dict to use for filtering
    or_and - True for OR logic else AND
    include_absent - if True and filter finds nothing return True

    returns True if line has entry from filter_dict
    Currently support simple keys and values
    """
    found = [filter_dict[k] in nl.nested_lookup(k, stat) for k in filter_dict]
    num_exist = [nl.get_occurrence_of_key(stat, k) for k in filter_dict]

    return (include_absent and sum(num_exist) == 0) or (or_and and any(found)) or (not or_and and all(found))


class NameObfuscator:
    def __init__(self, prefix):
        self.__prefix = prefix
        self.__name_map = dict()

    def __call__(self, old_name):
        if old_name != '':
            return self.__name_map.setdefault(old_name, self.__prefix + str(len(self.__name_map)))
        return ''

    def __str__(self):
        return pformat(self.__name_map)


class ListObfuscator(NameObfuscator):
    def __call__(self, old_list):
        print(old_list)
        if isinstance(old_list, str):  # str due to nested_lookup behavior
            old_list = ast.literal_eval(old_list)

        if not isinstance(old_list, list):
            raise Exception(f"List obfuscator got incompatible type {type(old_list)}")

        return [super(ListObfuscator, self).__call__(old_name) for old_name in old_list]


def process_line(stat, obfuscator_dict):
    """
    obfuscator_dict - dict with structure {key: callback_function}
    """
    new_stat = deepcopy(stat)
    for k in obfuscator_dict:
        nl.nested_alter(new_stat, k, callback_function=obfuscator_dict[k], conversion_function=str, in_place=True)
    return new_stat


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-i",
        "--input-file",
        type=pathlib.Path,
        help="Path to the extracted JSONL file (the output of extract.py)"
    )
    p.add_argument(
        "-o",
        "--output-file",
        type=pathlib.Path,
        default="./processed_summary.jsonl.gz",
        help="Path to the resulting zipped JSONL file",
    )

    p.add_argument("-l", "--limit", type=int)
    p.add_argument("--fail-on-error", action="store_true", default=False)
    p.add_argument("-q", "--quiet", action="store_true", default=False)
    # filter parameters
    p.add_argument("--filter-schema", type=str, help="Save queries for this schema only")
    # obfuscation parameters
    p.add_argument("--remove-query", action="store_true", default=False)
    p.add_argument("--rename-schemas", action="store_true", default=False)
    p.add_argument("--rename-catalogs", action="store_true", default=False)
    p.add_argument("--remove-locations", action="store_true", default=False)
    p.add_argument("--rename-user", action="store_true", default=False)
    p.add_argument("--rename-partitions", action="store_true", default=False)
    args = p.parse_args()

    if not args.input_file:
        p.error("Input filename missing")

    # prepare obfuscators
    obfuscator_dict = dict()
    if args.remove_query:
        obfuscator_dict["query"] = lambda x: ''
        # In case of EXPLAIN remove the query details as well (TODO: would also remove VALUES)
        obfuscator_dict["rows"] = lambda x: ''

    if args.rename_schemas:
        schema_obfuscator = NameObfuscator('schema')
        obfuscator_dict["schema"] = schema_obfuscator
        obfuscator_dict["schemaName"] = schema_obfuscator

    if args.rename_catalogs:
        catalog_obfuscator = NameObfuscator('catalog')
        obfuscator_dict["catalogName"] = catalog_obfuscator

    if args.remove_locations:
        obfuscator_dict["location"] = lambda x: ''
        obfuscator_dict["targetPath"] = lambda x: ''
        obfuscator_dict["writePath"] = lambda x: ''

    if args.rename_user:
        user_obfuscator = NameObfuscator('user')
        obfuscator_dict["user"] = user_obfuscator
        obfuscator_dict["principal"] = user_obfuscator

    if args.rename_partitions:
        partitions_obfuscator = ListObfuscator('partition')
        obfuscator_dict["partitionIds"] = partitions_obfuscator

    obfuscate = len(obfuscator_dict) > 0

    log.info(
        "loading {} = {:.3f} MB", args.input_file, args.input_file.stat().st_size / 1e6
    )
    if args.input_file.name.endswith(".gz"):
        lines = gzip.open(args.input_file.open("rb"), "rt")
    else:
        lines = args.input_file.open("rt")

    if args.limit:
        lines = itertools.islice(lines, args.limit)

    lines = list(lines)
    compressed_file = args.output_file
    if args.output_file.suffix != ".gz":
        compressed_file = compressed_file + '.gz'
    with gzip.open(str(compressed_file), "wt") as output:
        for line in tqdm(lines, unit="lines", disable=args.quiet):
            try:
                s = json.loads(line)
                if args.filter_schema and not filter_line(s, {"schema": args.filter_schema, "schemaName": args.filter_schema}):
                    continue

                if obfuscate:
                    json.dump(process_line(s, obfuscator_dict), output)
                else:
                    json.dump(s, output)
                output.write("\n")
            except Exception:
                log.exception("failed to process {}", line)
                if args.fail_on_error:
                    raise
    if not args.quiet:
        log.info("{} processing done", len(lines))
        if args.rename_schemas:
            log.info("Schemas translation table:\n{}", schema_obfuscator)
        if args.rename_catalogs:
            log.info("Catalogs translation table:\n{}", catalog_obfuscator)
        if args.rename_user:
            log.info("Users translation table:\n{}", user_obfuscator)
        if args.rename_partitions:
            log.info("Partitions translation table:\n{}", partitions_obfuscator)


if __name__ == "__main__":
    main()
