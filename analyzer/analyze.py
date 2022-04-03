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
import collections
import datetime
import gzip
import itertools
import json
import logbook
import numpy
import pathlib
import re
import sys
import zipfile
from tqdm import tqdm
from inspect import signature

logbook.StreamHandler(sys.stderr).push_application()
log = logbook.Logger("analyze")

from bokeh.embed import json_item
from bokeh.models import ColumnDataSource, TapTool, Span, Slope, ranges, LabelSet
from bokeh.models.callbacks import CustomJS
from bokeh.palettes import Category20c, Category10, Colorblind
from bokeh.plotting import figure, output_file, save


def groupby(keys, values, reducer):
    result = collections.defaultdict(list)
    for k, v in zip(keys, values):
        result[k].append(v)
    result = [(k, reducer(vs)) for k, vs in result.items()]
    result.sort()
    return result


_ANALYZERS = []


def run(func):
    _ANALYZERS.append(func)
    return func


def query_datetime(query_id):
    return datetime.datetime.strptime(query_id[:15], "%Y%m%d_%H%M%S")


def trunc_hour(dt):
    return datetime.datetime(dt.year, dt.month, dt.day, dt.hour)


def trunc_date(dt):
    return datetime.datetime(dt.year, dt.month, dt.day)


TOOLS = "tap,pan,wheel_zoom,box_zoom,save,reset"
TOOLS_SIMPLE = "save,reset"

COPY_JS = """
    const values = source.data['copy_on_tap'];
    copyToClipboard(values[source.selected.indices[0]]);
"""  # copyToClipboard() is defined in output.template.html


@run
def scheduled_by_date(stats):
    """
    Shows cluster scheduled time by date, useful for high-level trend analysis.
    """
    scheduled_cores = [s["scheduled_time"] / (60 * 60 * 24) for s in stats]
    date = [query_datetime(s["query_id"]) for s in stats]
    date = [trunc_date(d) for d in date]
    date, scheduled_cores = zip(*groupby(date, scheduled_cores, sum))
    p = figure(
        title="Scheduled time by date",
        x_axis_label="Date",
        y_axis_label="Average scheduled cores",
        x_axis_type="datetime",
        sizing_mode="scale_width"
    )
    p.vbar(x=date, top=scheduled_cores, width=24 * 3600e3)
    return p


def add_constant_line(p, dim, value, line_color='black',
                      line_dash='dashed', line_width=2):
    constant_line_value = value
    constant_line = Span(location=constant_line_value,
                         dimension=dim, line_color=line_color,
                         line_dash=line_dash, line_width=line_width)
    p.add_layout(constant_line)


@run
def scheduled_by_hour(stats):
    """
    Shows cluster scheduled time by the hour, useful for low-level trend analysis.
    """
    scheduled_cores = [s["scheduled_time"] / (60 * 60) for s in stats]
    date = [query_datetime(s["query_id"]) for s in stats]
    date = [trunc_hour(d) for d in date]

    date, scheduled_cores = zip(*groupby(date, scheduled_cores, sum))
    p = figure(
        title="Scheduled time by hour",
        x_axis_label="Time",
        y_axis_label="Average scheduled cores",
        x_axis_type="datetime",
        sizing_mode="scale_width",
    )
    p.vbar(x=date, top=scheduled_cores, width=3600e3)
    return p


@run
def input_by_date(stats):
    """
    Shows cluster input bytes read by date, useful for high-level trend analysis.
    """
    input_size = [s["input_size"] / (1e12) for s in stats]
    date = [query_datetime(s["query_id"]) for s in stats]
    date = [trunc_date(d) for d in date]

    date, input_size_by_date = zip(*groupby(date, input_size, sum))
    p = figure(
        title="Input data read by date",
        x_axis_label="Date",
        y_axis_label="Input [TB]",
        x_axis_type="datetime",
        sizing_mode="scale_width",
    )
    p.vbar(x=date, top=input_size_by_date, width=24 * 3600e3)
    return p


@run
def input_by_hour(stats):
    """
    Shows cluster input bytes read by the hour, useful for low-level trend analysis.
    """
    input_size = [s["input_size"] / (1e12) for s in stats]
    date = [query_datetime(s["query_id"]) for s in stats]
    date = [trunc_hour(d) for d in date]

    date, input_size_by_date = zip(*groupby(date, input_size, sum))
    p = figure(
        title="Input data read by hour",
        x_axis_label="Time",
        y_axis_label="Input [TB]",
        x_axis_type="datetime",
        sizing_mode="scale_width",
    )
    p.vbar(x=date, top=input_size_by_date, width=3600e3)
    return p


@run
def queries_by_date(stats):
    """
    Shows the number of queries running on the cluster by date, useful for high-level trend analysis.
    """
    ids = [s["query_id"] for s in stats]
    date = [query_datetime(s["query_id"]) for s in stats]
    date = [trunc_date(d) for d in date]

    date, queries_by_date = zip(*groupby(date, [1] * len(date), sum))
    p = figure(
        title="Number of queries ran by date",
        x_axis_label="Date",
        y_axis_label="Number of queries",
        x_axis_type="datetime",
        sizing_mode="scale_width",
    )
    p.vbar(x=date, top=queries_by_date, width=24 * 3600e3)
    return p


@run
def queries_by_hour(stats):
    """
    Shows the number of queries running on the cluster by hour, useful for low-level trend analysis.
    """
    ids = [s["query_id"] for s in stats]
    date = [query_datetime(s["query_id"]) for s in stats]
    date = [trunc_hour(d) for d in date]

    date, queries_by_date = zip(*groupby(date, [1] * len(date), sum))
    p = figure(
        title="Number of queries ran by hour",
        x_axis_label="Time",
        y_axis_label="Number of queries",
        x_axis_type="datetime",
        sizing_mode="scale_width",
    )
    p.vbar(x=date, top=queries_by_date, width=3600e3)
    return p


@run
def peak_mem_by_query(stats):
    """
    Shows the peak memory used by each query as a function of time.
    Allow quick analysis of memory utilization patterns by different queries.

    Optimization Tip - Queries above the dashed black line use more than 10GB of memory and investigation is recommended.
    """
    peak_mem = [s["peak_mem"] for s in stats]
    query_ids = [s["query_id"] for s in stats]
    date = [query_datetime(query_id) for query_id in query_ids]
    p = figure(
        title="Peak memory used by queries",
        x_axis_label="Time",
        x_axis_type="datetime",
        y_axis_label="Query peak memory [B]",
        y_axis_type="log",
        sizing_mode="scale_width",
        tools=TOOLS,
    )
    p.yaxis.ticker = [1, 1e3, 1e6, 1e9, 1e10, 1e12]

    source = ColumnDataSource(dict(peak_mem=peak_mem, date=date, copy_on_tap=query_ids))
    p.select(type=TapTool).callback = CustomJS(args=dict(source=source), code=COPY_JS)
    p.circle("date", "peak_mem", source=source, alpha=0.5)
    add_constant_line(p, 'width', 1e10)
    return p


@run
def input_size_by_query(stats):
    """
    Shows the input bytes read by each query as a function of time.
    Allow quick analysis of input patterns by different queries.

    Optimization Tip - Queries above the dashed black line read more than 1TB of data.
    Consider using indexing/partitioning to reduce the amount of data being read and make sure predicates are being pushed-down.

    """
    input_size = [s["input_size"] for s in stats]
    query_ids = [s["query_id"] for s in stats]
    date = [query_datetime(query_id) for query_id in query_ids]
    p = figure(
        title="Input data read by queries",
        x_axis_label="Time",
        x_axis_type="datetime",
        y_axis_label="Input data read [B]",
        y_axis_type="log",
        sizing_mode="scale_width",
        tools=TOOLS,
    )
    p.yaxis.ticker = [1, 1e3, 1e6, 1e9, 1e12]

    source = ColumnDataSource(dict(input_size=input_size, date=date, copy_on_tap=query_ids))
    p.select(type=TapTool).callback = CustomJS(args=dict(source=source), code=COPY_JS)
    p.circle("date", "input_size", source=source, alpha=0.5)
    add_constant_line(p, 'width', 1e12)
    return p


@run
def elapsed_time_by_query(stats):
    """
    Shows the elapsed time of each query as a function of the query execution time.
    Allows quick analysis of client-side latency by different queries.

    Optimization Tip - Queries above the dashed black line take more than 5 minutes, consider optimizing them.
    """
    elapsed_time = [s["elapsed_time"] for s in stats]
    query_ids = [s["query_id"] for s in stats]
    date = [query_datetime(query_id) for query_id in query_ids]
    p = figure(
        title="Elapsed time by queries",
        x_axis_label="Time",
        x_axis_type="datetime",
        y_axis_label="Elapsed time [s]",
        y_axis_type="log",
        sizing_mode="scale_width",
        tools=TOOLS,
    )
    p.yaxis.ticker = [1e-3, 1, 1e3, 1e6]
    source = ColumnDataSource(dict(elapsed_time=elapsed_time, date=date, copy_on_tap=query_ids))
    p.select(type=TapTool).callback = CustomJS(args=dict(source=source), code=COPY_JS)
    p.circle("date", "elapsed_time", source=source, alpha=0.5)
    add_constant_line(p, 'width', 300)
    return p


@run
def queries_by_user(stats):
    """
    The fraction of queries executed by each user.
    Can be used to understand which users are responsible for the bulk of the queries running on the cluster.
    """
    users = [s["user"] for s in stats]

    items = groupby(users, [1] * len(users), sum)
    items.sort(key=lambda i: i[1], reverse=True)
    users, queries_by_user = zip(*items)
    return pie_chart(
        keys=users, values=queries_by_user, title="Number of queries by user"
    )


@run
def scheduled_by_user(stats):
    """
    The fraction of scheduled time by user.
    Can be used to understand which users are responsible for most of the scheduled time of queries running on the cluster.
    """
    scheduled_days = [s["scheduled_time"] / (60 * 60 * 24) for s in stats]
    users = [s["user"] for s in stats]

    items = groupby(users, scheduled_days, sum)
    items.sort(key=lambda i: i[1], reverse=True)
    users, scheduled_days_per_user = zip(*items)
    return pie_chart(
        keys=users, values=scheduled_days_per_user, title="Scheduled time by user"
    )


@run
def scheduled_by_update(stats):
    """
    The fraction of scheduled time by query type (e.g. INSERT, SELECT, CREATE TABLE).
    Can be used to understand which query type uses the most scheduled time on the cluster.
    """
    scheduled_days = [s["scheduled_time"] / (60 * 60 * 24) for s in stats]
    update = [(s["update"] or "SELECT") for s in stats]

    items = groupby(update, scheduled_days, sum)
    items.sort(key=lambda i: i[1], reverse=True)
    update, scheduled_time_per_update = zip(*items)
    return pie_chart(
        keys=update,
        values=scheduled_time_per_update,
        title="Scheduled time by query type",
    )


@run
def input_by_user(stats):
    """
    The fraction of input bytes read by each user.
    Can be used to understand which users are responsible for the most I/O utilization on the cluster.
    """
    input_size = [s["input_size"] / 1e12 for s in stats]
    users = [s["user"] for s in stats]

    items = groupby(users, input_size, sum)
    items.sort(key=lambda i: i[1], reverse=True)
    users, input_size_per_user = zip(*items)
    return pie_chart(
        keys=users, values=input_size_per_user, title="Input data read by user"
    )


@run
def output_vs_input(stats):
    """
    Shows the queries using a scatter plot of the input bytes read (x coordinate) and output bytes written (y coordinate).
    Can be useful for detecting "ETL-like" jobs (reading and writing a lot of data), and separating them from more
    "interactive-like" jobs (which usually result in much less data being read).
    """
    source = ColumnDataSource(dict(
        input_size=[s["input_size"] for s in stats],
        output_size=[s["output_size"] for s in stats],
        copy_on_tap=[s["query_id"] for s in stats]
    ))
    p = figure(
        title="Output size vs input size",
        x_axis_label="Input size [B]",
        y_axis_label="Output size [B]",
        x_axis_type="log",
        y_axis_type="log",
        sizing_mode="scale_width",
        tools=TOOLS,
    )
    p.xaxis.ticker = [1, 1e3, 1e6, 1e9, 1e12]
    p.yaxis.ticker = [1, 1e3, 1e6, 1e9, 1e12]
    p.select(type=TapTool).callback = CustomJS(args=dict(source=source), code=COPY_JS)
    p.circle("input_size", "output_size", source=source)
    return p


@run
def scheduled_vs_input(stats):
    """
    Shows the queries using a scatter plot of the input bytes read (x coordinate) and scheduled time used (y coordinate).
    Can be useful for detecting queries that read a lot of data, and require significant scheduled time.
    """
    source = ColumnDataSource(dict(
        input_size=[s["input_size"] for s in stats],
        scheduled_time=[s["scheduled_time"] for s in stats],
        copy_on_tap=[s["query_id"] for s in stats]
    ))
    p = figure(
        title="Scheduled time vs input size",
        x_axis_label="Input size [B]",
        y_axis_label="Scheduled time",
        x_axis_type="log",
        y_axis_type="log",
        sizing_mode="scale_width",
        tools=TOOLS,
    )
    p.xaxis.ticker = [1, 1e3, 1e6, 1e9, 1e12]
    yticks = [1e-3, 1, 60, 60 * 60, 24 * 60 * 60]
    p.yaxis.ticker = yticks
    p.yaxis.major_label_overrides = dict(zip(yticks, ["1ms", "1s", "1m", "1h", "1d"]))
    p.select(type=TapTool).callback = CustomJS(args=dict(source=source), code=COPY_JS)
    p.circle("input_size", "scheduled_time", source=source)
    return p


@run
def elapsed_vs_input(stats):
    """
    Shows the queries using a scatter plot of the input bytes read (x coordinate) and their client-side duration (y coordinate).
    Can be useful for detecting queries that read a lot of data and have significant user latency.
    """
    source = ColumnDataSource(dict(
        input_size=[s["input_size"] for s in stats],
        elapsed_time=[s["elapsed_time"] for s in stats],
        copy_on_tap=[s["query_id"] for s in stats]
    ))
    p = figure(
        title="Elapsed time vs input size",
        x_axis_label="Input size [B]",
        y_axis_label="Elapsed time",
        x_axis_type="log",
        y_axis_type="log",
        sizing_mode="scale_width",
        tools=TOOLS,
    )
    p.xaxis.ticker = [1, 1e3, 1e6, 1e9, 1e12]
    yticks = [1e-3, 1, 60, 60 * 60, 24 * 60 * 60]
    p.yaxis.ticker = yticks
    p.yaxis.major_label_overrides = dict(zip(yticks, ["1ms", "1s", "1m", "1h", "1d"]))
    p.select(type=TapTool).callback = CustomJS(args=dict(source=source), code=COPY_JS)
    p.circle("input_size", "elapsed_time", source=source)
    return p


def pie_chart(keys, values, title, top=20):
    values = numpy.array(values)
    sum_values = values.sum()
    if not sum_values:
        return
    percent = 100 * values / sum_values
    relevant = (percent > 0.1) & (numpy.arange(len(keys)) < top - 1)

    keys = [t for r, t in zip(relevant, keys) if r]
    if numpy.any(~relevant):
        keys.append("All the rest")
        values = numpy.concatenate((values[relevant], [values[~relevant].sum()]))
    percent = 100 * values / values.sum()

    # make sure percent.sum() == 100%
    percent = percent.round(2)
    percent[-1] += (100 - percent.sum())

    angle = 2 * numpy.pi * numpy.concatenate(([0], percent.cumsum() / percent.sum()))

    p = figure(
        title=title,
        width=800,
        tooltips="@labels",
        tools="hover",
        toolbar_location=None,
        x_range=(-2, 1),
        sizing_mode="scale_width",
    )
    k = max(3, len(keys))
    color = Category20c[k][: len(keys)]
    data = ColumnDataSource(
        dict(
            start=angle[:-1],
            end=angle[1:],
            keys=keys,
            color=color,
            percent=percent,
            labels=["{} {:.2f}%".format(shorten(k), p) for k, p in zip(keys, percent)],
        )
    )
    p.wedge(
        x=-1,
        y=0,
        radius=0.8,
        start_angle="start",
        end_angle="end",
        fill_color="color",
        line_color="white",
        legend_field="labels",
        source=data,
    )
    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None
    return p


def shorten(s):
    if len(s) > 30:
        s = s[:30] + "..."
    return s


@run
def operator_wall(stats, top=20):
    """
    The fraction of wall time usage (CPU and waiting) by each Presto operator.
    Can be used to understand the amount of wall time used by each part of the query processing - e.g. scan/join/aggreggation.
    """
    operators = [op for s in stats for op in s["operators"]]
    types = [op["type"].replace("Operator", "") for op in operators]
    selectivity = [
        op["output_rows"] / op["input_rows"] for op in operators if op["input_rows"]
    ]
    wall = [
        op["input_wall"] + op["output_wall"] + op["finish_wall"] for op in operators
    ]
    items = groupby(types, wall, sum)
    items.sort(key=lambda item: item[1], reverse=True)
    types, total_wall = zip(*items)
    return pie_chart(
        keys=types, values=total_wall, title="Wall time usage by operator type"
    )


def scan_operators(operators):
    for op in operators:
        if "Scan" in op["type"]:
            yield op


def scanfilter_operators(operators):
    for op in operators:
        if "ScanFilter" in op["type"]:
            yield op


def last_element(iterator):
    for element in iterator:
        pass
    return element


def parse_table_name(scan_node):
    table = scan_node["table"]
    handle = table["connectorHandle"]
    schema_table_name = handle.get("schemaTableName")
    if schema_table_name:
        schema_name = schema_table_name["schema"]
        table_name = schema_table_name["table"]
    else:
        schema_name = handle.get("schemaName")  # may be missing
        table_name = handle.get("tableName") or handle.get("table")
        if table_name is None:
            # MemoryTableHandle doesn't contain its name in PrestoSQL 306+
            table_name = "{}:{}".format(handle["@type"], handle["id"])
        if isinstance(table_name, dict):  # JMX may contain schema information here
            schema_name = table_name["schema"]
            table_name = table_name["table"]

    connector_id = table.get("connectorId") or table["catalogName"]
    values = [connector_id, schema_name, table_name]
    values = [v for v in values if v is not None]
    return ".".join(values)


@run
def wall_by_table_scan(stats):
    """
    The fraction of wall time used to scan the top K tables.
    Useful for detecting the tables scanned the most.
    """
    tables = []
    wall = []
    for s in stats:
        node_map = {node["id"]: node for node in nodes_from_stats(s)}
        for op in scan_operators(s["operators"]):
            node_id = op["node_id"]
            node = node_map[node_id]
            scan_node = last_element(
                iter_nodes(node)
            )  # DFS to get the "lowest" table scan node
            try:
                table_name = parse_table_name(scan_node)
            except KeyError as e:
                log.error("node: {}", scan_node)
                raise
            tables.append(table_name)
            wall.append(op["input_wall"] + op["output_wall"] + op["finish_wall"])

    if not tables:
        return

    items = groupby(tables, wall, sum)
    items.sort(key=lambda item: item[1], reverse=True)
    tables, total_wall = zip(*items)
    return pie_chart(
        keys=tables, values=total_wall, title="Wall time utilization by table scan"
    )


def wall_by_selectivity_bins(stats, bins=10, max_selectivity=1, title="Wall time of table scans by selectivity bins"):
    selectivity = []
    wall = []
    for s in stats:
        node_map = {node["id"]: node for node in nodes_from_stats(s)}
        for op in scan_operators(s["operators"]):
            node_id = op["node_id"]
            node = node_map[node_id]
            if op["input_rows"]:
                wall.append(op["input_wall"] + op["output_wall"] + op["finish_wall"])
                selectivity.append(op["output_rows"] / op["input_rows"])

    if not selectivity:
        return

    bin_step = 1. / bins

    wall = numpy.array(wall)
    selectivity_bins = numpy.abs(numpy.round(numpy.array(selectivity) - bin_step / 2, 1))

    # each bin should be represented
    wall = numpy.append(wall, numpy.zeros(bins))
    selectivity_bins = numpy.append(selectivity_bins, numpy.arange(0, max_selectivity, bin_step))

    # convert bin to string for representation
    selectivity_bins = numpy.array([
        '%0.2f' % x if x <= max_selectivity + 1e-9 else "Above" for x in selectivity_bins])
    # convert total_wall to percentage for representation
    wall = wall / wall.sum() * 100

    items = groupby(selectivity_bins, wall, sum)
    items.sort(key=lambda item: item[0])
    selectivity_bins, total_wall = zip(*items)

    tooltips = ['Selectivity %0.2f-%0.2f: %0.2f%% of wall time' % (float(x), float(x) + bin_step, y) for (x, y) in
                zip(selectivity_bins, total_wall) if x != 'Above']
    if 'Above' == selectivity_bins[-1]:
        tooltips.append('Selectivity above %0.2f: %0.2f%% of wall time' % (float(selectivity_bins[-2]), total_wall[-1]))

    source = ColumnDataSource(dict(
        x=selectivity_bins,
        top=total_wall,
        text=['%0.1f%%' % x for x in total_wall],
        tooltips=tooltips))

    p = figure(
        width=800,
        title=title,
        x_axis_label="Selectivity Bin",
        y_axis_label="Wall time (%)",
        sizing_mode="scale_width",
        x_range=selectivity_bins,
        y_range=(0, 100),
        tools=TOOLS_SIMPLE,
        tooltips='@tooltips')

    labels = LabelSet(x='x', y='top', text='text', level='glyph',
                      x_offset=-800 / len(selectivity_bins) / 4, y_offset=5, source=source, render_mode='canvas',
                      text_font_size="10pt")

    p.vbar(source=source, x='x', top='top', width=1, line_width=1, line_color='black')

    p.add_layout(labels)

    return p


@run
def wall_by_selectivity_10(stats):
    """
    The scan wall time percentage for each selectivity bin - 0.1 sized bins. Selectivity is defined as output_rows / input_rows, so the bin for 0.20 would mean selecting between 20% to 30% of the data.
    Useful for detecting whether predicate pushdown is effecient. If most scans are not very selective (selectivity above 0.8, up to a full scan which is selectivity 1) then predicate pushdown is effecient and the filtering is taken care of at the data source and not by Presto.
    """
    return wall_by_selectivity_bins(stats)


@run
def wall_by_selectivity_100_first_20(stats):
    """
    The scan wall time percentage for each selectivity bin - with focus on the most selective queries with small 0.01 bins. Selectivity is defined as output_rows / input_rows, so the bin for 0.02 would mean selecting between 2% to 3% of the data.
    Useful for understading predicate pushdown effeciency in high selectivity cases.
    """
    return wall_by_selectivity_bins(stats, bins=100, max_selectivity=0.2,
                                    title="Wall time of table scans by selectivity bins (very selective)")


def _get_colors(colorblind=False):
    return Colorblind[8] if colorblind else Category10[10]


def _get_size(colorblind=False):
    return 8 if colorblind else 4


@run
def filter_selectivity_1(stats):
    """
    Shows the selectivity of filter operators, which is the fraction of rows left after filtering is over.
    It plots the output number of rows as a function of the input number of rows, so non-selective filters will be near the y=x line, and highly selective filters will be significantly below it.
    Such highly selective filters are excellent candidates for predicate push-down, enabled by Varada Inline Indexing.
    """

    data = {}
    for s in stats:
        for op in s["operators"]:
            if "Filter" in op["type"]:
                data.setdefault("input_rows", []).append(op["input_rows"])
                data.setdefault("output_rows", []).append(op["output_rows"])
                data.setdefault("copy_on_tap", []).append(s["query_id"])

    source = ColumnDataSource(data)
    p = figure(
        title="Input and output rows for filtering operators",
        x_axis_label="Number of input rows",
        x_axis_type="log",
        y_axis_label="Number of output rows",
        y_axis_type="log",
        sizing_mode="scale_width",
        tools=TOOLS,
    )
    p.circle(x="input_rows", y="output_rows", alpha=0.5, source=source)
    p.select(type=TapTool).callback = CustomJS(args=dict(source=source), code=COPY_JS)
    return p


@run
def walltime_vs_selectivity(stats, topK=5, colorblind=False):
    """
    Plots the wall time of Scan and Filter operators that were applied on the top K tables, as a function of the operator's selectivity.
    The top K tables are the tables with the highest amount of wall time spent on performing Scan and Filter operations.
    Useful for detecting tables in which Scan operations can be improved using partitioning/indexing.

    Optimization Tip - Queries to the left of the dashed black line are selective queries, and can benefit significantly from predicate push-down and partitioning/indexing.

    """
    tables = []
    wall = []
    selectivity = []
    query_ids = []
    for s in stats:
        node_map = {node["id"]: node for node in nodes_from_stats(s)}
        for op in scanfilter_operators(s["operators"]):
            node_id = op["node_id"]
            node = node_map[node_id]
            scan_node = last_element(
                iter_nodes(node)
            )  # DFS to get the "lowest" table scan node
            try:
                table_name = parse_table_name(scan_node)
            except KeyError as e:
                log.error("node: {}", scan_node)
                raise
            if op["input_rows"]:
                tables.append(table_name)
                wall.append(op["input_wall"] + op["output_wall"] + op["finish_wall"])
                selectivity.append(op["output_rows"] / op["input_rows"])
                query_ids.append(s["query_id"])

    if not tables:
        return

    tables = numpy.array(tables)
    wall = numpy.array(wall)
    selectivity = numpy.array(selectivity)
    query_ids = numpy.array(query_ids)

    items = groupby(tables, wall, sum)
    items.sort(key=lambda item: item[1], reverse=True)
    top_tables, total_wall = zip(*items)

    p = figure(
        width=800,
        title="Wall time vs Selectivity",
        x_axis_label="Selectivity",
        y_axis_label="Elapsed wall time",
        x_axis_type="log",
        y_axis_type="log",
        sizing_mode="scale_width",
        tools=TOOLS,
    )
    p.xaxis.ticker = [10 ** (-p) for p in range(9)]
    yticks = [1e-3, 1, 60, 60 * 60, 24 * 60 * 60]
    p.yaxis.ticker = yticks
    p.yaxis.major_label_overrides = dict(zip(yticks, ["1ms", "1s", "1m", "1h", "1d"]))

    shape_size = _get_size(colorblind)
    top_tables = set(top_tables[:topK])

    I = numpy.array([(t in top_tables) for t in tables], dtype=bool)
    selectivity = selectivity[I]
    elapsed_time = wall[I]
    tables = tables[I]
    query_ids = query_ids[I]

    markers = ["circle", "diamond", "square", "triangle", "cross", "asterisk"]
    markers = itertools.cycle(markers)
    markers = itertools.islice(markers, len(top_tables))
    markers_map = dict(zip(top_tables, markers))
    markers = [markers_map[t] for t in tables]

    colors = _get_colors(colorblind)
    colors = itertools.cycle(colors)
    colors_map = dict(zip(top_tables, colors))
    colors = [colors_map[t] for t in tables]

    source = ColumnDataSource(
        dict(selectivity=selectivity, elapsed_time=elapsed_time, table_name=tables, colors=colors, markers=markers,
             copy_on_tap=query_ids))
    p.scatter("selectivity", "elapsed_time", legend_group="table_name", color="colors", marker="markers", source=source,
              size=shape_size)
    p.select(type=TapTool).callback = CustomJS(args=dict(source=source), code=COPY_JS)
    add_constant_line(p, 'height', 1e-2)
    return p


@run
def inputrows_vs_selectivity(stats, topK=5, colorblind=False):
    """
    Shows the number of input rows that were scanned and filtered from the top K tables.
    The top K tables are chosen by the fraction of wall time needed for Scan and Filter operations.
    Useful for detecting the tables where scanning can be improved by partitioning/indexing.

    Optimization Tip - Queries to the left of the dashed black line are selective queries, and can benefit significantly from predicate push-down and partitioning/indexing.
    """
    tables = []
    wall = []
    input_rows = []
    selectivity = []
    query_ids = []
    for s in stats:
        node_map = {node["id"]: node for node in nodes_from_stats(s)}
        for op in scanfilter_operators(s["operators"]):
            node_id = op["node_id"]
            node = node_map[node_id]
            scan_node = last_element(
                iter_nodes(node)
            )  # DFS to get the "lowest" table scan node
            try:
                table_name = parse_table_name(scan_node)
            except KeyError as e:
                log.error("node: {}", scan_node)
                raise
            if op["input_rows"]:
                tables.append(table_name)
                wall.append(op["input_wall"] + op["output_wall"] + op["finish_wall"])
                input_rows.append(op["input_rows"])
                selectivity.append(op["output_rows"] / op["input_rows"])
                query_ids.append(s["query_id"])

    if not tables:
        return

    tables = numpy.array(tables)
    input_rows = numpy.array(input_rows)
    selectivity = numpy.array(selectivity)
    query_ids = numpy.array(query_ids)

    items = groupby(tables, wall, sum)
    items.sort(key=lambda item: item[1], reverse=True)
    top_tables, total_wall = zip(*items)

    p = figure(
        width=800,
        title="Input rows vs Selectivity",
        x_axis_label="Selectivity",
        y_axis_label="Input rows",
        x_axis_type="log",
        y_axis_type="log",
        sizing_mode="scale_width",
        tools=TOOLS,
    )
    p.xaxis.ticker = [10 ** (-p) for p in range(9)]
    p.yaxis.ticker = [10 ** p for p in range(9)]

    shape_size = _get_size(colorblind)
    top_tables = set(top_tables[:topK])

    I = numpy.array([(t in top_tables) for t in tables], dtype=bool)
    selectivity = selectivity[I]
    input_rows = input_rows[I]
    tables = tables[I]
    query_ids = query_ids[I]

    markers = ["circle", "diamond", "square", "triangle", "cross", "asterisk"]
    markers = itertools.cycle(markers)
    markers = itertools.islice(markers, len(top_tables))
    markers_map = dict(zip(top_tables, markers))
    markers = [markers_map[t] for t in tables]

    colors = _get_colors(colorblind)
    colors = itertools.cycle(colors)
    colors_map = dict(zip(top_tables, colors))
    colors = [colors_map[t] for t in tables]

    source = ColumnDataSource(
        dict(selectivity=selectivity, input_rows=input_rows, table_name=tables, colors=colors, markers=markers,
             copy_on_tap=query_ids))
    p.scatter("selectivity", "input_rows", legend_group="table_name", color="colors", marker="markers", source=source,
              size=shape_size)
    p.select(type=TapTool).callback = CustomJS(args=dict(source=source), code=COPY_JS)
    add_constant_line(p, 'height', 1e-2)
    return p


@run
def input_size_by_table_scan(stats, topK=20):
    """
    The fraction of bytes read while scanning the top K tables.
    Useful for detecting the tables that are scanned the most.
    """
    tables = []
    input_size = []
    for s in stats:
        node_map = {node["id"]: node for node in nodes_from_stats(s)}
        for op in scan_operators(s["operators"]):
            node_id = op["node_id"]
            node = node_map[node_id]
            scan_node = last_element(
                iter_nodes(node)
            )  # DFS to get the "lowest" table scan node
            try:
                table_name = parse_table_name(scan_node)
            except KeyError as e:
                log.error("node: {}", scan_node)
                raise
            tables.append(table_name)
            input_size.append(op["input_size"])

    if not tables:
        return

    items = groupby(tables, input_size, sum)
    items.sort(key=lambda item: item[1], reverse=True)
    tables, input_size = zip(*items)
    return pie_chart(
        keys=tables, values=input_size, title="Input bytes read by table scan"
    )


@run
def operator_input(stats):
    """
    The fraction of bytes read by each Presto operator.
    Can be used to understand the percentage of bytes read by each part of the query processing - e.g. scan/join/aggreggation.
    """
    operators = [op for s in stats for op in s["operators"]]
    types = [op["type"] for op in operators]
    input_size = [op["input_size"] for op in operators]
    items = groupby(types, input_size, sum)
    items.sort(key=lambda item: item[1], reverse=True)
    types, input_size = zip(*items)
    return pie_chart(
        keys=types, values=input_size, title="Input bytes read by operator"
    )


@run
def operator_rows(stats):
    """
    The fraction of rows read by each Presto operator.
    Can be used to understand the percentage of rows read by each part of the query processing - e.g. scan/join/aggreggation.
    """
    operators = [op for s in stats for op in s["operators"]]
    types = [op["type"] for op in operators]
    input_rows = [op["input_rows"] for op in operators]
    items = groupby(types, input_rows, sum)
    items.sort(key=lambda item: item[1], reverse=True)
    types, input_rows = zip(*items)
    return pie_chart(keys=types, values=input_rows, title="Rows read by operator")


def nodes_from_stats(s):
    fragments = s["fragments"] or []
    nodes_iterators = (iter_nodes(fragment["root"]) for fragment in fragments)
    return itertools.chain.from_iterable(nodes_iterators)


def get_node_type(node):
    node_type = node["@type"]
    match = re.search(r"\.(\w+)Node$", node_type)
    if match:  # convert PrestoDB to PrestoSQL naming
        node_type = match.group(1)
    return node_type.lower()


def iter_nodes(node):
    yield node
    node_type = get_node_type(node)

    if node_type == "exchange":
        children = node["sources"]
    elif node_type == "join":
        children = [node["left"], node["right"]]
    elif node_type in {"remotesource", "tablescan", "metadatadelete", "values", "tabledelete"}:
        children = []
    else:
        try:
            children = [node["source"]]
        except KeyError:
            log.error("no 'source' at {}", node)
            raise

    for child in children:
        yield from iter_nodes(child)


def group_operators_by_nodes(operators):
    operators_map = {}
    for op in operators:
        try:
            node_id = op["node_id"]
        except KeyError:
            log.error("Missing node ID from {}", op)
            raise
        operators_map.setdefault(node_id, []).append(op)
    return operators_map


def iter_joins(stats):
    for s in stats:
        if not s["operators"]:
            continue  # usually it's a DDL or a "... LIMIT 0" query.
        operators_map = group_operators_by_nodes(s["operators"])
        for node in nodes_from_stats(s):
            node_type = get_node_type(node)
            if node_type.endswith("join"):  # join & semijoin
                try:
                    join_ops = operators_map[node["id"]]
                except KeyError:
                    log.error("{}: {} id={} has no matching operator: {!r}", s["query_id"], node_type, node["id"],
                              s["query"])
                    continue

                join_ops = {op["type"]: op for op in join_ops}
                if node_type == "join":
                    if node["criteria"] or node["type"] != "INNER":
                        keys = ("LookupJoinOperator", "HashBuilderOperator")
                    else:  # i.e full-join
                        keys = ("NestedLoopJoinOperator", "NestedLoopBuildOperator")
                elif node_type == "semijoin":
                    keys = ("HashSemiJoinOperator", "SetBuilderOperator")
                else:
                    raise ValueError(
                        "{}: unsupported join type: {}".format(s["query_id"], node)
                    )

                try:
                    probe = join_ops[keys[0]]
                    build = join_ops[keys[1]]
                except KeyError:
                    log.error(
                        "missing keys {} in {}, node: {}",
                        keys,
                        join_ops,
                        json.dumps(node, indent=4),
                    )
                    raise

                yield s, node, probe, build


@run
def joins_sides(stats, colorblind=False):
    """
    Shows the join distribution using a scatter plot.
    Each join operator is shown by a dot. The x coordinate is the data read from the
    right-side table, and the y coordinate is the data read from the left-side table.
    Replicated joins are shown in a different color than Partitioned joins.
    For optimal performance, keep the right-side smaller than the left-side (the points should be above the y=x line). Replicated joins should be used as long as the right-side table is not too large, to prevent out-of-memory errors.
    If you are using CBO, ensure the correct statistics are estimated for all tables being joined using ANALYZE command.

    Optimization Tips -
    1. Queries to the left of the black dashed line and above the orange dashed line should all use the REPLICATED join distribution type.
    2. Queries to the right of the orange dashed line perform joins with an incorrect table order. Ensure statistics are used, or rewrite the queries to flip the table sides, to boost performance and save cluster resources.

    """
    joins = list(iter_joins(stats))
    if not joins:
        return

    p = figure(
        title="Joins distribution",
        x_axis_label="Right-side data read [bytes]",
        x_axis_type="log",
        y_axis_label="Left-side data read [bytes]",
        y_axis_type="log",
        sizing_mode="scale_width",
        tools=TOOLS,
    )

    data = {}
    for stat, node, probe, build in joins:
        data.setdefault("x", []).append(build["input_size"])  # right-side
        data.setdefault("y", []).append(probe["input_size"])  # left-side
        data.setdefault("dist", []).append(node["distributionType"])
        data.setdefault("copy_on_tap", []).append(stat["query_id"])

    shape_size = _get_size(colorblind)
    color_map = {"PARTITIONED": "red", "REPLICATED": "blue"}
    marker_map = {"PARTITIONED": "circle", "REPLICATED": "square"}

    data["color"] = [color_map[d] for d in data["dist"]]
    data["marker"] = [marker_map[d] for d in data["dist"]]
    source = ColumnDataSource(data)
    p.scatter("x", "y", marker="marker", color="color", legend_group="dist", alpha=0.5, size=shape_size, source=source)
    p.select(type=TapTool).callback = CustomJS(args=dict(source=source), code=COPY_JS)
    p.legend.title = "Join distribution"
    p.xaxis.ticker = [1, 1e3, 1e6, 1e9, 1e12]
    p.yaxis.ticker = [1, 1e3, 1e6, 1e9, 1e12]
    add_constant_line(p, 'height', 1e6)
    slope = Slope(gradient=1, y_intercept=0,
                  line_color='orange', line_dash='dashed', line_width=2)

    p.add_layout(slope)
    return p


@run
def joins_selectivity(stats):
    """
    Show the joins' selectivity, using a scatter plot.
    Each join operator is illustrated by a dot, whose distance below the y=x
    line indicates how much it will benefit from Dynamic Filtering.
    Joins on y=x line are non-selective, since they require all their input rows to be processed.
    """
    joins = list(iter_joins(stats))
    if not joins:
        return

    p = figure(
        title="Joins selectivity",
        x_axis_label="Input rows = max(left, right)",
        x_axis_type="log",
        y_axis_label="Output rows",
        y_axis_type="log",
        sizing_mode="scale_width",
        tools=TOOLS,
    )

    data = []
    for stat, node, probe, build in joins:
        x = max(probe["input_rows"], build["input_rows"])
        y = probe["output_rows"]
        data.append((x, y, stat["query_id"]))
    x, y, query_ids = zip(*data)
    source = ColumnDataSource(dict(x=x, y=y, copy_on_tap=query_ids))

    p.circle("x", "y", source=source, color="green", alpha=0.5)
    p.select(type=TapTool).callback = CustomJS(args=dict(source=source), code=COPY_JS)

    p.xaxis.ticker = [1, 1e3, 1e6, 1e9, 1e12]
    p.yaxis.ticker = [1, 1e3, 1e6, 1e9, 1e12]
    return p


def collect_metrics(stats):
    cpu_days = sum(s["cpu_time"] for s in stats) / (60 * 60 * 24)
    scheduled_days = sum(s["scheduled_time"] for s in stats) / (60 * 60 * 24)
    input_rows = sum(s["input_rows"] for s in stats)
    input_TB = sum(s["input_size"] for s in stats) / 1e12
    dates = (query_datetime(s["query_id"]) for s in stats)
    n_days = len(set(trunc_date(d) for d in dates))
    n_users = len(set(s["user"] for s in stats))
    return {
        "days": n_days,
        "cpu_days": cpu_days,
        "scheduled_days": scheduled_days,
        "queries": len(stats),
        "input_rows": input_rows,
        "input_TB": input_TB,
        "users": n_users,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-i",
        "--input-file",
        type=pathlib.Path,
        help="Path to the extracted JSONL file (the output of extract.py)",
    )
    p.add_argument(
        "-o",
        "--output-file",
        type=pathlib.Path,
        default="./output.zip",
        help="Path to the resulting zipped HTML report",
    )
    p.add_argument("-l", "--limit", type=int)
    p.add_argument("--filter", type=str)
    p.add_argument("--fail-on-error", action="store_true", default=False)
    p.add_argument("--high-contrast-mode", action="store_true", default=False)
    p.add_argument("-q", "--quiet", action="store_true", default=False)
    args = p.parse_args()

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
    stats = []
    for line in tqdm(lines, unit="files", disable=args.quiet):
        s = json.loads(line)
        if s["state"] == "FAILED":
            continue
        stats.append(s)
    log.info("{} queries loaded", len(stats))

    metrics = collect_metrics(stats)
    charts = []
    scripts = []
    for func in tqdm(_ANALYZERS, unit="graphs", disable=args.quiet):
        graph_id = func.__name__
        if args.filter is None or args.filter == func.__name__:
            try:
                if 'colorblind' in signature(func).parameters:
                    p = func(stats, colorblind=args.high_contrast_mode)
                else:
                    p = func(stats)

                if p is None:
                    log.warn("not enough data for {}", graph_id)
                    continue
                item = json_item(model=p, target=graph_id)
                item["doc"]["roots"]["references"].sort(key=lambda r: (r["type"], r["id"]))

                item = json.dumps({"doc": item["doc"]})
                scripts.append(
                    '<script type="application/json" id="{}">\n{}\n</script>\n'.format(
                        graph_id, item
                    )
                )
                charts.append(
                    {
                        "title": p.title.text or graph_id,
                        "description": func.__doc__.strip(),
                        "id": graph_id,
                    }
                )
            except Exception:
                log.exception("failed to generate {}", graph_id)
                if args.fail_on_error:
                    raise

    scripts.append(
        "<script>\nconst structure = {}</script>".format(
            json.dumps({"metrics": metrics, "charts": charts}, indent=4)
        )
    )

    template = pathlib.Path(__file__).parent / "output.template.html"
    placeholder = "<!-- PLACEHOLDER_FOR_BOKEH_JSONS -->"
    output = template.open().read().replace(placeholder, "\n".join(scripts))
    log.info("report is written to {}", args.output_file)
    suffix = args.output_file.suffix
    if suffix == ".zip":
        with zipfile.ZipFile(args.output_file, "w") as f:
            f.writestr("output.html", data=output, compress_type=zipfile.ZIP_DEFLATED)
    elif suffix == ".html":
        with open(args.output_file, "w") as f:
            f.write(output)
    else:
        raise ValueError("Unsupport output file extension: {}".format(args.output_file))


if __name__ == "__main__":
    main()
