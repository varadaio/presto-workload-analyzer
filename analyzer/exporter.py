import argparse
import asyncio
import sys
from asyncio import Queue
from time import sleep

import logbook
from prometheus_client import start_http_server, Summary

logbook.StreamHandler(sys.stderr).push_application()
log = logbook.Logger("exporter")


class PrestoWorkloadAnalyzerExporter:
    def __init__(self, port: int = 9877):
        self.exporter_port = port
        self.metrics = {}

        self._prefix = 'presto_workload_analyzer'

        self.metrics_map = {
            # 'info': {
            #     'cls': Info,
            #     'fields': {'main'},
            #     'dimensions': []
            # },
            'summary': {
                'cls': Summary,
                'fields': {'elapsed_time', 'cpu_time', 'scheduled_time', 'blocked_time', 'input_size', 'output_size',
                           'network_size', 'input_rows', 'output_rows', 'network_rows'}
            }
        }

        [self._init_metrics(x) for x in list(self.metrics_map.keys())]

        start_http_server(self.exporter_port)

    def _get_metric_name(self, name):
        return f'{self._prefix}_{name}'

    """
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
    """

    def _init_metrics(self, type):
        self.metrics[type] = {}
        metric_class = self.metrics_map[type].get('cls')
        fields = self.metrics_map[type].get('fields')

        for field in fields:
            metric_name = self._get_metric_name(field)

            self.metrics[type][metric_name] = metric_class(metric_name, metric_name, ['state'])

    def _refresh_metrics(self, payload):
        state = payload['state']
        for field, value in payload.items():
            for metric_type, metric_spec in self.metrics_map.items():
                if field in metric_spec['fields']:
                    metric_name = self._get_metric_name(field)

                    self.metrics[metric_type][metric_name].labels(state).observe(value)

    async def consumer(self, queue):
        log.info('consumer started')
        while True:
            try:
                payload = await queue.get()
                self._refresh_metrics(payload)
                log.debug(f'received: {payload}')
            except Exception:
                log.warning('error retrieved', exc_info=True)

    async def producer(self, queue, generator):
        log.info('producer started')
        while True:
            try:
                payload = next(generator)
                await queue.put(payload)
                log.debug(f'sent: {payload}')
                await asyncio.sleep(1)
            except StopIteration:
                log.warning('producer stopped', exc_info=True)
                break

    async def run(self, i):
        q = Queue()

        asyncio.create_task(self.consumer(q), name='query consumer')
        producer = asyncio.create_task(self.producer(q, i), name='query iterable')

        await asyncio.gather(producer)


def get_args_parser():
    from analyzer.collect import get_args_parser as get_collect_args_parser

    collect_args_parser = get_collect_args_parser()

    p = argparse.ArgumentParser(parents=[collect_args_parser], add_help=False)

    group = p.add_argument_group('exporter')
    group.add_argument("--port", type=int, default=9877)

    return p


def test():
    while True:
        """
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
        
        """
        yield dict(
            query='j["query"]',
            query_id='j["queryId"]',
            user='mariusz',
            state='completed',
            error_code='1001',
            update='zxc',
            elapsed_time=50000,
            cpu_time=40000,
            scheduled_time=30000,
            blocked_time=20000,
            input_size=(2137),
            output_size=21370,
            network_size=213700,
            input_rows=2137000,
            output_rows=21370000,
            network_rows=213700000,
            peak_mem=20,
            written_size=30,
            # operators=list(get_operators(stats["operatorSummaries"])),
            inputs=10,
            output=20,
        )

        sleep(10)


def main():
    args = get_args_parser().parse_args()

    e = PrestoWorkloadAnalyzerExporter(port=args.port)

    generator = test()

    asyncio.run(e.run(generator))


if __name__ == '__main__':
    main()
