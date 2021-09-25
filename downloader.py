import argparse
import asyncio
import datetime
import json
import os
import sys
import time

import aiohttp
import pandas as pd
from dateutil import parser, tz


class Downloader:
    def __init__(self, symbol: str, interval: str, path: str):
        self.session = aiohttp.ClientSession()
        self.base_endpoint = 'https://fapi.binance.com'
        self.endpoint = '/fapi/v1/continuousKlines'
        self.symbol = symbol
        self.interval = interval
        self.path = path
        self.filename = os.path.join(self.path, self.symbol + '-' + self.interval + '.csv')

    async def download_klines(self, symbol: str, interval: str, start_time: int = None,
                              end_time: int = None) -> pd.DataFrame:
        params = {'pair': symbol, 'contractType': 'PERPETUAL', 'interval': interval, 'limit': 999}
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        async with self.session.get(self.base_endpoint + self.endpoint, params=params) as response:
            result = await response.text()
            r = json.loads(result)
            return pd.DataFrame(r, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                            'Quote asset volume', 'Number of trades', 'Taker buy volume',
                                            'Taker buy quote asset volume', 'Ignore'])

    def timestamp_to_time(self, timestamp):
        return datetime.datetime.utcfromtimestamp(timestamp // 1000).strftime('%Y-%m-%d %H:%M:%S')

    async def get_klines(self):
        if os.path.isfile(self.filename):
            print('Reading existing file...')
            df = pd.read_csv(self.filename)
            start_time = int(df['Close time'].iloc[-1] + 1)
        else:
            if not os.path.isdir(self.path):
                os.makedirs(self.path, exist_ok=True)
            df = pd.DataFrame(
                columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
                         'Number of trades', 'Taker buy volume', 'Taker buy quote asset volume', 'Ignore'])
            start_time = int(
                parser.parse('2019-01-01 00:00:00').replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
        while int(datetime.datetime.now(tz.UTC).timestamp() * 1000) - start_time > 60000:
            print(datetime.datetime.now(), 'Fetching', self.symbol, 'from', self.timestamp_to_time(start_time))
            start = time.time()
            candles = await self.download_klines(self.symbol, self.interval, start_time=start_time)
            df = pd.concat([df, candles])
            df.sort_values('Open time', inplace=True)
            df.drop_duplicates('Open time', inplace=True)
            start_time = int(df['Close time'].iloc[-1])
            if start_time > int(datetime.datetime.now(tz.UTC).timestamp() * 1000):
                print('Cutting of last', start_time)
                df = df[df['Close time'] < start_time]
            wait_time = max(0.0, 0.25 - time.time() - start)
            await asyncio.sleep(wait_time)
        print('Saving file...')
        df.to_csv(self.filename, index=False)
        try:
            await self.session.close()
        except:
            pass

    async def get_frame(self) -> pd.DataFrame:
        df = pd.read_csv(self.filename)
        return df


async def main(args: list):
    argparser = argparse.ArgumentParser(prog='Downloader', add_help=True,
                                        description='Downloads kline data for symbol on Binance Futures.')
    argparser.add_argument('-s', '--symbol', type=str, required=True, dest='s',
                           help='Symbol pair to download data for.')
    argparser.add_argument('-i', '--interval', nargs='*', type=str, required=True, dest='i',
                           help='Interval of the candles in the form 1m, 4h, etc.')
    argparser.add_argument('-p', '--path', type=str, required=False, dest='p', default='data',
                           help='Path to save data to.')
    args = argparser.parse_args()
    intervals = args.i
    symbol = args.s
    path = args.p
    for interval in intervals:
        downloader = Downloader(symbol, interval, path)
        await downloader.get_klines()
    # Needed to avoid getting an event loop error when exiting.
    time.sleep(0.1)


if __name__ == '__main__':
    asyncio.run(main(sys.argv))
