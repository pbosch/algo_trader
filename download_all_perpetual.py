import argparse
import asyncio
import json
import sys

import aiohttp

from downloader import Downloader


async def get_all_symbols() -> list:
    session = aiohttp.ClientSession()
    base_endpoint = 'https://fapi.binance.com'
    endpoint = '/fapi/v1/exchangeInfo'
    async with session.get(base_endpoint + endpoint, params={}) as response:
        result = await response.text()
        r = json.loads(result)
    symbols = []
    for i in r['symbols']:
        if i['contractType'] == 'PERPETUAL':
            symbols.append(i['symbol'])
    return symbols


async def main(args: list):
    argparser = argparse.ArgumentParser(prog='DownloadAllPerpetual', add_help=True,
                                        description='Downloads kline data for all symbols on Binance Futures with specified interval.')
    argparser.add_argument('-i', '--interval', nargs='*', type=str, required=True, dest='i',
                           help='Interval of the candles in the form 1m, 4h, etc.')
    argparser.add_argument('-p', '--path', type=str, required=False, dest='p', default='data',
                           help='Path to save data to.')
    args = argparser.parse_args()
    intervals = args.i
    path = args.p
    symbols = await get_all_symbols()
    for interval in intervals:
        for symbol in symbols:
            downloader = Downloader(symbol, interval, path)
            await downloader.get_klines()


if __name__ == '__main__':
    asyncio.run(main(sys.argv))
