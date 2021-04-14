import argparse
import asyncio
import json
import os
import sys

import hjson
import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from talib import *

from downloader import Downloader


def read_config(config_path: str) -> dict:
    config = hjson.load(open(config_path + '.hjson'))
    return config


def calc_cross_long_liq_price(balance, pos_size, pos_price, leverage, mm=0.004) -> float:
    d = (pos_size * mm - pos_size)
    if d == 0.0:
        return 0.0
    return (balance - pos_size * pos_price) / d


def calc_cross_shrt_liq_price(balance, pos_size, pos_price, leverage, mm=0.004) -> float:
    abs_pos_size = abs(pos_size)
    d = (abs_pos_size * mm - pos_size)
    if d == 0.0:
        return 0.0
    return (balance - pos_size * pos_price) / d


def backtest(config: dict, ohlc: np.ndarray, return_results: bool = False):
    balance = config['starting_balance']
    period_fast = config['ema_fast']
    period_slow = config['ema_slow']
    warmup = max(period_slow, period_fast) * 8
    leverage = config['leverage']
    min_size = config['min_size']
    taker_fee = config['taker_fee']
    pos = 0
    pos_price = 0
    # pos_cost = 0
    closes = []
    prev_ema_fast = 0
    prev_ema_slow = 0
    size = min_size * leverage
    buy = False
    sell = False
    liquidation_price = 0
    result = []
    # Open, High, Low, Close, Close time
    for index in range(len(ohlc)):
        row = ohlc[index]
        if index % 1000 == 0:
            print(index, row[4])
        closes.append(row[3])
        tmp = np.asarray(closes)
        ema_fast = EMA(tmp, timeperiod=period_fast)[-1]
        ema_slow = EMA(tmp, timeperiod=period_slow)[-1]

        if index >= warmup:
            if buy and balance > 0:
                if pos == 0:
                    pos_price = row[0]
                    pos = size
                    pos_cost = pos * pos_price / leverage
                    fee = (pos * pos_price) * taker_fee
                    balance -= fee
                    liquidation_price = calc_cross_long_liq_price(balance, pos, pos_price, leverage,
                                                                  taker_fee)
                    liquidation_price -= 0.0065 * liquidation_price
                elif pos < 0:
                    pnl = abs(pos) * (pos_price - row[0])
                    fee = (abs(pos) * row[0]) * taker_fee
                    balance -= fee
                    balance += pnl

                    pos_price = row[0]
                    pos = size
                    pos_cost = pos * pos_price / leverage
                    fee = (pos * pos_price) * taker_fee
                    balance -= fee
                    liquidation_price = calc_cross_long_liq_price(balance, pos, pos_price, leverage,
                                                                  taker_fee)
                    liquidation_price -= 0.0065 * liquidation_price
                buy = False
            if sell and balance > 0:
                if pos == 0:
                    pos_price = row[0]
                    pos = -size
                    pos_cost = abs(pos) * pos_price / leverage
                    fee = (abs(pos) * pos_price) * taker_fee
                    balance -= fee
                    liquidation_price = calc_cross_shrt_liq_price(balance, pos, pos_price, leverage,
                                                                  taker_fee)
                    liquidation_price += 0.0065 * liquidation_price
                elif pos > 0:
                    pnl = abs(pos) * (row[0] - pos_price)
                    fee = (abs(pos) * row[0]) * taker_fee
                    balance -= fee
                    balance += pnl

                    pos_price = row[0]
                    pos = -size
                    pos_cost = abs(pos) * pos_price / leverage
                    fee = (abs(pos) * pos_price) * taker_fee
                    balance -= fee
                    liquidation_price = calc_cross_shrt_liq_price(balance, pos, pos_price, leverage,
                                                                  taker_fee)
                    liquidation_price += 0.0065 * liquidation_price
                sell = False

            if ema_fast > ema_slow and prev_ema_fast < prev_ema_slow:
                buy = True
            if ema_fast < ema_slow and prev_ema_fast > prev_ema_slow:
                sell = True
        if buy:
            action = 'Buy'
        elif sell:
            action = 'Sell'
        else:
            action = 'Hold'

        if pos > 0 and row[2] < liquidation_price:
            pnl = abs(pos) * (row[2] - pos_price)
            fee = (abs(pos) * row[2]) * taker_fee
            balance -= fee
            balance += pnl
            pos_price = 0
            pos = 0
            pos_cost = 0
            print('Long liquidated', index, pos_price, liquidation_price, row[2], balance)
            break
        if pos < 0 and row[1] > liquidation_price:
            pnl = abs(pos) * (pos_price - row[1])
            fee = (abs(pos) * row[1]) * taker_fee
            balance -= fee
            balance += pnl
            pos_price = 0
            pos = 0
            pos_cost = 0
            print('Short liquidated', index, pos_price, liquidation_price, row[1], balance)
            break
        if pos < 0:
            pnl = abs(pos) * (pos_price - row[3])
        elif pos > 0:
            pnl = abs(pos) * (row[3] - pos_price)
        else:
            pnl = 0
        close_liq = 1
        if pos < 0:
            close_liq = (liquidation_price - row[1]) / row[1]
        if pos > 0:
            close_liq = (liquidation_price - row[2]) / row[2]
        d = {'balance': balance, 'pnl': pnl, 'Close time': row[4], 'action': action, 'closest_liq': close_liq}

        result.append(d)

        prev_ema_fast = ema_fast
        prev_ema_slow = ema_slow
    result = pd.DataFrame(result)
    if return_results:
        return result
    else:
        tune.report(objective=objective_function(result))


def objective_function(result: pd.DataFrame) -> float:
    if not result.empty:
        gain = ((result['balance'].iloc[-1] + result['pnl'].iloc[-1]) - (
                result['balance'].iloc[0] + result['pnl'].iloc[0]))
        days = (result['Close time'].iloc[-1] - result['Close time'].iloc[0]) / (1000 * 60 * 60 * 24)
        return gain / days * result['closest_liq'].min()
    else:
        return -1000


def create_config(backtest_config: dict) -> dict:
    config = {}
    config['balance'] = backtest_config['starting_balance']
    config['min_size'] = backtest_config['min_size']
    config['taker_fee'] = backtest_config['taker_fee']

    # config['qty_pct'] = tune.quniform(backtest_config['ranges']['qty_pct'][0],
    #                                   backtest_config['ranges']['qty_pct'][1],
    #                                   backtest_config['ranges']['qty_pct'][2])

    config['leverage'] = tune.qrandint(backtest_config['ranges']['leverage'][0],
                                       backtest_config['ranges']['leverage'][1],
                                       backtest_config['ranges']['leverage'][2])
    config['ema_fast'] = tune.qrandint(backtest_config['ranges']['ema_fast'][0],
                                       backtest_config['ranges']['ema_fast'][1],
                                       backtest_config['ranges']['ema_fast'][2])
    config['ema_slow'] = tune.qrandint(backtest_config['ranges']['ema_slow'][0],
                                       backtest_config['ranges']['ema_slow'][1],
                                       backtest_config['ranges']['ema_slow'][2])
    return config


def backtest_tune(ohlc: np.ndarray, backtest_config: dict):
    config = create_config(backtest_config)
    if not os.path.isdir(os.path.join('reports', backtest_config['symbol'])):
        os.makedirs(os.path.join('reports', backtest_config['symbol']), exist_ok=True)
    session_dirpath = os.path.join('reports', backtest_config['symbol'])
    iters = 10
    if 'iters' in backtest_config:
        iters = backtest_config['iters']
    else:
        print('Parameter iters should be defined in the configuration. Defaulting to 10.')
    num_cpus = 2
    if 'num_cpus' in backtest_config:
        num_cpus = backtest_config['num_cpus']
    else:
        print('Parameter num_cpus should be defined in the configuration. Defaulting to 2.')

    initial_points = max(1, min(int(iters / 10), 20))

    ray.init(num_cpus=num_cpus)

    algo = HyperOptSearch(n_initial_points=initial_points)
    algo = ConcurrencyLimiter(algo, max_concurrent=num_cpus)
    scheduler = AsyncHyperBandScheduler()

    analysis = tune.run(tune.with_parameters(backtest, ohlc=ohlc), metric='objective', mode='max', name='search',
                        search_alg=algo, scheduler=scheduler, num_samples=iters, config=config, verbose=1,
                        reuse_actors=True, local_dir=session_dirpath)

    ray.shutdown()
    session_dir = os.path.join('sessions', config['session_name'])
    if not os.path.isdir(session_dir):
        os.makedirs(session_dir, exist_ok=True)

    print('Best candidate found is: ', analysis.best_config)
    json.dump(analysis.best_config, open(os.path.join(session_dir, 'best_config.json'), 'w'), indent=4)
    result = backtest(analysis.best_config, ohlc, True)
    result.to_csv(os.path.join(session_dir, 'best_trades.csv'), index=False)
    return analysis


async def main(args: list):
    argparser = argparse.ArgumentParser(prog='Backtester', add_help=True,
                                        description='Backtests and optimizes a strategy.')
    argparser.add_argument('-c', '--config', type=str, required=True, dest='c',
                           help='Configuration to test.')
    args = argparser.parse_args()
    config = args.c
    config = read_config(os.path.join('configs', config))
    downloader = Downloader(config['symbol'], config['interval'], config['path'])
    await downloader.get_klines()
    ohlc = await downloader.get_frame()
    ohlc = ohlc[['Open', 'High', 'Low', 'Close', 'Close time']].values
    backtest_tune(ohlc, config)


if __name__ == '__main__':
    asyncio.run(main(sys.argv))
