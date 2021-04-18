import argparse
import asyncio
import json
import os
import sys

import hjson
import numpy as np
import pandas as pd
import ray
import talib as ta
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch

from downloader import Downloader
from utils import calc_cross_long_liq_price, calc_cross_shrt_liq_price, round_dn, calc_diff


def read_config(config_path: str) -> dict:
    config = hjson.load(open(config_path + '.hjson'))
    return config


def decide(config: dict, ohlc_view: np.ndarray, warmup: int) -> (bool, bool):
    ema_fast = ta.EMA(ohlc_view[-warmup:, 3], timeperiod=config['ema_fast'])
    ema_slow = ta.EMA(ohlc_view[-warmup:, 3], timeperiod=config['ema_slow'])
    buy = False
    sell = False
    if ema_fast[-1] > ema_slow[-1] and ema_fast[-2] < ema_slow[-2]:
        buy = True
    if ema_fast[-1] < ema_slow[-1] and ema_fast[-2] > ema_slow[-2]:
        sell = True
    return buy, sell


def do_buy(balance: float, pos_price: float, pos_size: float, pos_cost: float, price: float, taker_fee: float,
           qty_pct: float, min_step: float, leverage: int, slippage: float):
    full_fee = 0
    if pos_size < 0:
        pnl = abs(pos_size) * (pos_price - price)
        fee = (abs(pos_size) * price) * taker_fee
        balance -= fee
        balance += pnl + pos_cost
        full_fee += fee
    pos_price = price * (1 + slippage)
    pos_size = round_dn(qty_pct * balance * leverage / pos_price, min_step)
    pos_cost = pos_size * pos_price / leverage
    fee = (pos_size * pos_price) * taker_fee
    full_fee += fee
    balance -= (fee + pos_cost)
    liquidation_price = calc_cross_long_liq_price(balance, pos_size, pos_price, taker_fee)
    liquidation_price -= 0.0065 * liquidation_price
    return balance, pos_price, pos_size, pos_cost, full_fee, liquidation_price


def do_sell(balance: float, pos_price: float, pos_size: float, pos_cost: float, price: float, taker_fee: float,
            qty_pct: float, min_step: float, leverage: int, slippage: float):
    full_fee = 0
    if pos_size > 0:
        pnl = abs(pos_size) * (price - pos_price)
        fee = (abs(pos_size) * price) * taker_fee
        balance -= fee
        balance += pnl + pos_cost
        full_fee += fee
    pos_price = price * (1 - slippage)
    pos_size = -round_dn(qty_pct * balance * leverage / pos_price, min_step)
    pos_cost = abs(pos_size) * pos_price / leverage
    fee = (abs(pos_size) * pos_price) * taker_fee
    full_fee += fee
    balance -= (fee + pos_cost)
    liquidation_price = calc_cross_shrt_liq_price(balance, pos_size, pos_price, taker_fee)
    liquidation_price += 0.0065 * liquidation_price
    return balance, pos_price, pos_size, pos_cost, full_fee, liquidation_price


def backtest(config: dict, ohlc: np.ndarray, return_results: bool = False):
    balance = config['starting_balance']
    period_fast = config['ema_fast']
    period_slow = config['ema_slow']
    warmup = max(period_slow, period_fast) * 10
    leverage = config['leverage']
    min_size = config['min_size']
    min_step = config['min_step']
    qty_pct = config['qty_pct']
    taker_fee = config['taker_fee']
    slippage = 0.001
    pos_size = 0
    pos_price = 0
    pos_cost = 0
    buy = False
    sell = False
    liquidation_price = 0
    result = []
    # Open, High, Low, Close, Close time
    for index in range(len(ohlc)):
        row = ohlc[index]

        trade = {'balance': balance,
                 'current_pnl': 0.0,
                 'current_pos_cost': abs(pos_size) * pos_price / leverage,
                 'fee': 0.0,
                 'Close time': row[4],
                 'action': 'Hold',
                 'closest_liq': 1.0}
        if index >= warmup:
            if pos_size < 0:
                trade['closest_liq'] = calc_diff(liquidation_price, row[1])
                trade['current_pnl'] = abs(pos_size) * (pos_price - row[3])
            elif pos_size > 0:
                trade['closest_liq'] = calc_diff(liquidation_price, row[2])
                trade['current_pnl'] = abs(pos_size) * (row[3] - pos_price)
            if buy and balance > 0:
                balance, pos_price, pos_size, pos_cost, fee, liquidation_price = do_buy(balance, pos_price, pos_size,
                                                                                        pos_cost, row[0], taker_fee,
                                                                                        qty_pct, min_step, leverage,
                                                                                        slippage)
                buy = False
                trade['balance'] = balance
                trade['current_pos_cost'] = pos_cost
                trade['fee'] = fee
                trade['current_pnl'] = abs(pos_size) * (row[3] - pos_price)
                trade['closest_liq'] = (liquidation_price - row[2]) / row[2]
                trade['action'] = 'Buy'
            if sell and balance > 0:
                balance, pos_price, pos_size, pos_cost, fee, liquidation_price = do_sell(balance, pos_price, pos_size,
                                                                                         pos_cost, row[0], taker_fee,
                                                                                         qty_pct, min_step, leverage,
                                                                                         slippage)

                sell = False
                trade['balance'] = balance
                trade['current_pos_cost'] = pos_cost
                trade['fee'] = fee
                trade['current_pnl'] = abs(pos_size) * (pos_price - row[3])
                trade['closest_liq'] = (liquidation_price - row[1]) / row[1]
                trade['action'] = 'Sell'

            buy, sell = decide(config, ohlc[:index + 1], warmup)

        if pos_size > 0 and row[2] <= liquidation_price:
            pnl = abs(pos_size) * (row[2] - pos_price)
            fee = (abs(pos_size) * row[2]) * taker_fee
            balance -= fee
            balance += pnl
            pos_price = 0
            pos = 0
            pos_cost = 0
            # print('Long liquidated', index, pos_price, liquidation_price, row[2], balance)
            break
        if pos_size < 0 and row[1] > liquidation_price:
            pnl = abs(pos_size) * (pos_price - row[1])
            fee = (abs(pos_size) * row[1]) * taker_fee
            balance -= fee
            balance += pnl
            pos_price = 0
            pos = 0
            pos_cost = 0
            # print('Short liquidated', index, pos_price, liquidation_price, row[1], balance)
            break

        if trade:
            result.append(trade)
    result = pd.DataFrame(result)
    if return_results:
        return result
    else:
        tune.report(objective=objective_function(result))


def objective_function(result: pd.DataFrame) -> float:
    if not result.empty:
        gain = ((result['balance'].iloc[-1] + result['current_pnl'].iloc[-1] + result['current_pos_cost'].iloc[-1]) - (
                result['balance'].iloc[0] + result['current_pnl'].iloc[0] + result['current_pos_cost'].iloc[0]))
        days = (result['Close time'].iloc[-1] - result['Close time'].iloc[0]) / (1000 * 60 * 60 * 24)
        return gain / days * result['closest_liq'].min()
    else:
        return 0.0


def create_config(backtest_config: dict) -> dict:
    config = {}
    config['starting_balance'] = backtest_config['starting_balance']
    config['min_size'] = backtest_config['min_size']
    config['min_step'] = backtest_config['min_step']
    config['taker_fee'] = backtest_config['taker_fee']

    config['qty_pct'] = tune.uniform(backtest_config['ranges']['qty_pct'][0], backtest_config['ranges']['qty_pct'][1])

    config['leverage'] = tune.randint(backtest_config['ranges']['leverage'][0],
                                      backtest_config['ranges']['leverage'][1])
    config['ema_fast'] = tune.randint(backtest_config['ranges']['ema_fast'][0],
                                      backtest_config['ranges']['ema_fast'][1])
    config['ema_slow'] = tune.randint(backtest_config['ranges']['ema_slow'][0],
                                      backtest_config['ranges']['ema_slow'][1])
    return config


def backtest_tune(ohlc: np.ndarray, backtest_config: dict):
    config = create_config(backtest_config)
    if not os.path.isdir(os.path.join('reports', backtest_config['symbol'])):
        os.makedirs(os.path.join('reports', backtest_config['symbol']), exist_ok=True)
    report_path = os.path.join('reports', backtest_config['symbol'])
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

    ray.init(num_cpus=num_cpus)  # , logging_level=logging.FATAL, log_to_driver=False)

    algo = HyperOptSearch(n_initial_points=initial_points)
    algo = ConcurrencyLimiter(algo, max_concurrent=num_cpus)
    scheduler = AsyncHyperBandScheduler()

    analysis = tune.run(tune.with_parameters(backtest, ohlc=ohlc), metric='objective', mode='max', name='search',
                        search_alg=algo, scheduler=scheduler, num_samples=iters, config=config, verbose=1,
                        reuse_actors=True, local_dir=report_path)

    ray.shutdown()
    session_path = os.path.join(os.path.join('sessions', backtest_config['symbol']), backtest_config['session_name'])
    if not os.path.isdir(session_path):
        os.makedirs(session_path, exist_ok=True)

    print('Best candidate found is: ', analysis.best_config)
    json.dump(analysis.best_config, open(os.path.join(session_path, 'best_config.json'), 'w'), indent=4)
    result = backtest(analysis.best_config, ohlc, True)
    result.to_csv(os.path.join(session_path, 'best_trades.csv'), index=False)
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
