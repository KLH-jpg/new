#!/usr/bin/env python3
"""
backtest_forex.py

Backtests the 5/20 SMA crossover strategy on EUR/USD hourly candles,
simulating Deriv Multipliers-style trades: percentage stop-loss / take-profit,
percentage position sizing, a daily loss circuit breaker, and a cooldown
after consecutive losses.

Data source: Deriv's public WebSocket API (ticks_history). Historical candle
data for major forex pairs does not require an API token -- only an app_id.

Usage:
    python backtest_forex.py                  # last 2 years, default
    python backtest_forex.py --years 1
    python backtest_forex.py --years 3 --outdir results

READ THIS BEFORE TRUSTING THE OUTPUT:

1. Same-bar ambiguity. Hourly candles only give open/high/low/close -- not
   the path price took inside the hour. If a candle's range touches both
   the stop-loss and take-profit, we can't know which happened first. This
   script assumes the WORSE outcome (stop-loss) in that case, so it will
   not systematically flatter the strategy. Real results could occasionally
   be a little better than this, not worse, because of that assumption.

2. Costs are approximated, not measured. SPREAD_PCT below is an estimate,
   not a live quote from Deriv. Check what frxEURUSD Multipliers actually
   spread at on your Deriv account and adjust the constant if it's off --
   this one input meaningfully changes whether a marginal strategy looks
   profitable or not. No overnight/weekend financing charge is modelled at
   all; if the live bot holds positions across a weekend, real costs will
   run higher than this backtest shows.

3. This is one strategy on one historical window. A strategy that looks
   good here could be a real edge, or could just be this particular slice
   of history -- there's no way to tell the two apart from a single
   backtest. If the numbers look promising, the next honest step is
   forward-testing on the demo account for real weeks, not going live off
   this alone.

Keep FAST_MA, SLOW_MA, STAKE_PCT, SL_PCT, TP_PCT, DAILY_LOSS_LIMIT_PCT,
LOSS_STREAK_FOR_COOLDOWN and COOLDOWN_CHECKS in sync with bot_check.py --
if they drift apart, this is backtesting a different bot than the one
that's actually live.
"""

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import websockets
except ImportError:
    print("Missing dependency 'websockets'. Install it with:\n    pip install websockets", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Strategy / risk config
# ---------------------------------------------------------------------------
SYMBOL = "frxEURUSD"
GRANULARITY = 3600              # 1 hour, in seconds
FAST_MA = 5
SLOW_MA = 20
STAKE_PCT = 0.02                # 2% of balance risked per trade
SL_PCT = 0.006                  # 0.6% stop-loss
TP_PCT = 0.012                  # 1.2% take-profit (1:2 R:R)
DAILY_LOSS_LIMIT_PCT = 0.05     # 5% daily loss circuit breaker
LOSS_STREAK_FOR_COOLDOWN = 3
COOLDOWN_CHECKS = 5
STARTING_BALANCE = 1000.0       # arbitrary; everything that matters is reported in %
SPREAD_PCT = 0.0002             # ~2 pips on EUR/USD -- ESTIMATE, verify against your Deriv account
TREND_FILTER_MA = 100           # regime filter period, in hours (~4 days) -- a starting hypothesis, not a tuned answer

APP_ID = 1089                   # Deriv's public demo app_id -- fine for unauthenticated candle history
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
MAX_CANDLES_PER_REQUEST = 5000


@dataclass
class Candle:
    epoch: int
    open: float
    high: float
    low: float
    close: float

    @property
    def dt(self):
        return datetime.fromtimestamp(self.epoch, tz=timezone.utc)

    @property
    def date(self):
        return self.dt.date()


@dataclass
class Trade:
    direction: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    exit_reason: str
    risk_amount: float
    pnl: float
    balance_after: float


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

async def fetch_candles(years: float) -> list:
    """Pull hourly EUR/USD candles from Deriv, paging backward until we have
    `years` worth of history (or Deriv's history genuinely runs out first).

    IMPORTANT: a single page can legitimately come back with fewer than
    MAX_CANDLES_PER_REQUEST candles without that meaning history is
    exhausted -- so that is NOT used as a stop condition here. The only
    things that stop the loop are: an empty response (Deriv has nothing
    earlier), a page that adds no new candles (we're not making progress,
    which would otherwise infinite-loop), enough span already covered, or
    a hard safety cap on page count.
    """
    seconds_needed = int(years * 365.25 * 24 * 3600)
    all_candles = {}
    end = "latest"
    max_pages = 40  # safety cap so a bug can't loop forever / hammer the API

    async with websockets.connect(DERIV_WS_URL) as ws:
        for page in range(1, max_pages + 1):
            request = {
                "ticks_history": SYMBOL,
                "adjust_start_time": 1,
                "count": MAX_CANDLES_PER_REQUEST,
                "end": end,
                "start": 1,
                "style": "candles",
                "granularity": GRANULARITY,
            }
            await ws.send(json.dumps(request))
            response = json.loads(await ws.recv())

            if "error" in response:
                raise RuntimeError(f"Deriv API error: {response['error'].get('message')}")

            candles = response.get("candles", [])
            if not candles:
                print(f"  page {page}: empty response -- reached the earliest history Deriv has")
                break

            before = len(all_candles)
            for c in candles:
                all_candles[c["epoch"]] = Candle(
                    epoch=c["epoch"], open=float(c["open"]), high=float(c["high"]),
                    low=float(c["low"]), close=float(c["close"]),
                )
            added = len(all_candles) - before

            overall_min = min(all_candles)
            overall_max = max(all_candles)
            span_days = (overall_max - overall_min) / 86400
            print(f"  page {page}: got {len(candles)} candles ({added} new), "
                  f"covering {span_days:.1f} days back to {datetime.fromtimestamp(overall_min, tz=timezone.utc).date()}")

            if added == 0:
                print(f"  page {page}: no new candles added -- stopping to avoid looping in place")
                break
            if (overall_max - overall_min) >= seconds_needed:
                break
            end = overall_min - 1  # page further back in time

    ordered = sorted(all_candles.values(), key=lambda c: c.epoch)
    cutoff = ordered[-1].epoch - seconds_needed
    return [c for c in ordered if c.epoch >= cutoff]


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def sma(values, period):
    out = [None] * len(values)
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(candles, use_trend_filter: bool = False) -> dict:
    closes = [c.close for c in candles]
    fast = sma(closes, FAST_MA)
    slow = sma(closes, SLOW_MA)
    trend = sma(closes, TREND_FILTER_MA)

    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    max_drawdown_pct = 0.0
    equity_curve = [balance]

    position = None
    trades = []
    consecutive_losses = 0
    cooldown_remaining = 0
    current_date = None
    day_start_balance = balance
    day_pl = 0.0
    breaker_tripped_today = False
    breaker_trip_count = 0
    cooldown_trigger_count = 0

    n = len(candles)
    for i in range(SLOW_MA + 1, n):
        c = candles[i]

        # -- day rollover (UTC) ---------------------------------------------
        if c.date != current_date:
            current_date = c.date
            day_start_balance = balance
            day_pl = 0.0
            breaker_tripped_today = False

        # -- manage an open position first -----------------------------------
        if position is not None:
            if position["direction"] == "long":
                sl_hit = c.low <= position["sl"]
                tp_hit = c.high >= position["tp"]
            else:
                sl_hit = c.high >= position["sl"]
                tp_hit = c.low <= position["tp"]

            exit_price = reason = None
            if sl_hit and tp_hit:
                exit_price, reason = position["sl"], "sl"   # ambiguous bar -> assume the worse outcome
            elif sl_hit:
                exit_price, reason = position["sl"], "sl"
            elif tp_hit:
                exit_price, reason = position["tp"], "tp"

            if exit_price is not None:
                if reason == "sl":
                    pnl = -position["risk_amount"]
                else:
                    pnl = position["risk_amount"] * (TP_PCT / SL_PCT)
                spread_cost = position["risk_amount"] * (SPREAD_PCT / SL_PCT)
                pnl -= spread_cost

                balance += pnl
                day_pl += pnl
                equity_curve.append(balance)
                peak_balance = max(peak_balance, balance)
                if peak_balance > 0:
                    max_drawdown_pct = max(max_drawdown_pct, (peak_balance - balance) / peak_balance)

                trades.append(Trade(
                    direction=position["direction"],
                    entry_time=position["entry_time"], entry_price=position["entry_price"],
                    exit_time=c.dt.isoformat(), exit_price=exit_price, exit_reason=reason,
                    risk_amount=position["risk_amount"], pnl=pnl, balance_after=balance,
                ))

                if pnl < 0:
                    consecutive_losses += 1
                    if consecutive_losses >= LOSS_STREAK_FOR_COOLDOWN:
                        cooldown_remaining = COOLDOWN_CHECKS
                        cooldown_trigger_count += 1
                        consecutive_losses = 0
                else:
                    consecutive_losses = 0

                if day_pl <= -DAILY_LOSS_LIMIT_PCT * day_start_balance and not breaker_tripped_today:
                    breaker_tripped_today = True
                    breaker_trip_count += 1

                position = None
            continue  # this candle managed a position -- never also opens one

        # -- cooldown / circuit breaker gates ---------------------------------
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        if breaker_tripped_today:
            continue

        # -- look for a crossover confirmed on the candle that just closed ----
        f0, s0 = fast[i - 2], slow[i - 2]
        f1, s1 = fast[i - 1], slow[i - 1]
        if None in (f0, s0, f1, s1):
            continue

        direction = None
        if f0 <= s0 and f1 > s1:
            direction = "long"
        elif f0 >= s0 and f1 < s1:
            direction = "short"
        if direction is None:
            continue

        if use_trend_filter:
            t1 = trend[i - 1]
            if t1 is None:
                continue  # not enough history yet for the trend SMA -- wait
            if direction == "long" and not (closes[i - 1] > t1):
                continue  # crossover fired against the broader trend -- skip it
            if direction == "short" and not (closes[i - 1] < t1):
                continue

        entry_price = c.open
        risk_amount = balance * STAKE_PCT
        if direction == "long":
            sl = entry_price * (1 - SL_PCT)
            tp = entry_price * (1 + TP_PCT)
        else:
            sl = entry_price * (1 + SL_PCT)
            tp = entry_price * (1 - TP_PCT)

        position = {
            "direction": direction, "entry_time": c.dt.isoformat(),
            "entry_price": entry_price, "sl": sl, "tp": tp, "risk_amount": risk_amount,
        }

    results = summarize(trades, equity_curve, max_drawdown_pct, candles, breaker_trip_count, cooldown_trigger_count)
    results["config"] = {"trend_filter": use_trend_filter, "trend_filter_ma": TREND_FILTER_MA if use_trend_filter else None}
    return results


def summarize(trades, equity_curve, max_drawdown_pct, candles, breaker_trip_count, cooldown_trigger_count) -> dict:
    total = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    win_rate = len(wins) / total if total else 0.0
    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = None if gross_profit == 0 else float("inf")
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss) if total else 0.0

    longest_win_streak = longest_loss_streak = cur_win = cur_loss = 0
    for t in trades:
        if t.pnl > 0:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        longest_win_streak = max(longest_win_streak, cur_win)
        longest_loss_streak = max(longest_loss_streak, cur_loss)

    final_balance = equity_curve[-1]
    total_return_pct = (final_balance - STARTING_BALANCE) / STARTING_BALANCE * 100
    buy_hold_pct = (candles[-1].close - candles[0].close) / candles[0].close * 100

    return {
        "period": {
            "start": candles[0].dt.isoformat(),
            "end": candles[-1].dt.isoformat(),
            "candles": len(candles),
        },
        "trades": {
            "total": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 3) if isinstance(profit_factor, float) and profit_factor != float("inf") else profit_factor,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy_per_trade": round(expectancy, 2),
            "longest_win_streak": longest_win_streak,
            "longest_loss_streak": longest_loss_streak,
        },
        "risk_events": {
            "daily_circuit_breaker_triggers": breaker_trip_count,
            "cooldown_triggers": cooldown_trigger_count,
        },
        "performance": {
            "starting_balance": STARTING_BALANCE,
            "final_balance": round(final_balance, 2),
            "total_return_pct": round(total_return_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct * 100, 2),
            "buy_and_hold_pct": round(buy_hold_pct, 2),
        },
        "trade_log": [asdict(t) for t in trades],
        "equity_curve": equity_curve,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _format_block(results: dict, label: str) -> str:
    p, t, r = results["performance"], results["trades"], results["risk_events"]
    pf = t["profit_factor"]
    pf_display = "inf" if pf == float("inf") else ("n/a" if pf is None else f"{pf}")
    return f"""### {label}
- Total trades: {t['total']}
- Win rate: {t['win_rate_pct']:.2f}%
- Profit factor: {pf_display}
- Total return: {p['total_return_pct']:.2f}%
- Max drawdown: {p['max_drawdown_pct']:.2f}%
- Avg win: ${t['avg_win']:.2f} / Avg loss: ${t['avg_loss']:.2f}
- Expectancy per trade: ${t['expectancy_per_trade']:.2f}
- Longest win / loss streak: {t['longest_win_streak']} / {t['longest_loss_streak']}
- Circuit breaker triggered: {r['daily_circuit_breaker_triggers']}x | Cooldown triggered: {r['cooldown_triggers']}x
"""


def write_comparison_report(baseline: dict, filtered: dict, outdir: Path) -> str:
    """Writes baseline (raw crossover) and trend-filtered results side by
    side, so improvements are shown honestly against the old numbers rather
    than just asserted."""
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results_baseline.json").write_text(json.dumps(baseline, indent=2))
    (outdir / "results_trend_filter.json").write_text(json.dumps(filtered, indent=2))

    period = baseline["period"]
    bt, ft = baseline["trades"], filtered["trades"]
    bp, fp = baseline["performance"], filtered["performance"]
    breakeven_wr = SL_PCT / (SL_PCT + TP_PCT) * 100

    md = f"""# EUR/USD Backtest Results -- Baseline vs. Trend Filter

**Period:** {period['start'][:10]} to {period['end'][:10]} ({period['candles']} hourly candles)
**Base strategy:** {FAST_MA}/{SLOW_MA} SMA crossover | {SL_PCT*100:.1f}% SL / {TP_PCT*100:.1f}% TP | {STAKE_PCT*100:.0f}% stake
**Breakeven win rate at this risk:reward:** {breakeven_wr:.1f}% -- below this, more trades means more loss, not less.

| Metric | Baseline (no filter) | With {TREND_FILTER_MA}h trend filter |
|---|---|---|
| Trades | {bt['total']} | {ft['total']} |
| Win rate | {bt['win_rate_pct']:.1f}% | {ft['win_rate_pct']:.1f}% |
| Profit factor | {bt['profit_factor']} | {ft['profit_factor']} |
| Total return | {bp['total_return_pct']:.2f}% | {fp['total_return_pct']:.2f}% |
| Max drawdown | {bp['max_drawdown_pct']:.2f}% | {fp['max_drawdown_pct']:.2f}% |
| Expectancy/trade | ${bt['expectancy_per_trade']:.2f} | ${ft['expectancy_per_trade']:.2f} |

{_format_block(baseline, "Baseline -- full detail")}
{_format_block(filtered, f"With {TREND_FILTER_MA}h trend filter -- full detail")}

---
**How to read this honestly:** the trend filter takes fewer trades on purpose -- it's designed to skip signals that fire against the broader move. If it shows a meaningfully better win rate and expectancy on those fewer trades, that's worth following up on with a bigger dataset. If both columns tell roughly the same losing story, that means the problem isn't *which* crossovers to skip -- it's the crossover signal itself, and the next fix needs to be more fundamental than a filter.

*Backtest only, run on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Spread approximated at {SPREAD_PCT*100:.2f}% per trade; no overnight financing modelled. Full trade logs in results_baseline.json / results_trend_filter.json.*
"""
    (outdir / "latest_summary.md").write_text(md)
    return md


def post_to_discord(md: str):
    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        return
    content = md[:1900]
    payload = json.dumps({"content": f"```md\n{content}\n```"}).encode()
    req = urllib.request.Request(webhook, data=payload, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"(Discord post failed, non-fatal: {e})")


async def main():
    parser = argparse.ArgumentParser(description="Backtest the 5/20 SMA crossover strategy on EUR/USD.")
    parser.add_argument("--years", type=float, default=2.0, help="Years of history to pull (default: 2)")
    parser.add_argument("--outdir", type=str, default="backtest_results", help="Where to write results")
    args = parser.parse_args()

    print(f"Fetching ~{args.years} years of hourly {SYMBOL} candles from Deriv...")
    candles = await fetch_candles(args.years)
    print(f"Got {len(candles)} candles: {candles[0].dt.date()} to {candles[-1].dt.date()}")

    print("Running baseline (no trend filter)...")
    baseline = run_backtest(candles, use_trend_filter=False)
    print("Running with trend filter...")
    filtered = run_backtest(candles, use_trend_filter=True)

    md = write_comparison_report(baseline, filtered, Path(args.outdir))
    post_to_discord(md)

    print("\n" + md)


if __name__ == "__main__":
    asyncio.run(main())
