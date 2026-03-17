from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeEntry:
    key: str
    title: str
    aliases: tuple[str, ...]
    answer_html: str


_KB: tuple[KnowledgeEntry, ...] = (
    KnowledgeEntry(
        key="smc",
        title="SMC (smart money concepts)",
        aliases=("smc", "smart money concepts", "smart money"),
        answer_html=(
            "<b>smc</b> = a way to read price like liquidity engineering: "
            "where stops sit, where big orders likely filled, and how structure shifts.\n\n"
            "<b>core pieces</b>: liquidity (highs/lows), bos/choch, order blocks, fvgs/imbalances.\n"
            "<b>why it matters</b>: it tells you where price is likely to raid + where a move is invalidated.\n"
            "<b>common L</b>: drawing an order block on every candle and calling it \u201csmart money\u201d. pick the cleanest displacement + structure break.\n"
            "<b>quick example</b>: sweep yesterday's high → displacement down → retest into fvg/order block → continuation.\n"
        ),
    ),
    KnowledgeEntry(
        key="order_block",
        title="Order block",
        aliases=("order block", "ob"),
        answer_html=(
            "<b>order block</b> = the last opposing candle before a strong displacement that breaks structure.\n"
            "<b>idea</b>: institutions likely filled there, so retests can react.\n"
            "<b>how to use</b>: mark the candle that precedes the impulse + confirm with a bos/choch; treat it as a zone, not a single line.\n"
        ),
    ),
    KnowledgeEntry(
        key="fvg",
        title="FVG (fair value gap)",
        aliases=("fvg", "fair value gap", "imbalance"),
        answer_html=(
            "<b>fvg</b> = a 3-candle imbalance where price moved so fast it left a gap in efficient trading.\n"
            "<b>why it matters</b>: price often revisits that gap to rebalance liquidity.\n"
            "<b>usage</b>: trade the retrace into the gap in the direction of the displacement, with structure as your filter.\n"
        ),
    ),
    KnowledgeEntry(
        key="bos",
        title="BOS (break of structure)",
        aliases=("bos", "break of structure"),
        answer_html=(
            "<b>bos</b> = price takes out the last meaningful swing in the trend direction.\n"
            "think: confirmation that the trend is still intact, not a signal by itself.\n"
        ),
    ),
    KnowledgeEntry(
        key="choch",
        title="CHOCH (change of character)",
        aliases=("choch", "change of character"),
        answer_html=(
            "<b>choch</b> = the first structure break against the prior trend that hints regime change.\n"
            "use it to stop fading the move and start looking for the new direction setups.\n"
        ),
    ),
    KnowledgeEntry(
        key="liquidity_sweep",
        title="Liquidity sweep / stop hunt",
        aliases=("liquidity sweep", "sweep", "stop hunt", "liquidity grab", "raid"),
        answer_html=(
            "<b>liquidity sweep</b> = price runs a prior high/low to trigger stops + fill orders, then often snaps back.\n"
            "<b>tell</b>: wick through the level + quick reclaim + displacement.\n"
        ),
    ),
    KnowledgeEntry(
        key="tp",
        title="TP (take profit)",
        aliases=("tp", "take profit", "take-profit"),
        answer_html=(
            "<b>tp</b> = where you close some/all of a winning trade.\n"
            "best practice: take partials into liquidity/levels and trail the rest, instead of praying for the exact top.\n"
        ),
    ),
    KnowledgeEntry(
        key="sl",
        title="SL (stop loss)",
        aliases=("sl", "stop loss", "stop-loss", "stop"),
        answer_html=(
            "<b>sl</b> = the price where your trade thesis is invalidated and you exit.\n"
            "place it where structure breaks (not where your emotions break).\n"
        ),
    ),
    KnowledgeEntry(
        key="dca",
        title="DCA (dollar-cost averaging)",
        aliases=("dca", "dollar cost averaging", "averaging"),
        answer_html=(
            "<b>dca</b> = spreading entries over time/levels to reduce timing risk.\n"
            "<b>warning</b>: dca is not a magic spell — if the thesis is wrong, you're just scaling into rekt.\n"
        ),
    ),
    KnowledgeEntry(
        key="leverage",
        title="Leverage",
        aliases=("leverage", "lev"),
        answer_html=(
            "<b>leverage</b> = borrowing to amplify exposure.\n"
            "it doesn't change direction, it changes how fast you get liquidated. size small, respect the stop.\n"
        ),
    ),
    # ── Indicators ─────────────────────────────────────────────────────────────
    KnowledgeEntry(
        key="rsi",
        title="RSI (relative strength index)",
        aliases=("rsi", "relative strength index", "relative strength"),
        answer_html=(
            "<b>rsi</b> = momentum oscillator (0–100) measuring speed + change of price moves.\n\n"
            "<b>reads</b>: above 70 = overbought (crowded longs); below 30 = oversold (crowded shorts).\n"
            "<b>why it matters</b>: tells you if a move is overextended before you chase it.\n"
            "<b>better use</b>: divergence — price makes new high but rsi doesn't → hidden weakness. "
            "also: rsi holding above 50 in uptrend = healthy; below 50 = distribution.\n"
            "<b>common L</b>: selling just because rsi hit 70. in a strong trend it can stay overbought for days.\n"
        ),
    ),
    KnowledgeEntry(
        key="macd",
        title="MACD",
        aliases=("macd", "moving average convergence divergence", "macd histogram"),
        answer_html=(
            "<b>macd</b> = difference between two emas (usually 12 and 26), with a signal line (9 ema of the macd).\n\n"
            "<b>reads</b>: macd crosses above signal → bullish momentum; below → bearish. "
            "histogram growing → momentum accelerating; shrinking → fading.\n"
            "<b>best use</b>: divergence (price new low, macd higher low = potential reversal) + "
            "trend confirmation (is the move accelerating or dying?).\n"
            "<b>common L</b>: using macd crossovers as entry signals in chop — they lag and generate fake-outs constantly.\n"
        ),
    ),
    KnowledgeEntry(
        key="ema",
        title="EMA (exponential moving average)",
        aliases=("ema", "exponential moving average", "moving average", "sma", "simple moving average", "ma"),
        answer_html=(
            "<b>ema</b> = moving average that weights recent candles more heavily than older ones (faster than sma).\n\n"
            "<b>key levels used in crypto</b>: ema9/21 (scalp), ema50/100 (swing), ema200 (macro trend bias).\n"
            "<b>read</b>: price above ema200 = bull territory. price reclaims ema20 after pullback = continuation signal.\n"
            "<b>confluence</b>: when multiple emas cluster together, that area becomes strong support/resistance.\n"
            "<b>sma</b> = same idea but equal weight on all candles; slower, commonly used on daily charts.\n"
        ),
    ),
    KnowledgeEntry(
        key="vwap",
        title="VWAP (volume-weighted average price)",
        aliases=("vwap", "volume weighted average price", "volume-weighted"),
        answer_html=(
            "<b>vwap</b> = average price weighted by volume — the 'fair price' institutions pay on the day.\n\n"
            "<b>read</b>: price above vwap = buyers in control intraday; below = sellers.\n"
            "<b>use</b>: reclaim of vwap after a drop is a high-probability long on the lower timeframe. "
            "rejection at vwap from below = bearish continuation.\n"
            "<b>limit</b>: resets at midnight UTC — mainly useful for intraday/scalp setups, not swing.\n"
        ),
    ),
    KnowledgeEntry(
        key="bollinger_bands",
        title="Bollinger Bands",
        aliases=("bollinger bands", "bollinger", "bb", "bband", "bbands"),
        answer_html=(
            "<b>bollinger bands</b> = middle band (20 sma) + upper/lower bands (2 std deviations).\n\n"
            "<b>reads</b>: price touching lower band = potentially oversold; upper band = potentially overbought. "
            "bands squeezing together (low volatility) = big move incoming. "
            "bands expanding = trend in motion, don't fade it.\n"
            "<b>walk the band</b>: in strong trends price can hug the upper/lower band for extended periods. don't blindly fade.\n"
        ),
    ),
    KnowledgeEntry(
        key="fibonacci",
        title="Fibonacci retracement",
        aliases=("fibonacci", "fib", "fibs", "fibonacci retracement", "fib levels", "fib retracement"),
        answer_html=(
            "<b>fibonacci retracement</b> = key levels derived from the fib sequence, used to find where a pullback might pause or reverse.\n\n"
            "<b>main levels</b>: 0.236, 0.382, 0.5, 0.618 (golden ratio), 0.786.\n"
            "<b>use</b>: draw from swing low to swing high (or vice versa for downtrend). "
            "0.618 ('the golden pocket' with 0.65) is the most respected retracement in crypto.\n"
            "<b>confluence</b>: fib level + order block/fvg + support = high-probability entry zone.\n"
            "<b>common L</b>: drawing fibs on every minor swing. only use on significant impulse moves.\n"
        ),
    ),
    KnowledgeEntry(
        key="support_resistance",
        title="Support and resistance",
        aliases=("support", "resistance", "support and resistance", "support resistance", "s/r", "s&r", "key level", "key levels"),
        answer_html=(
            "<b>support</b> = price area where buying has historically absorbed selling. "
            "<b>resistance</b> = area where selling has historically capped buying.\n\n"
            "<b>key rule</b>: broken support flips to resistance (and vice versa). "
            "the more times a level is tested, the weaker it gets — not stronger.\n"
            "<b>best levels</b>: round numbers, prior swing highs/lows, previous monthly/weekly open/close, high-volume nodes.\n"
            "<b>trade it</b>: enter near support with sl just below; short near resistance with sl just above. "
            "wait for a reaction candle, don't just buy the touch.\n"
        ),
    ),
    KnowledgeEntry(
        key="atr",
        title="ATR (average true range)",
        aliases=("atr", "average true range"),
        answer_html=(
            "<b>atr</b> = average daily price range — the market's volatility ruler.\n\n"
            "<b>use 1</b>: set stop losses using atr multiples (e.g. sl = entry - 1.5× atr) so volatility doesn't shake you out.\n"
            "<b>use 2</b>: set realistic targets. if atr = $1500, a $3000 tp is 2× atr — reasonable. $500 tp is too tight.\n"
            "<b>use 3</b>: compare atr to recent move. if price moved 3× atr in one candle, the move is probably exhausted.\n"
        ),
    ),
    KnowledgeEntry(
        key="divergence",
        title="Divergence",
        aliases=("divergence", "bullish divergence", "bearish divergence", "hidden divergence", "rsi divergence", "macd divergence"),
        answer_html=(
            "<b>divergence</b> = price and an indicator (rsi/macd) disagreeing on direction.\n\n"
            "<b>regular bullish</b>: price makes lower low, rsi makes higher low → potential reversal up.\n"
            "<b>regular bearish</b>: price makes higher high, rsi makes lower high → potential reversal down.\n"
            "<b>hidden bullish</b>: price makes higher low, rsi makes lower low → continuation of uptrend.\n"
            "<b>hidden bearish</b>: price makes lower high, rsi makes higher high → continuation of downtrend.\n"
            "<b>note</b>: divergence is a warning, not a trigger. wait for structure break/entry signal to confirm.\n"
        ),
    ),
    KnowledgeEntry(
        key="funding_rate",
        title="Funding rate",
        aliases=("funding rate", "funding", "funding fee"),
        answer_html=(
            "<b>funding rate</b> = periodic fee paid between longs and shorts on perpetual futures to keep price near spot.\n\n"
            "<b>positive funding</b>: longs pay shorts → market is overleveraged long → contrarian signal (crowded trade).\n"
            "<b>negative funding</b>: shorts pay longs → overleveraged short → potential squeeze catalyst.\n"
            "<b>extreme funding</b>: when funding spikes (e.g. 0.1%+ per 8h), the crowd is one-sided — "
            "these moments often precede violent reversals as the trade unwinds.\n"
            "<b>check it</b>: coinglass.com shows live funding rates across exchanges.\n"
        ),
    ),
    KnowledgeEntry(
        key="open_interest",
        title="Open interest",
        aliases=("open interest", "oi", "futures oi"),
        answer_html=(
            "<b>open interest</b> = total number of outstanding futures/perp contracts that haven't been settled.\n\n"
            "<b>rising oi + rising price</b>: new money entering longs — healthy trend.\n"
            "<b>rising oi + falling price</b>: new money entering shorts — bearish conviction.\n"
            "<b>falling oi + price move</b>: existing positions closing (liquidations or profit-taking) — move may exhaust.\n"
            "<b>oi spike</b>: sudden jump in oi = large positions opening. watch for a flush/squeeze of whoever is wrong.\n"
        ),
    ),
    KnowledgeEntry(
        key="liquidation",
        title="Liquidation",
        aliases=("liquidation", "liquidated", "liq", "liquidation cascade", "get liquidated"),
        answer_html=(
            "<b>liquidation</b> = when an exchange force-closes your leveraged position because margin ran out.\n\n"
            "<b>cascade</b>: when liq prices cluster together, one liquidation triggers the next → "
            "violent price spike that blows through levels in seconds.\n"
            "<b>liquidation heatmaps</b> (coinglass) show where large liquidations are stacked — "
            "price is magnetically drawn to these clusters before reversing.\n"
            "<b>avoid</b>: use low leverage, set sl before liq price, and size so 1 atr move doesn't threaten your margin.\n"
        ),
    ),
    KnowledgeEntry(
        key="spot_vs_perp",
        title="Spot vs perpetual futures",
        aliases=("spot", "perpetual", "perp", "perps", "futures", "spot vs futures", "spot vs perp", "perpetual futures"),
        answer_html=(
            "<b>spot</b> = you own the actual asset. no expiry, no funding, no leverage risk of liquidation.\n"
            "<b>perpetual futures (perp)</b> = a contract tracking the spot price with no expiry date. "
            "uses funding rates to stay pegged to spot. supports leverage.\n\n"
            "<b>key difference</b>: spot = you can hold forever and it can't go to zero unless the coin dies. "
            "perp = leverage amplifies both gains and losses; wrong direction + bad sizing = liquidation.\n"
            "<b>when to use what</b>: spot for longer holds + no liquidation risk. "
            "perp for shorter trades where you want leverage or want to short.\n"
        ),
    ),
    KnowledgeEntry(
        key="market_cap",
        title="Market cap",
        aliases=("market cap", "marketcap", "market capitalization", "mcap"),
        answer_html=(
            "<b>market cap</b> = current price × circulating supply.\n\n"
            "<b>tiers</b>: large cap (BTC, ETH) = liquid, lower volatility, safer. "
            "mid cap = more upside/downside. small/micro cap = high volatility, illiquid, rug risk.\n"
            "<b>not the full picture</b>: a low market cap doesn't mean undervalued — "
            "check fdv (fully diluted valuation) to see total supply impact.\n"
        ),
    ),
    KnowledgeEntry(
        key="fdv",
        title="FDV (fully diluted valuation)",
        aliases=("fdv", "fully diluted valuation", "fully diluted"),
        answer_html=(
            "<b>fdv</b> = what the market cap would be if ALL tokens (including locked/unvested) were in circulation.\n\n"
            "<b>why it matters</b>: if fdv >> market cap, there's a ton of supply coming. "
            "those tokens will eventually be sold → selling pressure → price suppression.\n"
            "<b>red flag</b>: fdv 10× the market cap = 90% of supply not yet circulating. "
            "team/vcs hold most of it and will sell when vesting unlocks.\n"
            "<b>check it</b>: coingecko and coinmarketcap both show fdv and vesting schedules.\n"
        ),
    ),
    KnowledgeEntry(
        key="btc_dominance",
        title="BTC dominance",
        aliases=("btc dominance", "bitcoin dominance", "btc.d", "btc dom"),
        answer_html=(
            "<b>btc dominance</b> = btc's market cap as a % of total crypto market cap.\n\n"
            "<b>rising dominance</b>: money flowing into btc, rotating out of alts (risk-off or btc-specific catalyst).\n"
            "<b>falling dominance</b>: money rotating into alts — altseason territory.\n"
            "<b>typical pattern</b>: btc pumps first → dominance peaks → dominance rolls over → alts go parabolic.\n"
            "<b>watch for</b>: dominance at historical resistance (e.g. 58–60%) → potential alt rotation signal. "
            "dominance breaking above = stay in btc/stables.\n"
        ),
    ),
    KnowledgeEntry(
        key="altseason",
        title="Altseason",
        aliases=("altseason", "alt season", "alts season", "alt run", "altcoin season"),
        answer_html=(
            "<b>altseason</b> = period where altcoins significantly outperform btc.\n\n"
            "<b>conditions</b>: btc dominance falling, btc stable or slowly rising, risk appetite high, liquidity expanding.\n"
            "<b>how it plays</b>: large caps (ETH, SOL) move first, then mid caps, then small/micro caps. "
            "by the time micro caps are pumping, the cycle is usually late.\n"
            "<b>tell</b>: eth/btc ratio rising = altseason starting. falling = btc dominance phase.\n"
            "<b>trap</b>: alts can get destroyed fast when btc has a big move down. size accordingly.\n"
        ),
    ),
    KnowledgeEntry(
        key="pullback",
        title="Pullback / retracement",
        aliases=("pullback", "retracement", "retrace", "dip", "correction", "healthy correction"),
        answer_html=(
            "<b>pullback</b> = temporary counter-trend move within a larger trend — normal and healthy.\n\n"
            "<b>vs correction</b>: pullback = minor (5–15% typically). correction = deeper (15–30%). crash = structural break.\n"
            "<b>trade it</b>: wait for pullback into key level (fib 0.5–0.618, ema, order block), "
            "watch for rejection candle + volume tapering, then enter with sl below the structure.\n"
            "<b>don't</b>: assume every dip bounces. check if structure is still intact (higher highs/lows holding).\n"
        ),
    ),
    KnowledgeEntry(
        key="scalp",
        title="Scalp / scalping",
        aliases=("scalp", "scalping", "scalp trade", "scalper"),
        answer_html=(
            "<b>scalp</b> = very short-term trade targeting small price moves, usually minutes to a few hours.\n\n"
            "<b>requires</b>: tight spreads (liquid pairs), fast execution, strict discipline on stops.\n"
            "<b>typical setup</b>: 1m/5m chart, clear support/resistance, low-latency entry, 1:1.5–2 r:r minimum.\n"
            "<b>edge</b>: more setups per day, faster feedback loop. "
            "<b>downside</b>: fees add up fast, emotionally exhausting, easy to overtrade.\n"
            "<b>key rule</b>: the lower the timeframe, the more noise. only scalp clear structure breaks with volume.\n"
        ),
    ),
    KnowledgeEntry(
        key="swing_trade",
        title="Swing trade",
        aliases=("swing trade", "swing trading", "swing", "swing trader"),
        answer_html=(
            "<b>swing trade</b> = holding a position for days to weeks, targeting a larger price move.\n\n"
            "<b>uses</b>: 4h/daily charts for bias, 1h for entry timing.\n"
            "<b>advantages</b>: less screen time, bigger r:r targets (2:1 to 5:1), fewer fees.\n"
            "<b>risks</b>: overnight/weekend exposure to news, macro events, funding costs on perps.\n"
            "<b>key</b>: thesis must be clear before entering. define your invalidation level (sl) before entry price.\n"
        ),
    ),
    KnowledgeEntry(
        key="rr",
        title="Risk/reward ratio",
        aliases=("risk reward", "risk/reward", "r:r", "rr", "risk to reward", "reward ratio"),
        answer_html=(
            "<b>risk/reward (r:r)</b> = how much you stand to gain vs how much you risk on a trade.\n\n"
            "<b>example</b>: risk $500 to make $1500 = 3:1 r:r.\n"
            "<b>minimum</b>: most pros won't take less than 2:1. below that, your win rate has to be very high to be profitable.\n"
            "<b>key insight</b>: with 2:1 r:r, you can be wrong 40% of the time and still profit. "
            "with 1:1, you need 50%+ win rate just to break even (before fees).\n"
        ),
    ),
    KnowledgeEntry(
        key="position_sizing",
        title="Position sizing",
        aliases=("position size", "position sizing", "how much to buy", "how much to risk", "sizing"),
        answer_html=(
            "<b>position sizing</b> = how much capital to deploy on a single trade.\n\n"
            "<b>rule</b>: risk only 1–2% of your total account per trade (not 1–2% of the position — of your whole account).\n"
            "<b>formula</b>: position size = (account × risk %) ÷ (entry - sl).\n"
            "<b>example</b>: $10k account, 1% risk = $100 max loss. "
            "if sl is $500 below entry → position = $100 ÷ $500 × entry price.\n"
            "<b>why it matters</b>: proper sizing means a losing streak doesn't blow the account. "
            "no trade should be existential.\n"
        ),
    ),
    KnowledgeEntry(
        key="breakout",
        title="Breakout / breakdown",
        aliases=("breakout", "breakdown", "break out", "break down", "breaking out", "breaking down"),
        answer_html=(
            "<b>breakout</b> = price closes above a key resistance level with conviction (ideally volume).\n"
            "<b>breakdown</b> = price closes below key support.\n\n"
            "<b>confirmed vs fake</b>: a real breakout has: clean close above/below the level, "
            "increased volume, and a successful retest of the level as new support/resistance.\n"
            "<b>entry</b>: either on the candle close above (aggressive) or wait for retest of the level (conservative).\n"
            "<b>common mistake</b>: chasing the wick break before a confirmed close — "
            "these are often stop hunts/fake-outs.\n"
        ),
    ),
    KnowledgeEntry(
        key="consolidation",
        title="Consolidation / ranging",
        aliases=("consolidation", "ranging", "range", "chopping", "chop", "sideways", "accumulation", "distribution"),
        answer_html=(
            "<b>consolidation</b> = price moving sideways between a clear high and low — "
            "the market is deciding its next direction.\n\n"
            "<b>types</b>: accumulation (buyers loading quietly before breakout up) or "
            "distribution (sellers unloading before breakdown).\n"
            "<b>trade it</b>: buy the low of range / sell the high of range with tight stops, "
            "or wait for the range break with volume and trade the expansion.\n"
            "<b>rule</b>: the longer the range, the bigger the eventual move. "
            "don't force trades in the middle of consolidation — it's noise.\n"
        ),
    ),
    KnowledgeEntry(
        key="fear_greed",
        title="Fear & Greed Index",
        aliases=("fear and greed", "fear & greed", "fear greed", "fear greed index", "crypto fear", "sentiment index"),
        answer_html=(
            "<b>fear & greed index</b> = 0–100 sentiment gauge for crypto markets.\n\n"
            "<b>reads</b>: 0–25 = extreme fear (capitulation zone, potential buy); "
            "75–100 = extreme greed (euphoria zone, potential top).\n"
            "<b>contrarian tool</b>: buy when others are fearful, sell when greedy — "
            "but it's a timing filter, not a standalone signal.\n"
            "<b>components</b>: volatility, market momentum/volume, social media, dominance, trends.\n"
            "<b>limit</b>: can stay in greed for weeks in bull runs. don't use it alone.\n"
        ),
    ),
    KnowledgeEntry(
        key="long_short",
        title="Long / short position",
        aliases=("long", "short", "long position", "short position", "going long", "going short", "shorting"),
        answer_html=(
            "<b>long</b> = betting price goes up. you buy and profit if it rises.\n"
            "<b>short</b> = betting price goes down. you borrow + sell, then buy back cheaper to profit.\n\n"
            "<b>on perps</b>: go long by buying the contract; go short by selling it. "
            "both use margin and can be liquidated.\n"
            "<b>short selling on spot</b>: usually not available in crypto spot — use perps/futures to short.\n"
            "<b>short squeeze</b>: when shorts are forced to buy back (stopped out), causing a rapid price spike up.\n"
        ),
    ),
    KnowledgeEntry(
        key="ichimoku",
        title="Ichimoku cloud",
        aliases=("ichimoku", "ichimoku cloud", "kumo", "cloud"),
        answer_html=(
            "<b>ichimoku cloud</b> = multi-component Japanese indicator showing trend, momentum, and support/resistance all at once.\n\n"
            "<b>key read</b>: price above cloud = bullish. price below cloud = bearish. "
            "inside cloud = indecision.\n"
            "<b>cloud color</b>: green cloud (kumo) = bullish. red cloud = bearish.\n"
            "<b>key signals</b>: tenkan/kijun cross (like ema crossover), "
            "price breaking through the cloud with volume = strong signal.\n"
            "<b>best on</b>: daily/4h charts. on lower timeframes, too much noise.\n"
        ),
    ),
    KnowledgeEntry(
        key="adx",
        title="ADX (average directional index)",
        aliases=("adx", "average directional index", "directional movement"),
        answer_html=(
            "<b>adx</b> = measures trend <i>strength</i>, not direction (0–100 scale).\n\n"
            "<b>reads</b>: below 20 = weak/no trend (range). 20–40 = developing trend. "
            "above 40 = strong trend. above 50 = extremely strong.\n"
            "<b>how to use</b>: if adx is below 20, avoid trend-following setups — the market is chopping. "
            "use adx above 25 as a filter before taking breakouts.\n"
            "<b>note</b>: adx rising doesn't tell you if the trend is up or down — "
            "use +di/-di lines or price structure for direction.\n"
        ),
    ),
    KnowledgeEntry(
        key="volume",
        title="Volume",
        aliases=("volume", "trading volume", "vol"),
        answer_html=(
            "<b>volume</b> = the amount of an asset traded in a given period — the fuel behind price moves.\n\n"
            "<b>rules</b>: high volume + price move = conviction. low volume + price move = suspect, likely fades.\n"
            "<b>volume spike</b>: sudden large volume = institutions/whales acting. "
            "green volume spike on support = accumulation. red spike at resistance = distribution.\n"
            "<b>divergence</b>: price making new highs on declining volume = warning sign, trend weakening.\n"
            "<b>don't</b>: trade low-volume breakouts — they fail more often than they follow through.\n"
        ),
    ),
    KnowledgeEntry(
        key="stochastic",
        title="Stochastic oscillator",
        aliases=("stochastic", "stoch", "stochastic oscillator"),
        answer_html=(
            "<b>stochastic</b> = momentum oscillator (0–100) comparing close to recent high-low range.\n\n"
            "<b>reads</b>: above 80 = overbought. below 20 = oversold. "
            "%k crossing %d from below in oversold zone = bullish signal.\n"
            "<b>best use</b>: confirm entries in ranging markets. "
            "in strong trends, stochastic can stay overbought/oversold for a long time — don't fight it.\n"
            "<b>combined</b>: use with support/resistance. stochastic oversold at key support = "
            "higher-probability long setup.\n"
        ),
    ),
    KnowledgeEntry(
        key="pnl",
        title="PnL (profit and loss)",
        aliases=("pnl", "p&l", "profit and loss", "unrealized pnl", "realized pnl", "upnl"),
        answer_html=(
            "<b>pnl</b> = your profit or loss on a trade.\n\n"
            "<b>unrealized pnl</b>: paper gain/loss on an open position — doesn't count until you close.\n"
            "<b>realized pnl</b>: locked in once you close. this is what actually hits your account balance.\n"
            "<b>trap</b>: watching unrealized pnl too closely causes emotional decisions. "
            "define your exit (sl/tp) before you enter, then let it play out.\n"
        ),
    ),
    KnowledgeEntry(
        key="liquidation_price",
        title="Liquidation price",
        aliases=("liquidation price", "liq price", "liq level"),
        answer_html=(
            "<b>liquidation price</b> = the price at which the exchange force-closes your leveraged position "
            "because your margin is exhausted.\n\n"
            "<b>formula</b>: for a 10× long, your liq is roughly 10% below entry. "
            "for 20×, roughly 5% below entry.\n"
            "<b>rule</b>: your stop loss should always be hit before your liquidation price. "
            "if you're relying on not getting liquidated instead of having a sl, you're gambling.\n"
            "<b>check</b>: most exchanges show your estimated liq price when you open a position — always look.\n"
        ),
    ),
    KnowledgeEntry(
        key="market_structure",
        title="Market structure",
        aliases=("market structure", "structure", "hh hl", "lh ll", "higher highs", "higher lows"),
        answer_html=(
            "<b>market structure</b> = the pattern of swing highs and lows that defines the current trend.\n\n"
            "<b>uptrend</b>: higher highs (hh) + higher lows (hl) — buyers in control.\n"
            "<b>downtrend</b>: lower highs (lh) + lower lows (ll) — sellers in control.\n"
            "<b>shift</b>: uptrend breaks when price takes out the last hl → first sign of reversal. "
            "wait for a confirmed lh after the break to start looking for shorts.\n"
            "<b>use</b>: always define trend direction on the higher timeframe before picking entries on lower tf.\n"
        ),
    ),
)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def extract_definition_term(text: str) -> str | None:
    """Best-effort extraction of the term being defined."""
    t = _normalize(text)
    if not t:
        return None
    # Common patterns
    patterns = (
        r"^what is\s+(.+)$",
        r"^what's\s+(.+)$",
        r"^define\s+(.+)$",
        r"^explain\s+(.+)$",
        r"^meaning of\s+(.+)$",
        r"^what does\s+(.+)\s+mean\??$",
        r"^how does\s+(.+)\s+work\??$",
    )
    for p in patterns:
        m = re.match(p, t)
        if m:
            term = m.group(1).strip(" ?.!,'\"")
            # trim trailing filler
            term = re.sub(r"\b(in crypto|in trading|for crypto|for trading)\b", "", term).strip()
            return term[:60] if term else None
    # fallback: single-token definition like "smc?"
    if len(t) <= 16 and re.fullmatch(r"[a-z0-9\-_/]+", t):
        return t
    return None


def try_answer_definition(text: str) -> str | None:
    """Return a high-quality KB answer (Telegram HTML) if matched."""
    term = extract_definition_term(text)
    if not term:
        return None
    term_n = _normalize(term)

    # Direct alias match first
    for entry in _KB:
        if term_n == entry.key or term_n in entry.aliases:
            return entry.answer_html

    # Fuzzy containment (e.g. "what is fair value gap")
    for entry in _KB:
        if any(a in term_n for a in entry.aliases):
            return entry.answer_html

    return None
