from datamodel import OrderDepth, TradingState, Order, Symbol, Trade, Listing, Observation, ProsperityEncoder
from typing import List, Dict, Any
import json
import numpy as np
from collections import deque

class Logger:
    """Handles state serialization and efficient log compression."""
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp, trader_data, self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths), self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades), state.position, self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.symbol, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {sym: [depth.buy_orders, depth.sell_orders] for sym, depth in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for arr in trades.values() for t in arr]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conv_obs = {p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex] 
                    for p, o in observations.conversionObservations.items()}
        return [observations.plainValueObservations, conv_obs]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

class Product:
    AMETHYSTS = "RAINFOREST_RESIN"
    STARFRUIT = "KELP"
    BASKET1 = "PICNIC_BASKET1"
    BASKET2 = "PICNIC_BASKET2"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    COUPON_9500 = "COUPON_9500"
    COUPON_9750 = "COUPON_9750"
    COUPON_10000 = "COUPON_10000"
    COUPON_10250 = "COUPON_10250"
    COUPON_10500 = "COUPON_10500"

# Round 4 Volatility Framework Parameters
PARAMS = {
    Product.VOLCANIC_ROCK: {"LIMIT": 400, "WINDOW": 20, "VOL_THRES": 2.3, "DEV_THRES": 3.5},
    Product.COUPON_9500:   {"LIMIT": 200, "WINDOW": 20, "VOL_THRES": 1.75, "DEV_THRES": 1.2},
    Product.COUPON_9750:   {"LIMIT": 200, "WINDOW": 20, "VOL_THRES": 1.8,  "DEV_THRES": 1.1},
    Product.COUPON_10000:  {"LIMIT": 200, "WINDOW": 20, "VOL_THRES": 2.2,  "DEV_THRES": 1.2},
}

# Deactivated extreme out-of-the-money options to optimize limit allocations
Mean_Revert = [Product.VOLCANIC_ROCK, Product.COUPON_9500, Product.COUPON_9750, Product.COUPON_10000]

class Trader:
    def __init__(self):
        self.history_size = 300
        self.product_data_history = {}
        
        # Hard Global Limits
        self.position_limits = {
            Product.AMETHYSTS: 50, Product.STARFRUIT: 50,
            Product.BASKET1: 70, Product.BASKET2: 70,
            Product.VOLCANIC_ROCK: 400, Product.COUPON_9500: 200,
            Product.COUPON_9750: 200, Product.COUPON_10000: 200
        }
        
        # Memory blocks for R4 EMA calculation
        self.last_mean = {}

    def update_history(self, product: str, mid_price: float):
        if product not in self.product_data_history:
            self.product_data_history[product] = deque(maxlen=self.history_size)
        self.product_data_history[product].append(mid_price)

    # ---------------------------------------------------------
    # ROUND 1 & 2: LINEAR MARKET MAKING & CLEARING HELPERS
    # ---------------------------------------------------------
    def take_best_orders(self, product, fair_value, take_width, orders, order_depth, position, buy_vol, sell_vol, prevent_adverse=False, adverse_volume=0):
        limit = self.position_limits.get(product, 50)
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            if not prevent_adverse or (-order_depth.sell_orders[best_ask] >= adverse_volume):
                if best_ask <= fair_value - take_width:
                    qty = min(-order_depth.sell_orders[best_ask], limit - position)
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))
                        buy_vol += qty

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            if not prevent_adverse or (order_depth.buy_orders[best_bid] >= adverse_volume):
                if best_bid >= fair_value + take_width:
                    qty = min(order_depth.buy_orders[best_bid], limit + position)
                    if qty > 0:
                        orders.append(Order(product, best_bid, -qty))
                        sell_vol += qty
        return buy_vol, sell_vol

    def market_make(self, product, orders, bid, ask, position, buy_vol, sell_vol):
        limit = self.position_limits.get(product, 50)
        buy_qty = limit - (position + buy_vol)
        if buy_qty > 0: orders.append(Order(product, bid, buy_qty))
        sell_qty = limit + (position - sell_vol)
        if sell_qty > 0: orders.append(Order(product, ask, -sell_qty))

    def clear_position_order(self, product, fair_value, orders, order_depth, position, buy_vol, sell_vol):
        pos_after_take = position + buy_vol - sell_vol
        fair_for_bid = int(np.floor(fair_value))
        fair_for_ask = int(np.ceil(fair_value))
        limit = self.position_limits.get(product, 50)

        if pos_after_take > 0 and fair_for_ask in order_depth.buy_orders:
            qty = min(order_depth.buy_orders[fair_for_ask], pos_after_take)
            sent_qty = min(limit + (position - sell_vol), qty)
            if sent_qty > 0:
                orders.append(Order(product, fair_for_ask, -sent_qty))
                sell_vol += sent_qty

        if pos_after_take < 0 and fair_for_bid in order_depth.sell_orders:
            qty = min(abs(order_depth.sell_orders[fair_for_bid]), abs(pos_after_take))
            sent_qty = min(limit - (position + buy_vol), qty)
            if sent_qty > 0:
                orders.append(Order(product, fair_for_bid, sent_qty))
                buy_vol += sent_qty

        return buy_vol, sell_vol

    # ---------------------------------------------------------
    # STRATEGY EXECUTIONS
    # ---------------------------------------------------------
    def amethyst_orders(self, state: TradingState) -> List[Order]:
        orders = []
        product = Product.AMETHYSTS
        if product not in state.order_depths: return orders
        depth = state.order_depths[product]
        pos = state.position.get(product, 0)
        fair = 10000
        
        b_vol, s_vol = 0, 0
        baaf = min([p for p in depth.sell_orders if p > fair + 1], default=fair + 2)
        bbbf = max([p for p in depth.buy_orders if p < fair - 1], default=fair - 2)

        b_vol, s_vol = self.take_best_orders(product, fair, 0.5, orders, depth, pos, b_vol, s_vol)
        b_vol, s_vol = self.clear_position_order(product, fair, orders, depth, pos, b_vol, s_vol)
        self.market_make(product, orders, int(bbbf + 1), int(baaf - 1), pos, b_vol, s_vol)
        return orders

    def starfruit_orders(self, state: TradingState) -> List[Order]:
        orders = []
        product = Product.STARFRUIT
        if product not in state.order_depths: return orders
        depth = state.order_depths[product]
        pos = state.position.get(product, 0)
        
        if not depth.sell_orders or not depth.buy_orders: return orders
        f_ask = [p for p in depth.sell_orders if abs(depth.sell_orders[p]) >= 15]
        f_bid = [p for p in depth.buy_orders if abs(depth.buy_orders[p]) >= 15]
        
        mm_ask = min(f_ask) if f_ask else min(depth.sell_orders.keys())
        mm_bid = max(f_bid) if f_bid else max(depth.buy_orders.keys())
        fair = (mm_ask + mm_bid) / 2

        b_vol, s_vol = 0, 0
        b_vol, s_vol = self.take_best_orders(product, int(fair), 1.0, orders, depth, pos, b_vol, s_vol, True, 20)
        b_vol, s_vol = self.clear_position_order(product, fair, orders, depth, pos, b_vol, s_vol)
        
        baaf = min([p for p in depth.sell_orders if p > fair + 1], default=fair + 2)
        bbbf = max([p for p in depth.buy_orders if p < fair - 1], default=fair - 2)
        self.market_make(product, orders, int(bbbf + 1), int(baaf - 1), pos, b_vol, s_vol)
        return orders

    def enhanced_basket_arbitrage(self, state: TradingState, basket: str, components: dict) -> List[Order]:
        orders = []
        if basket not in state.order_depths: return orders
        b_depth = state.order_depths[basket]
        if not b_depth.buy_orders or not b_depth.sell_orders: return orders

        pos = state.position.get(basket, 0)
        max_size = self.position_limits.get(basket, 70)
        b_bid, b_ask = max(b_depth.buy_orders.keys()), min(b_depth.sell_orders.keys())
        
        fair_val = 0
        for comp, qty in components.items():
            if comp in state.order_depths:
                c_depth = state.order_depths[comp]
                if c_depth.buy_orders and c_depth.sell_orders:
                    fair_val += ((max(c_depth.buy_orders.keys()) + min(c_depth.sell_orders.keys())) / 2) * qty
                else: return orders
        
        arb_threshold = 40
        if b_ask < fair_val - arb_threshold and pos < max_size:
            max_buy = min(abs(b_depth.sell_orders[b_ask]), max_size - pos)
            factor = min(1.0, (fair_val - b_ask) / (arb_threshold * 3))
            qty = max(1, int(max_buy * factor))
            if qty > 0: orders.append(Order(basket, b_ask, qty))
            
        elif b_bid > fair_val + arb_threshold and pos > -max_size:
            max_sell = min(b_depth.buy_orders[b_bid], max_size + pos)
            factor = min(1.0, (b_bid - fair_val) / (arb_threshold * 3))
            qty = max(1, int(max_sell * factor))
            if qty > 0: orders.append(Order(basket, b_bid, -qty))
            
        return orders

    def regime_switching_orders(self, state: TradingState, product: str) -> List[Order]:
        """Round 4: Nonlinear Volatility Regimes for Volcanic Rock and Coupons."""
        orders = []
        if product not in state.order_depths: return orders
        depth = state.order_depths[product]
        if not depth.buy_orders or not depth.sell_orders: return orders

        limit = PARAMS[product]["LIMIT"]
        position = state.position.get(product, 0)
        
        best_ask, best_bid = min(depth.sell_orders.keys()), max(depth.buy_orders.keys())
        current_price = (best_ask + best_bid) / 2
        
        prices = list(self.product_data_history.get(product, []))
        window = PARAMS[product]["WINDOW"]
        
        # EMA Trendline calculation
        if len(prices) >= window:
            alpha = 2 / (50 + 2.5) 
            if product not in self.last_mean:
                self.last_mean[product] = prices[0]
                moving_avg = self.last_mean[product]
            else: 
                moving_avg = (current_price - self.last_mean[product]) * alpha + self.last_mean[product]
                self.last_mean[product] = moving_avg
        else: 
            moving_avg = np.mean(prices) if len(prices) > 0 else current_price
            
        std = np.std(prices[-50:]) if len(prices) >= 50 else 1
        vol_threshold = PARAMS[product]["VOL_THRES"] 
        vol = np.std(prices[-51:]) if len(prices) >= 51 else vol_threshold
        
        deviation = current_price - moving_avg

        # Regime 1: High Volatility SNAPBACK
        if std > vol_threshold:
            dev_thres = vol * PARAMS[product]["DEV_THRES"]
            if deviation > dev_thres and position > -limit:
                sell_qty = limit + position
                if sell_qty > 0: orders.append(Order(product, best_bid, -sell_qty))
            elif deviation < -dev_thres and position < limit:
                buy_qty = limit - position
                if buy_qty > 0: orders.append(Order(product, best_ask, buy_qty))
                    
        # Regime 2: Low Volatility MARKET MAKING
        else:
            f_ask = [p for p in depth.sell_orders if abs(depth.sell_orders[p]) >= 15]
            f_bid = [p for p in depth.buy_orders if abs(depth.buy_orders[p]) >= 15]

            mm_ask = min(f_ask) if f_ask else best_ask
            mm_bid = max(f_bid) if f_bid else best_bid
            fair = (mm_ask + mm_bid) / 2

            aaf = [p for p in depth.sell_orders if p > fair + 3]
            bbf = [p for p in depth.buy_orders if p < fair - 3]

            best_ask_f = min(aaf) if aaf else fair + 2
            best_bid_f = max(bbf) if bbf else fair - 2

            rem_buy = limit - position
            rem_sell = limit + position
            if rem_buy > 0: orders.append(Order(product, int(best_bid_f + 2), rem_buy))
            if rem_sell > 0: orders.append(Order(product, int(best_ask_f - 2), -rem_sell))
                
        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """Main orchestrator executing the full cumulative universe."""
        result = {}

        # 1. Update Global Price Memory
        for product, order_depth in state.order_depths.items():
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_ask + best_bid) / 2
                self.update_history(product, mid_price)

        # 2. Execute R1 Legacy linear markets
        result[Product.AMETHYSTS] = self.amethyst_orders(state)
        result[Product.STARFRUIT] = self.starfruit_orders(state)
        
        # 3. Execute R2/R3 Basket Arbitrage
        b1_comps = {'CROISSANTS': 6, 'JAMS': 3, 'DJEMBES': 1}
        b2_comps = {'CROISSANTS': 4, 'JAMS': 2}
        result[Product.BASKET1] = self.enhanced_basket_arbitrage(state, Product.BASKET1, b1_comps)
        result[Product.BASKET2] = self.enhanced_basket_arbitrage(state, Product.BASKET2, b2_comps)
        
        # 4. Execute R4 Regime Switching engine
        for prod in Mean_Revert:
            result[prod] = self.regime_switching_orders(state, prod)

        return result, 1, ""