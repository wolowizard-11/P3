from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Trade, Listing, Observation, ProsperityEncoder
from typing import List, Dict, Any
import json
import numpy as np
from collections import deque

class Logger:
    """
    Handles state serialization and efficient log compression 
    to respect structural platform size limitations.
    """
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.symbol, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice, observation.askPrice, observation.transportFees,
                observation.exportTariff, observation.importTariff, observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        # Global Engine Configuration Limits
        self.max_buy = 10000
        self.max_sell = 10000
        self.history_size = 300  
        
        # Microstructural Memory Structures
        self.buy_prices = {}  
        self.sell_prices = {}
        self.product_data_history = {}  
        self.kelp_buys = {}

    def calc_sigma(self, prices: list, window: int) -> float:
        """Calculates rolling population standard deviation across target windows."""
        if len(prices) < window:
            return None
        return np.std(prices[-window:])
    
    def calculate_fair_value(self, order_depth: OrderDepth) -> float:
        """Derives a noise-filtered fair value via large institutional size depths."""
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        
        # Discard transient retail noise orders below 15 contract clip thresholds
        filtered_ask = [p for p in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[p]) >= 15]
        filtered_bid = [p for p in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[p]) >= 15]
        
        mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
        mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
        
        return (mm_ask + mm_bid) / 2

    def update_history(self, product: str, timestamp: int, buy: float, sell: float, mid_price: float) -> None:
        """Appends streaming state information into linear rolling queues."""
        if product not in self.product_data_history:
            self.product_data_history[product] = []

        if product not in self.buy_prices:
            self.buy_prices[product] = deque(maxlen=self.history_size)
        if product not in self.sell_prices:
            self.sell_prices[product] = deque(maxlen=self.history_size)

        self.buy_prices[product].append(buy)
        self.sell_prices[product].append(sell)
        self.product_data_history[product].append(mid_price)
    
    def resin_orders(self, state: TradingState) -> List[Order]:
        """Stationary Market Making Strategy for RAINFOREST_RESIN."""
        orders = []
        product = "RAINFOREST_RESIN"
        limit = 50
        
        # Strategy Parameters
        fair_value = 10000
        take_width = 2
        clear_width = 1.5
        disregard_edge = 0.5
        default_edge = 3
        join_edge = 2
        soft_position_limit = 50

        position = state.position.get(product, 0)
        if product in state.order_depths:
            order_depth = state.order_depths[product]
            buy_order_volume = 0
            sell_order_volume = 0

            # 1. Market-Taking Execution Layer
            best_ask = min(order_depth.sell_orders.keys(), default=None)
            best_bid = max(order_depth.buy_orders.keys(), default=None)

            if best_ask is not None and best_ask <= fair_value - take_width:
                quantity = min(-order_depth.sell_orders[best_ask], limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity

            if best_bid is not None and best_bid >= fair_value + take_width:
                quantity = min(order_depth.buy_orders[best_bid], limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity

            # 2. Position Clearing / Active Risk Offloading Layer
            position_after_take = position + buy_order_volume - sell_order_volume
            fair_for_bid = round(fair_value - clear_width)
            fair_for_ask = round(fair_value + clear_width)

            buy_quantity = limit - (position + buy_order_volume)
            sell_quantity = limit + (position - sell_order_volume)

            if position_after_take > 0:
                clear_quantity = sum(v for p, v in order_depth.buy_orders.items() if p >= fair_for_ask)
                clear_quantity = min(clear_quantity, position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_ask, -sent_quantity))
                    sell_order_volume += sent_quantity

            if position_after_take < 0:
                clear_quantity = sum(-v for p, v in order_depth.sell_orders.items() if p <= fair_for_bid)
                clear_quantity = min(clear_quantity, abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_bid, sent_quantity))
                    buy_order_volume += sent_quantity

            # 3. Passive Market Making Quoting Layer
            asks_above_fair = [p for p in order_depth.sell_orders if p > fair_value + disregard_edge]
            bids_below_fair = [p for p in order_depth.buy_orders if p < fair_value - disregard_edge]

            ask = round(fair_value + default_edge)
            if asks_above_fair:
                best = min(asks_above_fair)
                ask = best if abs(best - fair_value) <= join_edge else best - 1

            bid = round(fair_value - default_edge)
            if bids_below_fair:
                best = max(bids_below_fair)
                bid = best if abs(fair_value - best) <= join_edge else best + 1

            # Inventory Skew Pricing Adjustments
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1

            if limit - (position + buy_order_volume) > 0:
                quantity = limit - (position + buy_order_volume)
                orders.append(Order(product, bid, quantity))

            if limit + (position - sell_order_volume) > 0:
                quantity = limit + (position - sell_order_volume)
                orders.append(Order(product, ask, -quantity))

        return orders

    def update_kelp_orders(self, state: TradingState) -> None:
        """Internal inventory trade ledger updating for KELP tracking blocks."""
        symbol = 'KELP'
        own_trades = state.own_trades
        if symbol in own_trades.keys():
            own_trades = state.own_trades[symbol]
            if own_trades and len(own_trades) > 0:
                for trade in own_trades:
                    if trade.symbol == symbol:
                        if trade.buyer == 'SUBMISSION':
                            self.kelp_buys[trade.price] = trade.quantity
                        if trade.seller == 'SUBMISSION' and len(self.kelp_buys) > 0:
                            if trade.price >= min(self.kelp_buys.keys()):
                                if trade.quantity >= self.kelp_buys[min(self.kelp_buys.keys())]:
                                    del self.kelp_buys[min(self.kelp_buys.keys())] 
                                else:
                                    self.kelp_buys[min(self.kelp_buys.keys())] -= trade.quantity

    def kelp_orders(self, state: TradingState) -> List[Order]:
        """Adaptive Microstructure Market Making Strategy for KELP."""
        product = 'KELP'
        limit = 50  
        orders = []
        position = state.position.get(product, 0)
        
        if product not in state.order_depths:
            return orders
            
        order_depth = state.order_depths[product]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
            
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Compute Noise-Filtered Fair Metric Base
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        
        filtered_ask = [p for p in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[p]) >= 15]
        filtered_bid = [p for p in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[p]) >= 15]
        
        mm_ask = min(filtered_ask) if filtered_ask else best_ask
        mm_bid = max(filtered_bid) if filtered_bid else best_bid
        fair_value = (mm_ask + mm_bid) / 2
        
        if product not in self.product_data_history:
            self.product_data_history[product] = []
        self.product_data_history[product].append(fair_value)
        
        # Real-time Order Book VWAP Architecture Check
        volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
        vwap = (best_bid * -order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
        
        # Market-Taking Core with Adverse Selection Shielding
        take_width = 1
        if best_ask <= fair_value - take_width:
            quantity = min(-order_depth.sell_orders[best_ask], limit - position)
            if quantity > 0 and best_ask <= vwap - 20:
                orders.append(Order(product, best_ask, quantity))
                buy_order_volume += quantity
                
        if best_bid >= fair_value + take_width:
            quantity = min(order_depth.buy_orders[best_bid], limit + position)
            if quantity > 0 and best_bid >= vwap + 20:
                orders.append(Order(product, best_bid, -quantity))
                sell_order_volume += quantity
        
        # Position Clearing Module Logic
        position_after_take = position + buy_order_volume - sell_order_volume
        clear_width = 2
        fair_for_bid = round(fair_value - clear_width)
        fair_for_ask = round(fair_value + clear_width)
        
        if position_after_take > 0:
            clear_quantity = sum(v for p, v in order_depth.buy_orders.items() if p >= fair_for_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(limit + (position - sell_order_volume), clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -sent_quantity))
                sell_order_volume += sent_quantity
                
        if position_after_take < 0:
            clear_quantity = sum(-v for p, v in order_depth.sell_orders.items() if p <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(limit - (position + buy_order_volume), clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, sent_quantity))
                buy_order_volume += sent_quantity
        
        # Market-Making Inside Quoting Placement
        asks_above_fair = [p for p in order_depth.sell_orders if p > fair_value + 1]
        bids_below_fair = [p for p in order_depth.buy_orders if p < fair_value - 1]
        
        best_ask_filtered = min(asks_above_fair) if asks_above_fair else fair_value + 2
        best_bid_filtered = max(bids_below_fair) if bids_below_fair else fair_value - 2
        
        remaining_buy = limit - (position + buy_order_volume)
        remaining_sell = limit + (position - sell_order_volume)
        
        if remaining_buy > 0:
            orders.append(Order(product, best_bid_filtered + 1, remaining_buy))
            
        if remaining_sell > 0:
            orders.append(Order(product, best_ask_filtered - 1, -remaining_sell))
        
        return orders
    
    def ink_orders(self, state: TradingState) -> List[Order]:
        """Adaptive ATR Mean Reversion Strategy for SQUID_INK."""
        product = 'SQUID_INK'
        limit = 50
        position = state.position.get(product, 0)
        orders = []
        
        # Strategic Parameters and Memory Initialization
        if not hasattr(self, 'initial_profit_threshold'):
            self.initial_profit_threshold = 700  
        if not hasattr(self, 'profit_protection_percentage'):
            self.profit_protection_percentage = 0.6  
        if not hasattr(self, 'monitoring_position_size'):
            self.monitoring_position_size = 3  
        
        if not hasattr(self, 'cumulative_pnl'):
            self.cumulative_pnl = 0
        if not hasattr(self, 'high_water_mark'):
            self.high_water_mark = 0
        if not hasattr(self, 'current_profit_threshold'):
            self.current_profit_threshold = self.initial_profit_threshold
        if not hasattr(self, 'last_mid_price'):
            self.last_mid_price = None
        if not hasattr(self, 'last_position'):
            self.last_position = 0
        if not hasattr(self, 'threshold_activated'):
            self.threshold_activated = False
        if not hasattr(self, 'trading_mode'):
            self.trading_mode = "FULL_TRADING"  
        
        if product not in state.order_depths:
            return orders
        
        order_depth = state.order_depths[product]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        current_price = (best_ask + best_bid) / 2

        # Mark-to-Market Internal Accounting Matrix Update
        if self.last_mid_price is not None:
            price_change = current_price - self.last_mid_price
            self.cumulative_pnl += self.last_position * price_change
        self.last_mid_price = current_price
        self.last_position = position

        # Statistical Channel Calculation Framework
        prices = self.product_data_history.get(product, [])
        moving_avg = sum(prices) / len(prices) if prices else current_price
        deviation = current_price - moving_avg
        std = np.std(prices[-5:]) if len(prices) >= 5 else 0
        
        # 14-Period Volatility Scaling Channel Bounds (ATR Mapping)
        lookback_atr = 14  
        if len(prices) >= lookback_atr:
            atr = np.mean([abs(prices[i] - prices[i-1]) for i in range(len(prices)-lookback_atr+1, len(prices))])
        else:
            atr = std  
            
        deviation_threshold = atr * 6 
        effective_limit = limit

        # Mean-Reversion Aggressive Liquidation Routing Loops
        if deviation > deviation_threshold and position > -effective_limit:
            sell_qty = effective_limit + position
            if sell_qty > 0:
                orders.append(Order(product, best_bid, -sell_qty))
                
        elif deviation < -deviation_threshold and position < effective_limit:
            buy_qty = effective_limit - position
            if buy_qty > 0:
                orders.append(Order(product, best_ask, buy_qty))
                
        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """Main orchestrator block managing data flow pipelines and executions."""
        result = {}
        trader_data = ""
        timestamp = state.timestamp

        # Parse streaming metrics across market books
        self.update_kelp_orders(state)
        for product, order_depth in state.order_depths.items():
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                buys, sells = order_depth.buy_orders, order_depth.sell_orders

                # Microstructural Volume Weighted Mid Pricing
                mid_price = (best_ask * abs(sells[best_ask]) + best_bid * abs(buys[best_bid])) / (abs(buys[best_bid]) + abs(sells[best_ask]))
                self.update_history(product, timestamp, best_ask, best_bid, mid_price)
            else:
                continue  

        # Execute Round 1 Subroutines
        result['RAINFOREST_RESIN'] = self.resin_orders(state)
        result['SQUID_INK'] = self.ink_orders(state)
        result['KELP'] = self.kelp_orders(state)

        conversions = 1
        # To run log processing streams, uncomment the engine level compression engine loop below:
        # logger.flush(state, result, conversions, trader_data)
        
        return result, conversions, trader_data