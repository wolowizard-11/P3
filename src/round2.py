from datamodel import OrderDepth, TradingState, Order, Symbol, Trade, Listing, Observation, ProsperityEncoder
from typing import List, Any, Dict
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

class Trader:
    def __init__(self):
        # Rolling Buffers and Memory Limits
        self.history_size = 300  
        self.buy_prices = {}  
        self.sell_prices = {}
        self.product_data_history = {}  
        self.starfruit_prices = []
        self.starfruit_vwap = []
        
        # Position Hard Limits per Product
        self.LIMIT = {
            Product.AMETHYSTS: 50,
            Product.STARFRUIT: 50,
            Product.BASKET1: 70,
            Product.BASKET2: 70
        }
        self.trade_history = {}

    def take_best_orders(self, product: str, fair_value: int, take_width: float, orders: List[Order], 
                         order_depth: OrderDepth, position: int, buy_order_volume: int, 
                         sell_order_volume: int, prevent_adverse: bool = False, adverse_volume: int = 0):
        """Crosses the spread to execute aggressive market-taking entries."""
        position_limit = self.LIMIT.get(product, 50)
        
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if (prevent_adverse and best_ask_amount <= adverse_volume) or not prevent_adverse:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, int(best_ask), quantity))
                        buy_order_volume += quantity

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if (prevent_adverse and best_bid_amount <= adverse_volume) or not prevent_adverse:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, int(best_bid), -1 * quantity))
                        sell_order_volume += quantity

        return buy_order_volume, sell_order_volume

    def market_make(self, product: str, orders: List[Order], bid: int, ask: int, position: int, 
                    buy_order_volume: int, sell_order_volume: int):
        """Places passive liquidity at defined edges to capture spread."""
        buy_quantity = self.LIMIT.get(product, 50) - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, int(bid), buy_quantity))  

        sell_quantity = self.LIMIT.get(product, 50) + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, int(ask), -sell_quantity))  

        return buy_order_volume, sell_order_volume

    def clear_position_order(self, product: str, fair_value: float, width: int, orders: List[Order], 
                             order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int) -> List[Order]:
        """Routes tight offset orders to prevent inventory toxicity."""
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = int(np.floor(fair_value))
        fair_for_ask = int(np.ceil(fair_value))

        buy_quantity = self.LIMIT.get(product, 50) - (position + buy_order_volume)
        sell_quantity = self.LIMIT.get(product, 50) + (position - sell_order_volume)

        if position_after_take > 0 and fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            orders.append(Order(product, int(fair_for_ask), -abs(sent_quantity)))
            sell_order_volume += abs(sent_quantity)

        if position_after_take < 0 and fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(product, int(fair_for_bid), abs(sent_quantity)))
            buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def amethyst_orders(self, state: TradingState) -> List[Order]:
        """Stationary Round 1 logic retained for AMETHYSTS."""
        orders = []
        product = Product.AMETHYSTS
        if product not in state.order_depths: return orders
        
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        fair_value = 10000
        
        buy_volume, sell_volume = 0, 0
        baaf = min([p for p in order_depth.sell_orders.keys() if p > fair_value + 1], default=fair_value + 2)
        bbbf = max([p for p in order_depth.buy_orders.keys() if p < fair_value - 1], default=fair_value - 2)

        buy_volume, sell_volume = self.take_best_orders(product, fair_value, 0.5, orders, order_depth, position, buy_volume, sell_volume)
        buy_volume, sell_volume = self.clear_position_order(product, fair_value, 1, orders, order_depth, position, buy_volume, sell_volume)
        self.market_make(product, orders, int(bbbf + 1), int(baaf - 1), position, buy_volume, sell_volume)

        return orders

    def starfruit_orders(self, state: TradingState) -> List[Order]:
        """Adaptive VWAP logic retained for STARFRUIT."""
        orders = []
        product = Product.STARFRUIT
        if product not in state.order_depths: return orders
        
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        
        if not order_depth.sell_orders or not order_depth.buy_orders: return orders
        
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        filtered_ask = [p for p in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[p]) >= 15]
        filtered_bid = [p for p in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[p]) >= 15]
        
        mm_ask = min(filtered_ask) if filtered_ask else best_ask
        mm_bid = max(filtered_bid) if filtered_bid else best_bid
        fair_value = (mm_ask + mm_bid) / 2

        buy_vol, sell_vol = 0, 0
        buy_vol, sell_vol = self.take_best_orders(product, int(fair_value), 1.0, orders, order_depth, position, buy_vol, sell_vol, True, 20)
        buy_vol, sell_vol = self.clear_position_order(product, fair_value, 2, orders, order_depth, position, buy_vol, sell_vol)
        
        baaf = min([p for p in order_depth.sell_orders.keys() if p > fair_value + 1], default=fair_value + 2)
        bbbf = max([p for p in order_depth.buy_orders.keys() if p < fair_value - 1], default=fair_value - 2)
        
        self.market_make(product, orders, int(bbbf + 1), int(baaf - 1), position, buy_vol, sell_vol)
        return orders

    def enhanced_basket_arbitrage(self, state: TradingState, basket: str, components: dict) -> List[Order]:
        """
        Calculates theoretical Fair Value by weighting underlying constituents, 
        trading deviations if the spread implies profitability.
        """
        orders = []
        if basket not in state.order_depths: return orders
        
        basket_depth = state.order_depths[basket]
        if not basket_depth.buy_orders or not basket_depth.sell_orders: return orders

        basket_position = state.position.get(basket, 0)
        max_position_size = self.LIMIT.get(basket, 70)
        
        basket_best_bid = max(basket_depth.buy_orders.keys())
        basket_best_ask = min(basket_depth.sell_orders.keys())
        
        # Determine theoretical fair value from order book depth
        basket_fair_value = 0
        for component, quantity in components.items():
            if component in state.order_depths:
                comp_depth = state.order_depths[component]
                if comp_depth.buy_orders and comp_depth.sell_orders:
                    # Top of book weighted average
                    best_c_bid = max(comp_depth.buy_orders.keys())
                    best_c_ask = min(comp_depth.sell_orders.keys())
                    basket_fair_value += ((best_c_bid + best_c_ask) / 2) * quantity
                else:
                    return orders # Abort if component data is missing
        
        # Scaling limits
        arb_threshold = 40  # Trigger threshold for basis spread 
        edge_buy = basket_fair_value - basket_best_ask
        edge_sell = basket_best_bid - basket_fair_value
        
        # Arbitrage 1: Basket is theoretically undervalued -> Buy Basket
        if basket_best_ask < basket_fair_value - arb_threshold and basket_position < max_position_size:
            max_basket_buy = min(abs(basket_depth.sell_orders[basket_best_ask]), max_position_size - basket_position)
            edge_factor = min(1.0, edge_buy / (arb_threshold * 3)) 
            basket_buy_qty = max(1, int(max_basket_buy * edge_factor))
            
            if basket_buy_qty > 0:
                orders.append(Order(basket, basket_best_ask, basket_buy_qty))
                
        # Arbitrage 2: Basket is theoretically overvalued -> Sell Basket
        elif basket_best_bid > basket_fair_value + arb_threshold and basket_position > -max_position_size:
            max_basket_sell = min(basket_depth.buy_orders[basket_best_bid], max_position_size + basket_position)
            edge_factor = min(1.0, edge_sell / (arb_threshold * 3))
            basket_sell_qty = max(1, int(max_basket_sell * edge_factor))
            
            if basket_sell_qty > 0:
                orders.append(Order(basket, basket_best_bid, -basket_sell_qty))
                
        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """Orchestrator block managing cycle-by-cycle routing."""
        result = {}
        
        # Round 1 Integrations
        result[Product.AMETHYSTS] = self.amethyst_orders(state)
        result[Product.STARFRUIT] = self.starfruit_orders(state)

        # Round 2 ETF Basket Arbitrage Definitions
        basket1_components = {
            'CROISSANTS': 6,
            'JAMS': 3,
            'DJEMBES': 1
        }
        basket2_components = {
            'CROISSANTS': 4,
            'JAMS': 2
        }
        
        result[Product.BASKET1] = self.enhanced_basket_arbitrage(state, Product.BASKET1, basket1_components)
        result[Product.BASKET2] = self.enhanced_basket_arbitrage(state, Product.BASKET2, basket2_components)

        # logger.flush(state, result, 1, "")
        return result, 1, ""