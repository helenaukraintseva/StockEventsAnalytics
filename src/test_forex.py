import pandas as pd


class TimeSeriesSimulator:
    def __init__(self, dataset: pd.DataFrame, time_column: str, value_columns=False, initial_balance=1000.0, fee_rate=0.02):
        self.dataset = dataset.sort_values(by=time_column).reset_index(drop=True)
        self.time_column = time_column
        self.value_columns = value_columns if value_columns else ["open", "high", "low", "close", "volume"]
        self.current_index = 0
        self.total_steps = len(self.dataset)
        self.history = []
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.wallets = {}
        self.pending_signals = {}  # для отслеживания ожиданий по моделям

    def reset(self):
        self.current_index = 0
        self.history = []
        self.wallets = {}

    def step(self):
        if self.current_index < self.total_steps:
            row = self.dataset.iloc[self.current_index]
            self.history.append(row)
            self.current_index += 1
            return row
        return None

    def get_history(self, window_size=20):

        df = pd.DataFrame(self.history[-window_size:])
        return df[[self.time_column] + self.value_columns]

    def is_done(self):
        return self.current_index >= self.total_steps

    def _init_wallet(self, model_name):
        self.wallets[model_name] = {
            "cash": self.initial_balance,
            "asset": 0.0,
            "trade_log": []
        }
        self.pending_signals[model_name] = None

    def _execute_trade(self, wallet, action, price):
        if action == "buy" and wallet["cash"] > 0:
            amount = wallet["cash"] * (1 - self.fee_rate)
            wallet["asset"] = amount / price
            wallet["cash"] = 0.0
            wallet["trade_log"].append({"action": "buy", "price": price})
        elif action == "sell" and wallet["asset"] > 0:
            proceeds = wallet["asset"] * price * (1 - self.fee_rate)
            wallet["cash"] = proceeds
            wallet["asset"] = 0.0
            wallet["trade_log"].append({"action": "sell", "price": price})

    def run(self, models: list, window_size: int = 20, verbose: bool = True, lookahead: int = 5):
        self.reset()
        log = []

        model_names = [model.__class__.__name__ for model in models]
        for name in model_names:
            self._init_wallet(name)

        while not self.is_done():
            row = self.step()
            current_time = row[self.time_column]
            current_price = row[self.value_columns]
            history = self.get_history(window_size=window_size)

            step_result = {"time": current_time, "price": current_price, "signals": {}}

            for model in models:
                model_name = model.__class__.__name__
                signal_info = model.analyze(history)
                action = signal_info.get("signal", "hold")
                wallet = self.wallets[model_name]

                # Если нет отложенного сигнала — принимаем новый
                if self.pending_signals[model_name] is None and action in ["buy", "sell"]:
                    # Собираем lookahead-окно для поиска лучшей цены
                    lookahead_window = self.dataset.iloc[self.current_index:self.current_index + lookahead]
                    if lookahead_window.empty or "close" not in lookahead_window.columns:
                        best_price = None
                        continue
                    if action == "buy":
                        best_row = lookahead_window.loc[lookahead_window["close"].idxmax()]
                    else:
                        best_row = lookahead_window.loc[lookahead_window["close"].idxmin()]

                    # Исполняем сделку на лучшей цене
                    best_price = best_row["close"]
                    best_price = best_price.item()
                    best_time = best_row[self.time_column]
                    self._execute_trade(wallet, action, best_price)

                    # if verbose:
                    #     print(
                    #         f"  ✅ {model_name} {action.upper()} @ {best_price:.2f} (optimal after signal at {current_time})")

                    # Пропускаем lookahead шагов, имитируя время ожидания
                    self.current_index = self.dataset.index.get_loc(best_row.name) + 1
                    break  # выходим из модели — шаг уже смещён

                step_result["signals"][model_name] = signal_info

            log.append(step_result)

        return log

    def evaluate(self):
        """Возвращает метрики качества по каждому кошельку."""
        report = {}

        for model_name, wallet in self.wallets.items():
            final_cash = wallet["cash"] + wallet["asset"] * self.history[-1]["close"]
            trades = wallet["trade_log"]
            num_trades = len(trades)
            pnl = final_cash - self.initial_balance
            roi = (final_cash / self.initial_balance - 1) * 100 if self.initial_balance else 0

            # Подсчёт win-rate
            wins = 0
            for i in range(1, len(trades), 2):
                buy_price = trades[i-1]["price"]
                sell_price = trades[i]["price"]
                if sell_price > buy_price:
                    wins += 1
            win_rate = wins / (len(trades) // 2) if len(trades) >= 2 else 0

            report[model_name] = {
                "Final Balance": round(final_cash, 2),
                "PnL": round(pnl, 2),
                "ROI (%)": round(roi, 2),
                "Trades": num_trades,
                "Win Rate": round(win_rate * 100, 2)
            }

        return report
