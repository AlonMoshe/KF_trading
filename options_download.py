# ib_option_5sec_download_fixed.py
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import pandas as pd
import time


class OptionDownloader(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

        self.data = []
        self.req_id = 1

    def nextValidId(self, orderId):
        print("Connected. Requesting data...")

        # --- Define the Option Contract ---
        contract = Contract()
        contract.symbol = "AAPL"
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = "20251219"
        contract.right = "P"
        contract.strike = 280

        # --- Request 1 day of 5-second bars ---
        self.reqHistoricalData(
            reqId=self.req_id,
            contract=contract,
            endDateTime = "20251202 23:59:59 US/Eastern",
            durationStr="3 D",
            barSizeSetting="5 secs",
            whatToShow="TRADES",                 # or "TRADES", "MIDPOINT"
            useRTH=0,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

    def historicalData(self, reqId, bar):
        self.data.append({
            "Date": bar.date,
            "Open": bar.open,
            "High": bar.high,
            "Low": bar.low,
            "Close": bar.close,
            "Volume": bar.volume,
            "WAP": bar.wap
            # Removed bar.count (not available for options)
        })

    def historicalDataEnd(self, reqId, start, end):
        print("Download complete.")

        df = pd.DataFrame(self.data)

        df.to_csv(
            "C:/Users/User/OneDrive/Projects/TradingSystem/Data/Options/"
            "AAPL_280P19DEC2025_2025-12-02_5sec_3DAYS.csv",
            index=False
        )

        print("Saved CSV successfully.")
        self.disconnect()

    def error(self, reqId, errorTime, errorCode, errorString, advancedOrderReject=""):
        if errorCode in (2104, 2106, 2158):  # benign connection messages
            return
        print(f"Error {errorCode}: {errorString}")


def run_loop(app):
    app.run()


if __name__ == "__main__":
    app = OptionDownloader()
    app.connect("127.0.0.1", 7497, clientId=2)

    thread = threading.Thread(target=run_loop, args=(app,), daemon=True)
    thread.start()

    while app.isConnected():
        time.sleep(1)
