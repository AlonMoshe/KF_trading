from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
import pandas as pd


class VIXFastDownloader(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []

    def nextValidId(self, orderId):
        print("Connected. Requesting 1-year 1-minute VIX data...")

        contract = Contract()
        contract.symbol = "VIX"
        contract.secType = "IND"
        contract.exchange = "CBOE"
        contract.currency = "USD"

        self.reqHistoricalData(
            reqId=1,
            contract=contract,
            endDateTime="",
            durationStr="3 D",         # ← FULL YEAR AT ONCE
            barSizeSetting="1 min",    # ← 1-MINUTE BARS
            whatToShow="TRADES",
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
            "Volume": bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        print("Completed download.")
        df = pd.DataFrame(self.data)
        df.to_csv(f"C:/Users/User/OneDrive/Projects/TradingSystem/Data/VIX_Data/VIX_2025_12_03_2025-12-05_1min.csv", index=False)
        print("Saved: VIX_1min_2025_12_02.csv")
        self.disconnect()


    def error(self, reqId, errorTime, errorCode, errorString, advancedOrderReject=""):
        if errorCode in (2104, 2106, 2158):
            return
        print(f"IB ERROR {errorCode}: {errorString}")
        
def run_loop(app):
    app.run()


if __name__ == "__main__":
    app = VIXFastDownloader()
    app.connect("127.0.0.1", 7497, clientId=12)

    thread = threading.Thread(target=run_loop, args=(app,), daemon=True)
    thread.start()

    while app.isConnected():
        time.sleep(1)
