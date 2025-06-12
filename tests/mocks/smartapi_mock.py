class SmartConnect:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.refresh_token = None

    def generateSession(self, client_id, pin, totp):
        self.refresh_token = "dummy_token"
        return {"status": True, "message": "SUCCESS", "data": {"jwtToken": "dummy_token"}}

    def getProfile(self):
        return {"status": True, "message": "SUCCESS", "data": {"name": "Test User"}}

    def getAllSymbols(self):
        return [
            {"symbol": "TEST1", "token": "123", "exch_seg": "NSE"},
            {"symbol": "TEST2", "token": "456", "exch_seg": "NSE"}
        ]

    def getCandleData(self, params):
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        start = datetime.strptime(params["fromdate"], "%Y-%m-%d %H:%M")
        end = datetime.strptime(params["todate"], "%Y-%m-%d %H:%M")
        days = (end - start).days + 1

        data = []
        current = start
        for _ in range(days):
            close = 100 + np.random.normal(0, 1)
            data.append([
                current.strftime("%Y-%m-%d %H:%M"),  # timestamp
                close + np.random.normal(0, 0.1),    # open
                close + np.random.normal(0.5, 0.1),  # high
                close + np.random.normal(-0.5, 0.1), # low
                close,                               # close
                int(np.random.normal(10000, 1000))   # volume
            ])
            current += timedelta(days=1)

        return {"status": True, "data": data}

    def terminateSession(self, refresh_token):
        return {"status": True, "message": "Session terminated"}
