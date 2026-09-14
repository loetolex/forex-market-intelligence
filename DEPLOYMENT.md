# Deployment checklist

## Paper/demo only

1. Connect this GitHub repository to Railway.
2. Set `TWELVE_DATA_API_KEY` in Railway Variables.
3. Keep:
   - `TRADING_MODE=PAPER`
   - `LIVE_TRADING_ENABLED=false`
   - `ORDER_PLACEMENT_ENABLED=false`
4. Deploy.
5. Verify `/health`.
6. Verify `/market/EURUSD`.
7. Confirm logs show real provider data and no order calls.

Do not add live broker credentials at this stage.
