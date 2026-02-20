# trading-crew

Fully local, containerized, multi-agent AI trading system with strict isolation
between trading and training pipelines.

## Design goals

- Reproducible, auditable, and risk-controlled decisions
- Stateless agent tasks with explicit state passing
- Offline-capable local inference via Ollama
- Self-improving adapters (LoRA only), no base model mutation
- Strict training/trading isolation

## Runtime modes

- **Trading mode**: market data ingestion, signal generation, risk checks,
  order execution, and decision logging.
- **Training mode**: dataset build, LoRA fine-tuning, evaluation, and
  promotion workflow.

Both modes run concurrently but are isolated by queue boundaries and separate
containers.

## Exchange execution support

- Paper mode is the default (`LIVE_TRADING=false`).
- Live crypto execution supports Kraken via API key/secret.
- Set `TRADING_EXCHANGE=kraken` and provide `KRAKEN_API_KEY` +
  `KRAKEN_API_SECRET`.
- Kraken symbols should use ccxt format (example: `BTC/USD`).

## Quick start

1. Copy environment template:

   ```bash
   cp .env.example .env
   ```

2. Start infrastructure:

   ```bash
   docker compose up -d timescaledb redis minio ollama mlflow
   ```

3. Run migrations (placeholder command):

   ```bash
   make migrate
   ```

4. Start services:

   ```bash
   docker compose up -d trading training
   ```

## Testing

```bash
make test
```
