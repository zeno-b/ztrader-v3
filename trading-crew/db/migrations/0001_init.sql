-- Initial schema bootstrap for trading-crew.

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS decision_log (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    agent_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    signal_value JSONB NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    reasoning TEXT NOT NULL,
    data_sources TEXT[] NOT NULL DEFAULT '{}',
    market_regime TEXT NOT NULL,
    outcome_pnl DOUBLE PRECISION NULL,
    outcome_latency_days INTEGER NULL,
    contributed_to_trade BOOLEAN NOT NULL DEFAULT FALSE,
    trade_was_profitable BOOLEAN NULL
);

SELECT create_hypertable('decision_log', by_range('timestamp'), if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS coordinator_weight_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    agent_id TEXT NOT NULL,
    weight DOUBLE PRECISION NOT NULL
);
