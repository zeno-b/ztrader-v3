"""SQL query definitions for trading and training persistence."""

from __future__ import annotations

INSERT_DECISION_LOG = """
INSERT INTO decision_log (
    id,
    timestamp,
    agent_id,
    task_id,
    asset,
    signal_type,
    signal_value,
    confidence,
    reasoning,
    data_sources,
    market_regime,
    outcome_pnl,
    outcome_latency_days,
    contributed_to_trade,
    trade_was_profitable
) VALUES (
    %(id)s,
    %(timestamp)s,
    %(agent_id)s,
    %(task_id)s,
    %(asset)s,
    %(signal_type)s,
    %(signal_value)s,
    %(confidence)s,
    %(reasoning)s,
    %(data_sources)s,
    %(market_regime)s,
    %(outcome_pnl)s,
    %(outcome_latency_days)s,
    %(contributed_to_trade)s,
    %(trade_was_profitable)s
)
"""


UPDATE_DECISION_OUTCOME_IF_NULL = """
UPDATE decision_log
SET
    outcome_pnl = %(outcome_pnl)s,
    outcome_latency_days = %(outcome_latency_days)s,
    trade_was_profitable = %(trade_was_profitable)s
WHERE id = %(id)s AND outcome_pnl IS NULL
"""


COUNT_OUTCOME_READY_RECORDS = """
SELECT COUNT(*) AS total
FROM decision_log
WHERE outcome_pnl IS NOT NULL
  AND trade_was_profitable IS NOT NULL
"""
