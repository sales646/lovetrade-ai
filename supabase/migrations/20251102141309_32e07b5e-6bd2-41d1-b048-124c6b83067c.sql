-- Add Alpaca API credentials to bot_config table
ALTER TABLE bot_config
ADD COLUMN IF NOT EXISTS alpaca_api_key TEXT,
ADD COLUMN IF NOT EXISTS alpaca_secret_key TEXT,
ADD COLUMN IF NOT EXISTS alpaca_paper_trading BOOLEAN DEFAULT true;

COMMENT ON COLUMN bot_config.alpaca_api_key IS 'Alpaca API Key for live trading';
COMMENT ON COLUMN bot_config.alpaca_secret_key IS 'Alpaca Secret Key for live trading';
COMMENT ON COLUMN bot_config.alpaca_paper_trading IS 'Whether to use paper trading (true) or live trading (false)';