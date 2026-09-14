from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Primary market-data provider for the production research pipeline.
    market_data_provider: str = "TIINGO"
    tiingo_api_token: str = ""

    # Secondary provider retained for independent cross-checks/failover research.
    twelve_data_api_key: str = ""
    secondary_market_data_provider: str = "TWELVE_DATA"

    fred_api_key: str = ""

    trading_mode: str = "PAPER"
    live_trading_enabled: bool = False
    order_placement_enabled: bool = False

    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    ibkr_client_id: int = 901

    primary_interval: str = "15m"
    # Operational multi-timeframe stack: 15m / 30m / 1h / 4h / 1day.
    # 90D remains a separate long-horizon research output, not an execution timeframe.
    forecast_intervals: list[str] = ["15m", "30m", "1h", "4h", "1day"]

    # Keep intraday history bounded for reliable real-time freshness.
    # Daily research uses the native Tiingo daily endpoint separately.
    tiingo_intraday_history_days: int = 365
    tiingo_daily_history_days: int = 1095
    tiingo_cache_ttl_seconds: int = 300

    # Canonical FX trading-day boundary for session validation/reporting.
    # Tiingo's documented FX market hours close at 5pm New York time.
    fx_daily_boundary_timezone: str = "America/New_York"
    fx_daily_boundary_hour_local: int = 17

    forecast_outputsize: int = 500
    max_stale_minutes: int = 180

    # Timeframe-specific freshness limits for the operational stack.
    freshness_limits_minutes: dict[str, int] = {
        "15m": 45,
        "30m": 90,
        "1h": 180,
        "4h": 1440,
        "1day": 1440,
    }

    # Twelve Data request protection for secondary diagnostics/cross-checks.
    # Keep a safety margin below the documented free-tier request ceiling.
    twelve_data_requests_per_minute: int = 7
    twelve_data_cache_ttl_seconds: int = 300
    twelve_data_max_429_retries: int = 1
    twelve_data_retry_wait_seconds: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
