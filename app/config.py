from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Primary market-data provider for the production research pipeline.
    market_data_provider: str = "TIINGO"
    tiingo_api_token: str = ""

    # Retained for compatibility / future secondary-provider fallback.
    twelve_data_api_key: str = ""
    fred_api_key: str = ""

    trading_mode: str = "PAPER"
    live_trading_enabled: bool = False
    order_placement_enabled: bool = False

    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    ibkr_client_id: int = 901

    primary_interval: str = "1h"
    forecast_intervals: list[str] = ["1h", "4h", "1day"]

    # Tiingo FX returns verified 1h historical OHLC. The 4h and 1day layers
    # are calculated from that 1h source so the model has a consistent clock.
    tiingo_history_days: int = 365
    tiingo_cache_ttl_seconds: int = 300

    forecast_outputsize: int = 500
    max_stale_minutes: int = 180

    # Legacy Twelve Data protection remains available while the provider is
    # being phased out of the primary pipeline.
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
