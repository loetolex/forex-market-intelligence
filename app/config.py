from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
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
    forecast_outputsize: int = 500
    max_stale_minutes: int = 180

    # Twelve Data free-tier protection.
    # Keep a safety margin below the documented 8 credits/minute limit.
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
