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
    max_stale_minutes: int = 180

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
