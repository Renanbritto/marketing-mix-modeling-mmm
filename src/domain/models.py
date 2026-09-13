"""Domain models, enumerations, and core analytical entities."""

from enum import Enum
from typing import NamedTuple, Dict, List


class MediaChannel(str, Enum):
    """Supported media investment channels."""
    META_ADS = "meta_ads"
    GOOGLE_ADS = "google_ads"
    TIKTOK_ADS = "tiktok_ads"
    INFLUENCERS = "influencers"
    CRM_EMAIL = "crm_email"


CHANNEL_DISPLAY_NAMES: Dict[str, str] = {
    "meta_ads": "Meta Ads (Facebook & Instagram)",
    "google_ads": "Google Ads (Search & Display)",
    "tiktok_ads": "TikTok Ads",
    "influencers": "Influenciadores & Creators",
    "crm_email": "CRM & E-mail Marketing",
}


class ChannelHealth(str, Enum):
    """Channel investment efficiency classification."""
    SCALING = "SCALING"
    EFFICIENT = "EFFICIENT"
    SATURATED = "SATURATED"
    OVER_INVESTED = "OVER_INVESTED"


class AdstockResult(NamedTuple):
    """Result of Adstock temporal transformation."""
    transformed_series: list
    decay_rate: float
    halflife_periods: float


class HillResult(NamedTuple):
    """Result of Hill saturation transformation."""
    transformed_series: list
    half_saturation_K: float
    shape_S: float


class ChannelContribution:
    """Revenue contribution and efficiency metrics for a single media channel."""

    def __init__(
        self,
        channel: str,
        total_spend: float,
        attributed_revenue: float,
        roas: float,
        marginal_roas: float,
        contribution_pct: float,
        health: ChannelHealth,
    ):
        self.channel = channel
        self.display_name = CHANNEL_DISPLAY_NAMES.get(channel, channel)
        self.total_spend = total_spend
        self.attributed_revenue = attributed_revenue
        self.roas = roas
        self.marginal_roas = marginal_roas
        self.contribution_pct = contribution_pct
        self.health = health

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "display_name": self.display_name,
            "total_spend": round(self.total_spend, 2),
            "attributed_revenue": round(self.attributed_revenue, 2),
            "roas": round(self.roas, 2),
            "marginal_roas": round(self.marginal_roas, 4),
            "contribution_pct": round(self.contribution_pct, 2),
            "health": self.health.value,
        }
