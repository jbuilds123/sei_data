from sqlalchemy import Column, Integer, String, Sequence, JSON, ForeignKey, Boolean
from .base import Base
from sqlalchemy.orm import relationship
from datetime import datetime
from sqlalchemy import Numeric


class NewPairs(Base):
    __tablename__ = 'newpairs'

    pair_address = Column(String(50), primary_key=True)
    created_at = Column(Integer)
    initial_scan_data = Column(JSON)
    later_scan_data = Column(JSON)
    is_verified = Column(Boolean, default=False)
    is_locked = Column(Boolean, default=False)
    is_renounced = Column(Boolean, default=False)
    in_dex = Column(Boolean, default=False)
    in_dex_time = Column(JSON)
    socials_data = Column(JSON)
    goplus_data = Column(JSON)
    honeypot_data = Column(JSON)
    gecko_chart_data = Column(JSON)
    dexscreener_data = Column(JSON)
    lock_data = Column(JSON)
    ohlcv_data = Column(JSON)
    is_honeypot = Column(Boolean, default=None)
    # Relationship to PairTrades
    trades = relationship("PaperTrades", back_populates="pair",
                          primaryjoin="NewPairs.pair_address==foreign(PaperTrades.pair_address)")


class PaperTrades(Base):
    __tablename__ = 'papertrades'

    pair_address = Column(String(50), ForeignKey(
        'newpairs.pair_address'), primary_key=True)
    created_at = Column(Integer)
    trade_details = Column(JSON)
    # Relationship to NewPairs
    pair = relationship("NewPairs", back_populates="trades")


class HistoricPairs(Base):
    __tablename__ = 'historicpairs'

    pair_address = Column(String(50), primary_key=True)
    created_at = Column(Integer)
    initial_scan_data = Column(JSON)
    later_scan_data = Column(JSON)
    is_verified = Column(Boolean, default=False)
    is_locked = Column(Boolean, default=False)
    is_renounced = Column(Boolean, default=False)
    in_dex = Column(Boolean, default=False)
    in_dex_time = Column(JSON)
    socials_data = Column(JSON)
    goplus_data = Column(JSON)
    honeypot_data = Column(JSON)
    gecko_chart_data = Column(JSON)
    dexscreener_data = Column(JSON)
    lock_data = Column(JSON)
    ohlcv_data = Column(JSON)
    is_honeypot = Column(Boolean, default=None)
