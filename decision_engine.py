from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Decision(str, Enum):
    ALLOW = "ALLOW"
    ALLOW_REDUCED = "ALLOW_REDUCED"
    SKIP = "SKIP"
    LOCKOUT = "LOCKOUT"


@dataclass
class AccountRules:
    account_size: float = 50000.0
    daily_loss_limit: float = 1000.0
    max_trailing_drawdown: float = 2000.0
    max_contracts: int = 2
    risk_per_trade: float = 150.0
    max_consecutive_losses: int = 3
    max_trades_per_session: int = 5
    require_trend_alignment: bool = True
    allow_countertrend_but_reduce: bool = False


@dataclass
class SessionState:
    realized_pnl_today: float = 0.0
    unrealized_pnl_open: float = 0.0
    trailing_drawdown_used: float = 0.0
    consecutive_losses: int = 0
    trades_taken_this_session: int = 0
    cooldown_active: bool = False


@dataclass
class TradeSetup:
    symbol: str = "ES"
    side: str = "LONG"
    session: str = "RTH"
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    setup_tag: str = "trend_continuation"
    trend_aligned: bool = True
    setup_quality: str = "A"


@dataclass
class DecisionResult:
    decision: Decision
    max_contracts_allowed: int
    risk_per_contract: float
    total_risk_if_max_size: float
    reward_to_risk: Optional[float]
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score: int = 0


POINT_VALUE = {
    "ES": 50.0,
    "MES": 5.0,
    "NQ": 20.0,
    "MNQ": 2.0,
}


def _validate_side(side: str) -> str:
    side = side.upper().strip()
    if side not in {"LONG", "SHORT"}:
        raise ValueError("side must be LONG or SHORT")
    return side


def _symbol_value(symbol: str) -> float:
    symbol = symbol.upper().strip()
    if symbol not in POINT_VALUE:
        raise ValueError(f"unsupported symbol: {symbol}")
    return POINT_VALUE[symbol]


def risk_per_contract(setup: TradeSetup) -> float:
    side = _validate_side(setup.side)
    point_value = _symbol_value(setup.symbol)
    if side == "LONG":
        points_at_risk = setup.entry - setup.stop
        points_to_target = setup.target - setup.entry
    else:
        points_at_risk = setup.stop - setup.entry
        points_to_target = setup.entry - setup.target
    if points_at_risk <= 0:
        raise ValueError("invalid stop placement for trade side")
    if points_to_target <= 0:
        raise ValueError("invalid target placement for trade side")
    return points_at_risk * point_value


def reward_to_risk(setup: TradeSetup) -> float:
    side = _validate_side(setup.side)
    if side == "LONG":
        reward = setup.target - setup.entry
        risk = setup.entry - setup.stop
    else:
        reward = setup.entry - setup.target
        risk = setup.stop - setup.entry
    if risk <= 0:
        raise ValueError("risk must be positive")
    return reward / risk


def evaluate_trade(setup: TradeSetup, rules: AccountRules, state: SessionState) -> DecisionResult:
    reasons: List[str] = []
    warnings: List[str] = []
    score = 100

    rpc = risk_per_contract(setup)
    rr = reward_to_risk(setup)

    if state.cooldown_active:
        return DecisionResult(
            decision=Decision.LOCKOUT,
            max_contracts_allowed=0,
            risk_per_contract=rpc,
            total_risk_if_max_size=0.0,
            reward_to_risk=rr,
            reasons=["Cooldown is active after prior losses or rule break."],
            warnings=warnings,
            score=0,
        )

    if abs(state.realized_pnl_today) >= rules.daily_loss_limit and state.realized_pnl_today < 0:
        return DecisionResult(
            decision=Decision.LOCKOUT,
            max_contracts_allowed=0,
            risk_per_contract=rpc,
            total_risk_if_max_size=0.0,
            reward_to_risk=rr,
            reasons=["Daily loss limit already reached."],
            warnings=warnings,
            score=0,
        )

    if state.trailing_drawdown_used >= rules.max_trailing_drawdown:
        return DecisionResult(
            decision=Decision.LOCKOUT,
            max_contracts_allowed=0,
            risk_per_contract=rpc,
            total_risk_if_max_size=0.0,
            reward_to_risk=rr,
            reasons=["Maximum trailing drawdown threshold reached."],
            warnings=warnings,
            score=0,
        )

    if state.consecutive_losses >= rules.max_consecutive_losses:
        reasons.append("Maximum consecutive losses reached.")
        return DecisionResult(
            decision=Decision.LOCKOUT,
            max_contracts_allowed=0,
            risk_per_contract=rpc,
            total_risk_if_max_size=0.0,
            reward_to_risk=rr,
            reasons=reasons,
            warnings=warnings,
            score=10,
        )

    if state.trades_taken_this_session >= rules.max_trades_per_session:
        reasons.append("Maximum trades for this session reached.")
        return DecisionResult(
            decision=Decision.SKIP,
            max_contracts_allowed=0,
            risk_per_contract=rpc,
            total_risk_if_max_size=0.0,
            reward_to_risk=rr,
            reasons=reasons,
            warnings=warnings,
            score=20,
        )

    if rules.require_trend_alignment and not setup.trend_aligned:
        if rules.allow_countertrend_but_reduce:
            warnings.append("Countertrend trade allowed only with reduced size.")
            score -= 25
        else:
            return DecisionResult(
                decision=Decision.SKIP,
                max_contracts_allowed=0,
                risk_per_contract=rpc,
                total_risk_if_max_size=0.0,
                reward_to_risk=rr,
                reasons=["Trade is not aligned with required trend direction."],
                warnings=warnings,
                score=25,
            )

    if rr < 2.0:
        warnings.append("Reward-to-risk is below 2.0.")
        score -= 20

    quality = setup.setup_quality.upper().strip()
    if quality == "A":
        score += 0
    elif quality == "B":
        warnings.append("B-quality setup: consider reduced size.")
        score -= 15
    else:
        warnings.append("Low-quality setup.")
        score -= 30

    if rpc > rules.risk_per_trade:
        warnings.append("Risk per contract exceeds preferred risk per trade.")
        score -= 25

    remaining_daily_loss_buffer = rules.daily_loss_limit + min(state.realized_pnl_today, 0)
    if remaining_daily_loss_buffer <= 0:
        return DecisionResult(
            decision=Decision.LOCKOUT,
            max_contracts_allowed=0,
            risk_per_contract=rpc,
            total_risk_if_max_size=0.0,
            reward_to_risk=rr,
            reasons=["No daily loss buffer remaining."],
            warnings=warnings,
            score=0,
        )

    max_by_trade_risk = max(int(rules.risk_per_trade // rpc), 0)
    max_by_daily_buffer = max(int(remaining_daily_loss_buffer // rpc), 0)
    max_allowed = min(rules.max_contracts, max_by_trade_risk, max_by_daily_buffer)

    if max_allowed <= 0:
        return DecisionResult(
            decision=Decision.SKIP,
            max_contracts_allowed=0,
            risk_per_contract=rpc,
            total_risk_if_max_size=0.0,
            reward_to_risk=rr,
            reasons=["No contract size fits current risk limits."],
            warnings=warnings,
            score=max(score, 10),
        )

    total_risk = max_allowed * rpc

    if score >= 80 and max_allowed == rules.max_contracts:
        decision = Decision.ALLOW
        reasons.append("Setup meets quality and risk thresholds.")
    else:
        decision = Decision.ALLOW_REDUCED
        reasons.append("Trade is acceptable only with reduced size or extra caution.")

    return DecisionResult(
        decision=decision,
        max_contracts_allowed=max_allowed,
        risk_per_contract=round(rpc, 2),
        total_risk_if_max_size=round(total_risk, 2),
        reward_to_risk=round(rr, 2),
        reasons=reasons,
        warnings=warnings,
        score=max(min(score, 100), 0),
    )


if __name__ == "__main__":
    rules = AccountRules(
        account_size=50000,
        daily_loss_limit=1000,
        max_trailing_drawdown=2000,
        max_contracts=2,
        risk_per_trade=150,
        max_consecutive_losses=3,
        max_trades_per_session=5,
        require_trend_alignment=True,
        allow_countertrend_but_reduce=False,
    )

    state = SessionState(
        realized_pnl_today=-250,
        trailing_drawdown_used=400,
        consecutive_losses=1,
        trades_taken_this_session=2,
        cooldown_active=False,
    )

    setup = TradeSetup(
        symbol="ES",
        side="LONG",
        session="RTH",
        entry=5210.25,
        stop=5208.25,
        target=5214.75,
        setup_tag="trend_continuation",
        trend_aligned=True,
        setup_quality="A",
    )

    result = evaluate_trade(setup, rules, state)
    print(result)
