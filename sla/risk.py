from __future__ import annotations

from typing import Optional

import pandas as pd

LINE_CODE_TO_GROUP = {"L1": "(1 линия)", "L2": "(2 линия)", "L3": "(3 линия)"}

HIGH_RISK_RATE = 0.10     # >= 10% просрочек в истории похожих обращений
MEDIUM_RISK_RATE = 0.03   # >= 3% (средний по выгрузке ~1.2%)
LEVEL_NAMES = {"high": "высокий", "medium": "средний", "low": "низкий"}


class SLARiskEstimator:
    def __init__(self, analyzer=None, min_support: int = 15, min_support_duration: int = 10):
        if analyzer is None:
            from sla.sla_analytics import SLAAnalyzer
            analyzer = SLAAnalyzer()
        self.df = analyzer.df
        self.min_support = min_support
        self.min_support_duration = min_support_duration
        self.base_rate = float(self.df['is_overdue'].mean())
        self._priorities = sorted(self.df['Приоритет'].dropna().unique().tolist())

    # ------------------------------------------------------------------
    def _match_priority(self, priority: Optional[str]) -> Optional[str]:
        """'Средний' / '3' / '(3) Средний' -> '(3) Средний' (как в выгрузке)."""
        if not priority:
            return None
        p = str(priority).strip().lower()
        for value in self._priorities:
            v = value.lower()
            if p == v or p in v or p.strip('()') == v[1:2]:
                return value
        return None

    def _group_stat(self, mask) -> tuple[int, int]:
        sub = self.df.loc[mask, 'is_overdue']
        return int(sub.sum()), int(len(sub))

    def _expected_hours(self, category: Optional[str], line_group: Optional[str]) -> dict:
        df = self.df
        candidates = []
        if category and line_group:
            candidates.append((f"вид '{category}' на {line_group}",
                               (df['Вид запроса'] == category) & (df['Кем решен (группа)'] == line_group)))
        if category:
            candidates.append((f"вид '{category}'", df['Вид запроса'] == category))
        candidates.append(("все обращения", pd.Series(True, index=df.index)))
        for label, mask in candidates:
            hours = df.loc[mask, 'duration_hours']
            if len(hours) >= self.min_support_duration:
                return {"basis": label, "n": int(len(hours)),
                        "median": round(float(hours.median()), 1),
                        "p90": round(float(hours.quantile(0.9)), 1)}
        return {}

    # ------------------------------------------------------------------
    def assess(self, priority: Optional[str] = None, line_code: Optional[str] = None,
               category: Optional[str] = None) -> dict:
        df = self.df
        prio = self._match_priority(priority)
        line_group = LINE_CODE_TO_GROUP.get(line_code or "")
        factors = []

        # 1-2. Приоритет (+ линия, если истории достаточно)
        if prio:
            used = False
            if line_group:
                overdue, total = self._group_stat((df['Приоритет'] == prio) & (df['Кем решен (группа)'] == line_group))
                if total >= self.min_support:
                    factors.append((f"приоритет {prio} на {line_group}", overdue, total))
                    used = True
            if not used:
                overdue, total = self._group_stat(df['Приоритет'] == prio)
                if total >= self.min_support:
                    factors.append((f"приоритет {prio}", overdue, total))
        elif line_group:
            overdue, total = self._group_stat(df['Кем решен (группа)'] == line_group)
            factors.append((f"{line_group} (приоритет не указан)", overdue, total))

        # 3. Вид запроса
        if category:
            overdue, total = self._group_stat(df['Вид запроса'] == category)
            if total >= self.min_support:
                factors.append((f"вид запроса '{category}'", overdue, total))

        if factors:
            label, overdue, total = max(factors, key=lambda f: f[1] / f[2])
            rate = overdue / total
        else:
            label, overdue, total = "все обращения", int(df['is_overdue'].sum()), len(df)
            rate = self.base_rate

        level = "high" if rate >= HIGH_RISK_RATE else "medium" if rate >= MEDIUM_RISK_RATE else "low"
        reason = (f"В истории просрочено {overdue} из {total} обращений ({rate:.1%}) "
                  f"по признаку: {label}; в среднем по выгрузке {self.base_rate:.1%}.")
        if not prio:
            reason += " Приоритет не указан — оценка неполная."

        expected = self._expected_hours(category, line_group)
        return {
            "sla_risk_level": LEVEL_NAMES[level],
            "sla_risk_score": round(rate, 4),
            "sla_risk_reason": reason,
            "sla_expected_hours": expected,
            "sla_factors": [
                {"factor": f, "overdue": o, "total": t, "rate": round(o / t, 4)} for f, o, t in factors
            ],
        }
