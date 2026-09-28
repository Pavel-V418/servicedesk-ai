"""
Маршрутизация обращения по линиям поддержки (L1/L2/L3) — routing-agent коллеги.

    from routing import TicketRouter
Исходный код коллеги: routing/routing_agent/ (README.md там же).
"""
from routing.router import TicketRouter, build_routing_ticket, LINE_NAMES, STATUS_NAMES

__all__ = ["TicketRouter", "build_routing_ticket", "LINE_NAMES", "STATUS_NAMES"]
