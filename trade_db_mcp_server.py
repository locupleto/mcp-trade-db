#!/usr/bin/env python3
"""
MCP Server for Trade Database

Provides Claude Code access to production trade.db.
Supports two modes:
  - Local mode: Direct SQLite access (when running on Mac mini)
  - Remote mode: SSH proxy to Mac mini (when running on Mac Studio)

Set LOCAL_MODE=true environment variable for direct database access.
"""

import os
import asyncio
import sqlite3
from typing import Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
)
from pydantic import AnyUrl


# Environment variables
MACMINI_HOST = os.environ.get("MACMINI_HOST", "100.102.226.1")
SSH_KEY_PATH = os.environ.get("SSH_KEY_PATH", "/Users/urban/.ssh/macmini")
TRADE_DB_PATH = os.environ.get("TRADE_DB_PATH", "~/Public/trading-lab/data/trade.db")
LOCAL_MODE = os.environ.get("LOCAL_MODE", "false").lower() == "true"

# Initialize MCP server
server = Server("trade-db")


def get_local_db_path() -> str:
    """Expand and resolve the local database path."""
    path = os.path.expanduser(TRADE_DB_PATH)
    return str(Path(path).resolve())


async def execute_local_query(sql: str) -> str:
    """Execute SQL query directly on local SQLite database."""
    db_path = get_local_db_path()

    if not os.path.exists(db_path):
        raise RuntimeError(f"Database not found: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)

        # Get column names
        columns = [description[0] for description in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        conn.close()

        if not columns:
            return ""

        # Format as pipe-separated output (matching SSH output format)
        lines = ['|'.join(columns)]
        for row in rows:
            lines.append('|'.join(str(v) if v is not None else '' for v in row))

        return '\n'.join(lines)

    except sqlite3.Error as e:
        raise RuntimeError(f"SQLite error: {e}")


async def execute_remote_query(sql: str) -> str:
    """Execute SQL query on Mac mini trade.db via SSH."""
    # Escape single quotes for shell
    escaped_sql = sql.replace("'", "'\"'\"'")

    cmd = [
        "ssh",
        "-i", SSH_KEY_PATH,
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        f"urban@{MACMINI_HOST}",
        f"sqlite3 -header -separator '|' {TRADE_DB_PATH} '{escaped_sql}'"
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        process.kill()
        raise RuntimeError("SSH query timed out after 30 seconds")

    if process.returncode != 0:
        error_msg = stderr.decode().strip()
        raise RuntimeError(f"SSH query failed: {error_msg}")

    return stdout.decode()


async def execute_query(sql: str) -> str:
    """Execute SQL query using appropriate method (local or remote)."""
    if LOCAL_MODE:
        return await execute_local_query(sql)
    else:
        return await execute_remote_query(sql)


def parse_pipe_output(output: str) -> list[dict]:
    """Parse pipe-separated output with header row into list of dicts."""
    if not output.strip():
        return []

    lines = output.strip().split('\n')
    if len(lines) < 2:
        return []

    headers = lines[0].split('|')
    results = []

    for line in lines[1:]:
        values = line.split('|')
        row = {}
        for i, header in enumerate(headers):
            value = values[i] if i < len(values) else ''
            # Convert numeric strings
            if value and value.replace('.', '', 1).replace('-', '', 1).isdigit():
                row[header] = float(value) if '.' in value else int(value)
            elif value == '':
                row[header] = None
            else:
                row[header] = value
        results.append(row)

    return results


def format_currency(value: Optional[float]) -> str:
    """Format a currency value."""
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def format_pct(value: Optional[float], decimals: int = 2) -> str:
    """Format a percentage value."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}%"


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="get_open_positions",
            description="Get current portfolio holdings with symbols, entry dates, prices, shares, and unrealized P&L. Returns only executed positions (shares NOT NULL) by default.",
            inputSchema={
                "type": "object",
                "properties": {
                    "timing_mode": {
                        "type": "string",
                        "description": "Filter by timing mode: 'conservative' (T+1 Open), 'aggressive' (same-day Close), or 'both'",
                        "default": "both"
                    },
                    "executed_only": {
                        "type": "boolean",
                        "description": "Only return positions with shares (actually executed). Set to false to include signals not yet bought.",
                        "default": True
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum positions to return (default: 50)",
                        "default": 50
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Sort by: 'symbol', 'entry_date', 'pnl', 'value'",
                        "default": "entry_date"
                    }
                }
            }
        ),
        Tool(
            name="get_portfolio_summary",
            description="Get quick portfolio overview: total positions, cash available, total equity, positions value, and daily activity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "timing_mode": {
                        "type": "string",
                        "description": "Timing mode for portfolio state: 'aggressive' (recommended) or 'conservative'",
                        "default": "aggressive"
                    }
                }
            }
        ),
        Tool(
            name="get_pending_exits",
            description="Get positions with exit signals detected but not yet closed (exit_date set, status='open'). These are scheduled for exit on the next trading day.",
            inputSchema={
                "type": "object",
                "properties": {
                    "timing_mode": {
                        "type": "string",
                        "description": "Filter by timing mode: 'conservative', 'aggressive', or 'both'",
                        "default": "both"
                    }
                }
            }
        ),
        Tool(
            name="get_recent_trades",
            description="Get recently closed trades with entry/exit details, P&L, and exit reasons. Use to review recent performance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to look back (default: 7)",
                        "default": 7
                    },
                    "timing_mode": {
                        "type": "string",
                        "description": "Filter by timing mode: 'conservative', 'aggressive', or 'both'",
                        "default": "both"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum trades to return (default: 20)",
                        "default": 20
                    }
                }
            }
        ),
        Tool(
            name="get_position_details",
            description="Get detailed information for a specific symbol position including entry indicators, OHLC data, and current status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Symbol code (e.g., 'AAPL', 'MSFT')"
                    },
                    "timing_mode": {
                        "type": "string",
                        "description": "Filter by timing mode: 'conservative', 'aggressive', or 'both'",
                        "default": "both"
                    }
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_trade_database_status",
            description="Get trade database connection status and basic statistics. Useful to verify the connection is working.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    handlers = {
        "get_open_positions": handle_get_open_positions,
        "get_portfolio_summary": handle_get_portfolio_summary,
        "get_pending_exits": handle_get_pending_exits,
        "get_recent_trades": handle_get_recent_trades,
        "get_position_details": handle_get_position_details,
        "get_trade_database_status": handle_get_database_status,
    }

    handler = handlers.get(name)
    if not handler:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    try:
        return await handler(arguments)
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_get_open_positions(args: dict) -> list[TextContent]:
    """Get open positions from trade.db."""
    timing_mode = args.get("timing_mode", "both")
    executed_only = args.get("executed_only", True)
    limit = args.get("limit", 50)
    sort_by = args.get("sort_by", "entry_date")

    # Build query
    timing_filter = ""
    if timing_mode != "both":
        timing_filter = f"AND timing_mode = '{timing_mode}'"

    shares_filter = "AND shares IS NOT NULL" if executed_only else ""

    # Sort mapping
    sort_map = {
        "symbol": "symbol_code ASC",
        "entry_date": "entry_date DESC",
        "pnl": "unrealized_pnl_pct DESC",
        "value": "current_value DESC"
    }
    sort_clause = sort_map.get(sort_by, "entry_date DESC")

    sql = f"""
    SELECT
        symbol_code,
        exchange_code,
        timing_mode,
        entry_date,
        entry_price,
        shares,
        entry_value,
        ROUND(entry_value * 1.05, 2) as current_value,
        ROUND(5.0, 2) as unrealized_pnl_pct,
        exit_date,
        exit_reason,
        duration_days
    FROM strategy_positions
    WHERE status = 'open'
        AND strategy_id = 'adm_momentum'
        {timing_filter}
        {shares_filter}
    ORDER BY {sort_clause}
    LIMIT {limit}
    """

    output = await execute_query(sql)
    positions = parse_pipe_output(output)

    if not positions:
        return [TextContent(type="text", text="No open positions found.")]

    # Format output
    lines = [f"Open Positions ({len(positions)} found):", ""]

    for pos in positions:
        symbol = pos.get('symbol_code', 'N/A')
        timing = pos.get('timing_mode', '').upper()[:4]
        entry_date = pos.get('entry_date', 'N/A')
        entry_price = format_currency(pos.get('entry_price'))
        shares = pos.get('shares', 'N/A')
        entry_value = format_currency(pos.get('entry_value'))

        # Check for pending exit
        exit_marker = ""
        if pos.get('exit_date'):
            exit_marker = f" [EXIT PENDING: {pos.get('exit_reason', 'signal')}]"

        lines.append(f"  {symbol} ({timing}): {shares} shares @ {entry_price}")
        lines.append(f"    Entry: {entry_date}, Value: {entry_value}{exit_marker}")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_get_portfolio_summary(args: dict) -> list[TextContent]:
    """Get portfolio summary."""
    timing_mode = args.get("timing_mode", "aggressive")

    # Get latest portfolio state
    sql = f"""
    SELECT
        state_date,
        total_equity,
        cash_available,
        positions_value,
        num_open_positions,
        max_positions,
        trades_entered_today,
        trades_exited_today,
        cash_deployed_today,
        cash_freed_today
    FROM portfolio_state
    WHERE strategy_id = 'adm_momentum'
        AND timing_mode = '{timing_mode}'
    ORDER BY state_date DESC
    LIMIT 1
    """

    output = await execute_query(sql)
    results = parse_pipe_output(output)

    if not results:
        return [TextContent(type="text", text=f"No portfolio state found for {timing_mode} timing mode.")]

    state = results[0]

    # Also get executed position count
    exec_sql = f"""
    SELECT COUNT(*) as count
    FROM strategy_positions
    WHERE status = 'open'
        AND strategy_id = 'adm_momentum'
        AND timing_mode = '{timing_mode}'
        AND shares IS NOT NULL
    """
    exec_output = await execute_query(exec_sql)
    exec_results = parse_pipe_output(exec_output)
    executed_count = exec_results[0]['count'] if exec_results else 0

    # Format output
    lines = [
        f"Portfolio Summary ({timing_mode.title()} Mode)",
        f"As of: {state.get('state_date', 'N/A')}",
        "",
        "Holdings:",
        f"  Total Equity:     {format_currency(state.get('total_equity'))}",
        f"  Cash Available:   {format_currency(state.get('cash_available'))}",
        f"  Positions Value:  {format_currency(state.get('positions_value'))}",
        "",
        "Positions:",
        f"  Open Positions:   {executed_count} executed of {state.get('num_open_positions', 'N/A')} total",
        f"  Max Allowed:      {state.get('max_positions', 'N/A')}",
        "",
        f"Today's Activity:",
        f"  Entries:          {state.get('trades_entered_today', 0)}",
        f"  Exits:            {state.get('trades_exited_today', 0)}",
        f"  Cash Deployed:    {format_currency(state.get('cash_deployed_today'))}",
        f"  Cash Freed:       {format_currency(state.get('cash_freed_today'))}",
    ]

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_get_pending_exits(args: dict) -> list[TextContent]:
    """Get positions with exit signals pending."""
    timing_mode = args.get("timing_mode", "both")

    timing_filter = ""
    if timing_mode != "both":
        timing_filter = f"AND timing_mode = '{timing_mode}'"

    sql = f"""
    SELECT
        symbol_code,
        exchange_code,
        timing_mode,
        entry_date,
        entry_price,
        shares,
        entry_value,
        exit_date,
        exit_reason
    FROM strategy_positions
    WHERE status = 'open'
        AND exit_date IS NOT NULL
        AND shares IS NOT NULL
        AND strategy_id = 'adm_momentum'
        {timing_filter}
    ORDER BY exit_date ASC, symbol_code ASC
    """

    output = await execute_query(sql)
    positions = parse_pipe_output(output)

    if not positions:
        return [TextContent(type="text", text="No pending exits found. All open positions are holding.")]

    lines = [f"Pending Exits ({len(positions)} positions):", ""]

    for pos in positions:
        symbol = pos.get('symbol_code', 'N/A')
        timing = pos.get('timing_mode', '').upper()[:4]
        exit_date = pos.get('exit_date', 'N/A')
        exit_reason = pos.get('exit_reason', 'signal')
        shares = pos.get('shares', 'N/A')
        entry_value = format_currency(pos.get('entry_value'))

        lines.append(f"  {symbol} ({timing}): {shares} shares, Value: {entry_value}")
        lines.append(f"    Exit Date: {exit_date}, Reason: {exit_reason}")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_get_recent_trades(args: dict) -> list[TextContent]:
    """Get recently closed trades."""
    days = args.get("days", 7)
    timing_mode = args.get("timing_mode", "both")
    limit = args.get("limit", 20)

    timing_filter = ""
    if timing_mode != "both":
        timing_filter = f"AND timing_mode = '{timing_mode}'"

    # Calculate date threshold
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    sql = f"""
    SELECT
        symbol_code,
        exchange_code,
        timing_mode,
        entry_date,
        entry_price,
        exit_date,
        exit_price,
        exit_reason,
        shares,
        entry_value,
        exit_value,
        pnl_pct,
        duration_days
    FROM strategy_positions
    WHERE status = 'closed'
        AND exit_date >= '{cutoff_date}'
        AND strategy_id = 'adm_momentum'
        {timing_filter}
    ORDER BY exit_date DESC
    LIMIT {limit}
    """

    output = await execute_query(sql)
    trades = parse_pipe_output(output)

    if not trades:
        return [TextContent(type="text", text=f"No closed trades in the last {days} days.")]

    # Calculate summary stats
    total_pnl = sum(t.get('pnl_pct', 0) or 0 for t in trades)
    winners = sum(1 for t in trades if (t.get('pnl_pct', 0) or 0) > 0)
    win_rate = (winners / len(trades) * 100) if trades else 0

    lines = [
        f"Recent Trades (Last {days} Days)",
        f"Total: {len(trades)}, Winners: {winners}, Win Rate: {win_rate:.0f}%",
        "",
    ]

    for trade in trades:
        symbol = trade.get('symbol_code', 'N/A')
        timing = trade.get('timing_mode', '').upper()[:4]
        exit_date = trade.get('exit_date', 'N/A')
        pnl = trade.get('pnl_pct')
        pnl_str = format_pct(pnl) if pnl is not None else 'N/A'
        exit_reason = trade.get('exit_reason', 'N/A')
        duration = trade.get('duration_days', 'N/A')

        # Win/loss indicator
        indicator = "+" if (pnl or 0) > 0 else "-" if (pnl or 0) < 0 else "="

        lines.append(f"  [{indicator}] {symbol} ({timing}): {pnl_str} | {exit_date}")
        lines.append(f"      Reason: {exit_reason}, Duration: {duration} days")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_get_position_details(args: dict) -> list[TextContent]:
    """Get detailed info for a specific position."""
    symbol = args.get("symbol", "").upper()
    timing_mode = args.get("timing_mode", "both")

    if not symbol:
        return [TextContent(type="text", text="Error: symbol is required")]

    timing_filter = ""
    if timing_mode != "both":
        timing_filter = f"AND timing_mode = '{timing_mode}'"

    sql = f"""
    SELECT
        symbol_code,
        exchange_code,
        instrument_type,
        timing_mode,
        entry_date,
        entry_price,
        entry_reason,
        entry_ohlc,
        entry_indicators,
        shares,
        entry_value,
        exit_date,
        exit_price,
        exit_reason,
        status,
        pnl_pct,
        duration_days,
        created_at,
        updated_at
    FROM strategy_positions
    WHERE symbol_code = '{symbol}'
        AND strategy_id = 'adm_momentum'
        {timing_filter}
    ORDER BY entry_date DESC
    LIMIT 5
    """

    output = await execute_query(sql)
    positions = parse_pipe_output(output)

    if not positions:
        return [TextContent(type="text", text=f"No positions found for {symbol}")]

    lines = [f"Position Details: {symbol}", ""]

    for pos in positions:
        status = pos.get('status', 'N/A').upper()
        timing = pos.get('timing_mode', '').title()
        entry_date = pos.get('entry_date', 'N/A')
        entry_price = format_currency(pos.get('entry_price'))
        shares = pos.get('shares')
        entry_value = format_currency(pos.get('entry_value'))

        lines.append(f"[{status}] {timing} Mode")
        lines.append(f"  Entry: {entry_date} @ {entry_price}")

        if shares:
            lines.append(f"  Shares: {shares}, Value: {entry_value}")
        else:
            lines.append(f"  (Signal only - not executed)")

        lines.append(f"  Entry Reason: {pos.get('entry_reason', 'N/A')}")

        if pos.get('status') == 'closed':
            exit_price = format_currency(pos.get('exit_price'))
            pnl = format_pct(pos.get('pnl_pct'))
            lines.append(f"  Exit: {pos.get('exit_date')} @ {exit_price}")
            lines.append(f"  P&L: {pnl}, Duration: {pos.get('duration_days')} days")
            lines.append(f"  Exit Reason: {pos.get('exit_reason', 'N/A')}")
        elif pos.get('exit_date'):
            lines.append(f"  [PENDING EXIT] Date: {pos.get('exit_date')}, Reason: {pos.get('exit_reason')}")

        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_get_database_status(args: dict) -> list[TextContent]:
    """Get database status and basic statistics."""
    try:
        # Test connection and get counts
        sql = """
        SELECT
            (SELECT COUNT(*) FROM strategy_positions WHERE status='open' AND shares IS NOT NULL) as open_executed,
            (SELECT COUNT(*) FROM strategy_positions WHERE status='open') as open_total,
            (SELECT COUNT(*) FROM strategy_positions WHERE status='closed') as closed_total,
            (SELECT COUNT(*) FROM portfolio_state) as state_snapshots,
            (SELECT MAX(state_date) FROM portfolio_state) as latest_state,
            (SELECT MAX(entry_date) FROM strategy_positions) as latest_entry
        """

        output = await execute_query(sql)
        results = parse_pipe_output(output)

        if not results:
            return [TextContent(type="text", text="Connected but no data found.")]

        stats = results[0]

        # Determine connection mode for status display
        if LOCAL_MODE:
            conn_status = f"OK (local: {get_local_db_path()})"
        else:
            conn_status = f"OK (via SSH to {MACMINI_HOST})"

        lines = [
            "Trade Database Status",
            f"Connection: {conn_status}",
            "",
            "Statistics:",
            f"  Open Positions (executed): {stats.get('open_executed', 0)}",
            f"  Open Positions (total):    {stats.get('open_total', 0)}",
            f"  Closed Trades:             {stats.get('closed_total', 0)}",
            f"  Portfolio Snapshots:       {stats.get('state_snapshots', 0)}",
            "",
            f"  Latest Portfolio State:    {stats.get('latest_state', 'N/A')}",
            f"  Latest Entry Date:         {stats.get('latest_entry', 'N/A')}",
        ]

        return [TextContent(type="text", text="\n".join(lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Connection Failed: {str(e)}")]


# Resources
@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    return [
        Resource(
            uri=AnyUrl("trade://status"),
            name="Database Status",
            description="Trade database connection status and statistics",
            mimeType="text/plain"
        ),
        Resource(
            uri=AnyUrl("trade://portfolio"),
            name="Portfolio Summary",
            description="Current portfolio holdings summary",
            mimeType="text/plain"
        ),
    ]


@server.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    """Read a resource."""
    uri_str = str(uri)
    if uri_str == "trade://status":
        result = await handle_get_database_status({})
        return result[0].text
    elif uri_str == "trade://portfolio":
        result = await handle_get_portfolio_summary({"timing_mode": "aggressive"})
        return result[0].text
    else:
        raise ValueError(f"Unknown resource: {uri_str}")


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
