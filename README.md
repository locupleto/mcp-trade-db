# mcp-trade-db

MCP (Model Context Protocol) server for accessing production trade database via SSH proxy pattern.

## Overview

This server provides Claude Code with read-only access to the live `trade.db` on Mac mini production server. It uses SSH to execute SQLite queries remotely, enabling portfolio monitoring and trade analysis without copying sensitive production data.

## Architecture

```
Mac Studio (Claude Code) → SSH → Mac mini (trade.db)
```

The MCP server runs locally but proxies all database queries through SSH to the production server. This ensures:
- Live production data access
- No local data copies
- Secure SSH-authenticated connections
- Read-only queries only

## Tools

| Tool | Description |
|------|-------------|
| `get_open_positions` | Current portfolio holdings with entry dates, prices, shares, and P&L |
| `get_portfolio_summary` | Quick overview: total equity, cash, positions value, daily activity |
| `get_pending_exits` | Positions with exit signals detected, scheduled for next trading day |
| `get_recent_trades` | Recently closed trades with entry/exit details and P&L |
| `get_position_details` | Detailed info for a specific symbol including indicators |
| `get_trade_database_status` | Connection status and basic statistics |

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MACMINI_HOST` | `100.102.226.1` | Mac mini Tailscale IP |
| `SSH_KEY_PATH` | `~/.ssh/macmini` | SSH private key path |
| `TRADE_DB_PATH` | `~/Public/trading-lab/data/trade.db` | Remote database path |

## Installation

```bash
# Clone repository
git clone https://github.com/locupleto/mcp-trade-db.git
cd mcp-trade-db

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## MCP Configuration

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "trade-db": {
      "type": "stdio",
      "command": "/path/to/mcp-trade-db/venv/bin/python3",
      "args": ["/path/to/mcp-trade-db/trade_db_mcp_server.py"],
      "env": {
        "MACMINI_HOST": "100.102.226.1",
        "SSH_KEY_PATH": "/Users/urban/.ssh/macmini",
        "TRADE_DB_PATH": "~/Public/trading-lab/data/trade.db"
      }
    }
  }
}
```

## Usage Examples

```
# Get portfolio summary
mcp__trade-db__get_portfolio_summary(timing_mode="aggressive")

# List open positions
mcp__trade-db__get_open_positions(timing_mode="both", executed_only=true)

# Check recent trades
mcp__trade-db__get_recent_trades(days=7, limit=20)

# Get details for specific symbol
mcp__trade-db__get_position_details(symbol="AAPL")
```

## Timing Modes

The trade database tracks two execution timing strategies:

- **Conservative** (`T+1 Open`): Enter on next day's open after signal
- **Aggressive** (`Same-day Close`): Enter on signal day's close

Most tools accept `timing_mode` parameter: `"conservative"`, `"aggressive"`, or `"both"`.

## Requirements

- Python 3.10+
- SSH access to Mac mini with key authentication
- `mcp` Python package

## License

MIT
