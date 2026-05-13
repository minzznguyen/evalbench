#!/bin/bash
exec uv run --no-sync gunicorn -w 12 -k sync main:me --bind :${PORT:-3000} --forwarded-allow-ips="*" --timeout 120
