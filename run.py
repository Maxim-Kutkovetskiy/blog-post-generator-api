"""
Скрипт запуска для Koyeb - БЕЗ условия if __name__
"""
import os
import sys

# Отключаем буферизацию вывода
sys.stdout.reconfigure(line_buffering=True)

print("=" * 50, flush=True)
print("🚀 RUN.PY ЗАПУЩЕН!", flush=True)
print(f"📍 Python: {sys.version}", flush=True)
print(f"📍 CWD: {os.getcwd()}", flush=True)
print(f"📍 PORT: {os.environ.get('PORT', 'NOT SET')}", flush=True)
print(f"📍 OPENAI_API_KEY: {'SET' if os.environ.get('OPENAI_API_KEY') else 'NOT SET'}", flush=True)
print(f"📍 CURRENTS_API_KEY: {'SET' if os.environ.get('CURRENTS_API_KEY') else 'NOT SET'}", flush=True)
print("=" * 50, flush=True)

# Импорт uvicorn
print("📦 Импортируем uvicorn...", flush=True)
import uvicorn
print("✅ uvicorn импортирован", flush=True)

# Импорт приложения
print("📦 Импортируем app...", flush=True)
from app import app
print("✅ app импортирован", flush=True)

# Получаем порт
port = int(os.environ.get("PORT", 8000))
print(f"🌐 Запускаем сервер на 0.0.0.0:{port}", flush=True)

# ЗАПУСКАЕМ СЕРВЕР БЕЗ УСЛОВИЯ IF!
uvicorn.run(
    app,
    host="0.0.0.0",
    port=port,
    log_level="info",
)
