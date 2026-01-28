"""
Скрипт запуска для Koyeb - с принудительным flush
"""
import os
import sys

# ВАЖНО: Отключаем буферизацию вывода
os.environ['PYTHONUNBUFFERED'] = '1'

# Функция для печати с flush
def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()

log("=" * 50)
log("[START] RUN.PY STARTED!")
log("[INFO] Python: {}".format(sys.version.split()[0]))
log("[INFO] CWD: {}".format(os.getcwd()))
log("[INFO] FILES: {}".format(os.listdir('.')))
log("[INFO] PORT: {}".format(os.environ.get('PORT', 'NOT SET')))
log("[INFO] OPENAI_API_KEY: {}".format('SET' if os.environ.get('OPENAI_API_KEY') else 'NOT SET'))
log("[INFO] CURRENTS_API_KEY: {}".format('SET' if os.environ.get('CURRENTS_API_KEY') else 'NOT SET'))
log("=" * 50)

log("[STEP] Importing uvicorn...")
try:
    import uvicorn
    log("[OK] uvicorn imported")
except Exception as e:
    log("[ERROR] uvicorn import failed: {}".format(e))
    sys.exit(1)

log("[STEP] Importing app...")
try:
    from app import app
    log("[OK] app imported")
except Exception as e:
    log("[ERROR] app import failed: {}".format(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)

port = int(os.environ.get("PORT", 8000))
log("[RUN] Starting uvicorn on 0.0.0.0:{}".format(port))
log("=" * 50)

try:
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
except Exception as e:
    log("[ERROR] uvicorn.run failed: {}".format(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
