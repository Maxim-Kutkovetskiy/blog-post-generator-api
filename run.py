"""
Скрипт запуска для Koyeb - чистый ASCII вывод

"""
import os
import sys

# Принудительно устанавливаем UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # Python < 3.7

print("=" * 50)
print("[START] RUN.PY STARTED!")
print("[INFO] Python: {}".format(sys.version.split()[0]))
print("[INFO] CWD: {}".format(os.getcwd()))
print("[INFO] PORT: {}".format(os.environ.get('PORT', 'NOT SET')))
print("[INFO] OPENAI_API_KEY: {}".format('SET' if os.environ.get('OPENAI_API_KEY') else 'NOT SET'))
print("[INFO] CURRENTS_API_KEY: {}".format('SET' if os.environ.get('CURRENTS_API_KEY') else 'NOT SET'))
print("=" * 50)

print("[STEP] Importing uvicorn...")
import uvicorn
print("[OK] uvicorn imported")

print("[STEP] Importing app...")
from app import app
print("[OK] app imported")

port = int(os.environ.get("PORT", 8000))
print("[RUN] Starting server on 0.0.0.0:{}".format(port))

uvicorn.run(
    app,
    host="0.0.0.0",
    port=port,
    log_level="info",
)
