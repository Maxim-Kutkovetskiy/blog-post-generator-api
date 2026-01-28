"""
Скрипт запуска для Koyeb.
"""
import os
import sys

# Добавляем логи сразу
print("=" * 50)
print("🚀 run.py запущен!")
print(f"📍 Python: {sys.version}")
print(f"📍 Рабочая директория: {os.getcwd()}")
print(f"📍 Файлы: {os.listdir('.')}")
print(f"📍 PORT из окружения: {os.environ.get('PORT', 'НЕ УСТАНОВЛЕН')}")
print("=" * 50)

# Проверяем что app.py существует
if not os.path.exists('app.py'):
    print("❌ ОШИБКА: app.py не найден!")
    sys.exit(1)

# Импортируем uvicorn
try:
    import uvicorn
    print("✅ uvicorn импортирован")
except ImportError as e:
    print(f"❌ ОШИБКА импорта uvicorn: {e}")
    sys.exit(1)

# Проверяем что app импортируется
try:
    from app import app
    print("✅ app.app импортирован")
except Exception as e:
    print(f"❌ ОШИБКА импорта app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Запускаем сервер
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🌐 Запуск uvicorn на порту {port}...")
    
    try:
        uvicorn.run(
            app,  # Передаём объект напрямую, не строку!
            host="0.0.0.0",
            port=port,
            log_level="info",
        )
    except Exception as e:
        print(f"❌ ОШИБКА запуска uvicorn: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
