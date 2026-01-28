"""
FastAPI приложение для автоматической генерации контента.
Версия: 1.2.1 (исправлен для Koyeb)
"""

import os
import sys
import logging
from typing import Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

# ============================================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ (В ПЕРВУЮ ОЧЕРЕДЬ!)
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,  # Явно указываем stdout для Koyeb
    force=True  # Перезаписываем конфигурацию
)
logger = logging.getLogger(__name__)

# ============================================================================
# ЗАГРУЗКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ
# ============================================================================

# Проверяем наличие dotenv (только для локальной разработки)
try:
    from dotenv import load_dotenv
    # Загружаем только если файл существует
    if os.path.exists('.env'):
        load_dotenv()
        logger.info("📁 Загружены переменные из .env (локальная разработка)")
    else:
        logger.info("☁️ Используем переменные окружения (production)")
except ImportError:
    logger.info("☁️ python-dotenv не установлен, используем системные переменные")

# ============================================================================
# ИМПОРТЫ (ПОСЛЕ НАСТРОЙКИ ЛОГОВ!)
# ============================================================================

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import requests
from openai import OpenAI

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

class Settings:
    """Класс для хранения настроек приложения."""

    def __init__(self):
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.currents_api_key: str = os.getenv("CURRENTS_API_KEY", "")
        self.port: int = int(os.getenv("PORT", 8000))
        self.currents_api_url: str = "https://api.currentsapi.services/v1/latest-news"
        self.openai_model: str = "gpt-4o-mini"
        self.max_news_count: int = 5
        self.news_language: str = "en"
        self._validated: bool = False

    def validate(self) -> None:
        """Проверяет наличие обязательных переменных."""
        if self._validated:
            return
            
        missing_keys = []

        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")

        if not self.currents_api_key:
            missing_keys.append("CURRENTS_API_KEY")

        if missing_keys:
            error_message = f"Отсутствуют переменные: {', '.join(missing_keys)}"
            logger.error(f"❌ {error_message}")
            raise ValueError(error_message)

        self._validated = True
        logger.info("✅ Все переменные окружения загружены")

    def is_configured(self) -> bool:
        """Проверяет, настроено ли приложение."""
        return bool(self.openai_api_key and self.currents_api_key)


# Создаём экземпляр настроек
settings = Settings()

# Глобальная переменная для OpenAI клиента
openai_client: Optional[OpenAI] = None

# ============================================================================
# PYDANTIC МОДЕЛИ
# ============================================================================

class TopicRequest(BaseModel):
    """Модель запроса для генерации контента."""
    topic: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Тема для генерации контента",
        examples=["искусственный интеллект"]
    )
    language: Optional[str] = Field(
        default="en",
        description="Язык новостей (en, ru, de, fr и т.д.)"
    )
    max_news: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Максимальное количество новостей"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "topic": "машинное обучение",
                "language": "en",
                "max_news": 5
            }
        }
    }


class GeneratedContent(BaseModel):
    """Модель ответа с сгенерированным контентом."""
    title: str = Field(..., description="Заголовок статьи")
    meta_description: str = Field(..., description="Мета-описание для SEO")
    post_content: str = Field(..., description="Основной текст статьи")
    news_sources: List[str] = Field(default_factory=list, description="Источники новостей")
    generated_at: str = Field(..., description="Время генерации")
    topic: str = Field(..., description="Исходная тема")


class HealthResponse(BaseModel):
    """Модель ответа для проверки состояния."""
    status: str = Field(..., description="Статус сервиса")
    timestamp: str = Field(..., description="Текущее время")
    version: str = Field(default="1.2.0", description="Версия API")
    configured: bool = Field(default=False, description="Настроены ли API ключи")


class ErrorResponse(BaseModel):
    """Модель для ответов с ошибками."""
    error: str = Field(..., description="Тип ошибки")
    detail: str = Field(..., description="Описание ошибки")
    timestamp: str = Field(..., description="Время ошибки")


# ============================================================================
# LIFESPAN (СОВРЕМЕННЫЙ ПОДХОД ВМЕСТО on_event)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения.
    Заменяет устаревшие @app.on_event("startup") и @app.on_event("shutdown")
    """
    global openai_client
    
    # === STARTUP ===
    logger.info("🚀 Запуск Content Generation API...")
    logger.info(f"📌 Порт: {settings.port}")
    logger.info(f"🤖 Модель OpenAI: {settings.openai_model}")
    
    try:
        settings.validate()
        openai_client = OpenAI(api_key=settings.openai_api_key)
        logger.info("✅ OpenAI клиент инициализирован")
    except ValueError as e:
        logger.error(f"❌ Ошибка конфигурации: {e}")
        logger.warning("⚠️ Генерация контента недоступна!")
    
    logger.info("✅ Приложение запущено!")
    
    yield  # Приложение работает
    
    # === SHUTDOWN ===
    logger.info("🛑 Остановка Content Generation API...")
    logger.info("✅ Приложение остановлено!")


# ============================================================================
# ИНИЦИАЛИЗАЦИЯ FASTAPI
# ============================================================================

app = FastAPI(
    title="Content Generation API",
    description="""
    ## API для автоматической генерации контента
    
    ### Возможности:
    * 📰 Получение актуальных новостей по теме
    * ✍️ Генерация SEO-заголовков
    * 📝 Создание мета-описаний
    * 📄 Генерация статей с GPT-4
    
    ### Версия: 1.2.0
    """,
    version="1.2.0",
    lifespan=lifespan  # Используем lifespan вместо on_event
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def get_current_timestamp() -> str:
    """Возвращает текущее время в ISO формате."""
    return datetime.now().isoformat()


def get_openai_client() -> OpenAI:
    """Возвращает OpenAI клиент с проверкой."""
    global openai_client
    
    if openai_client is None:
        logger.error("❌ OpenAI клиент не инициализирован")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис не настроен. Проверьте API ключи."
        )
    
    return openai_client


def get_recent_news(
    topic: str,
    language: str = "en",
    max_count: int = 5
) -> tuple[str, List[str]]:
    """Получает новости по теме из Currents API."""
    logger.info(f"🔍 Поиск новостей: '{topic}' (язык: {language})")

    params = {
        "language": language,
        "keywords": topic,
        "apiKey": settings.currents_api_key
    }

    try:
        response = requests.get(
            settings.currents_api_url,
            params=params,
            timeout=30
        )

        if response.status_code != 200:
            logger.error(f"❌ Currents API: {response.status_code}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Ошибка новостного API: {response.text}"
            )

        data = response.json()
        news_articles = data.get("news", [])

        if not news_articles:
            logger.warning(f"⚠️ Новости по '{topic}' не найдены")
            return "Свежих новостей не найдено.", []

        news_articles = news_articles[:max_count]
        titles = [a.get("title", "Без заголовка") for a in news_articles]

        news_context = "\n".join([
            f"- {a.get('title', '')}: {a.get('description', '')[:200]}"
            for a in news_articles
        ])

        logger.info(f"✅ Найдено {len(titles)} новостей")
        return news_context, titles

    except requests.exceptions.Timeout:
        logger.error("❌ Таймаут Currents API")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Таймаут новостного сервиса"
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Ошибка сети: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ошибка соединения: {e}"
        )


def generate_title(topic: str, news_context: str) -> str:
    """Генерирует заголовок статьи."""
    logger.info(f"📝 Генерация заголовка: '{topic}'")

    client = get_openai_client()

    prompt = f"""Придумайте заголовок для статьи на тему '{topic}'.

Новости:
{news_context}

Требования:
- 50-70 символов
- Интересный, без кликбейта
- На русском языке

Верните только заголовок."""

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7
    )

    title = response.choices[0].message.content.strip().strip('"\'«»')
    logger.info(f"✅ Заголовок: '{title}'")
    return title


def generate_meta_description(title: str, topic: str) -> str:
    """Генерирует мета-описание."""
    logger.info(f"📋 Генерация мета-описания")

    client = get_openai_client()

    prompt = f"""Мета-описание для статьи.

Заголовок: {title}
Тема: {topic}

Требования:
- 150-160 символов
- Информативно, с ключевыми словами
- На русском

Верните только описание."""

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.5
    )

    meta = response.choices[0].message.content.strip().strip('"\'«»')
    logger.info(f"✅ Мета-описание ({len(meta)} символов)")
    return meta


def generate_article_content(topic: str, title: str, news_context: str) -> str:
    """Генерирует текст статьи."""
    logger.info(f"📄 Генерация статьи: '{topic}'")

    client = get_openai_client()

    prompt = f"""Напишите статью на тему '{topic}'.

Заголовок: {title}

Новости для контекста:
{news_context}

Требования:
1. 1500-2000 символов
2. Введение, 3-4 подраздела (##), заключение
3. Профессиональный стиль
4. На русском языке"""

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.7,
        presence_penalty=0.6,
        frequency_penalty=0.6
    )

    content = response.choices[0].message.content.strip()
    logger.info(f"✅ Статья ({len(content)} символов)")
    return content


def generate_content(topic: str, language: str = "en", max_news: int = 5) -> GeneratedContent:
    """Основная функция генерации контента."""
    logger.info(f"🚀 Генерация контента: '{topic}'")

    try:
        news_context, news_titles = get_recent_news(topic, language, max_news)
        title = generate_title(topic, news_context)
        meta_description = generate_meta_description(title, topic)
        post_content = generate_article_content(topic, title, news_context)

        return GeneratedContent(
            title=title,
            meta_description=meta_description,
            post_content=post_content,
            news_sources=news_titles,
            generated_at=get_current_timestamp(),
            topic=topic
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка генерации: {e}"
        )


# ============================================================================
# API ЭНДПОИНТЫ
# ============================================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root() -> HealthResponse:
    """Корневой эндпоинт."""
    return HealthResponse(
        status="running",
        timestamp=get_current_timestamp(),
        version="1.2.0",
        configured=settings.is_configured()
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check для Koyeb."""
    return HealthResponse(
        status="healthy",
        timestamp=get_current_timestamp(),
        version="1.2.0",
        configured=settings.is_configured()
    )


@app.get("/heartbeat", tags=["Health"])
async def heartbeat() -> dict:
    """Простой heartbeat."""
    return {"status": "OK", "timestamp": get_current_timestamp()}


@app.post("/generate-post", response_model=GeneratedContent, tags=["Content"])
async def generate_post_api(request: TopicRequest) -> GeneratedContent:
    """Генерирует контент по теме."""
    logger.info(f"📨 Запрос: '{request.topic}'")

    if not request.topic.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Тема не может быть пустой"
        )

    return generate_content(
        topic=request.topic.strip(),
        language=request.language,
        max_news=request.max_news
    )


@app.get("/news/{topic}", tags=["News"])
async def get_news(topic: str, language: str = "en", limit: int = 5) -> dict:
    """Получает новости по теме."""
    news_context, titles = get_recent_news(topic, language, min(limit, 10))
    return {
        "topic": topic,
        "language": language,
        "count": len(titles),
        "news": titles,
        "timestamp": get_current_timestamp()
    }


# ============================================================================
# ОБРАБОТЧИКИ ОШИБОК
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Обработчик HTTP ошибок."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "detail": exc.detail,
            "status_code": exc.status_code,
            "timestamp": get_current_timestamp()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Обработчик всех ошибок."""
    logger.error(f"❌ Ошибка: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "Внутренняя ошибка сервера",
            "timestamp": get_current_timestamp()
        }
    )
