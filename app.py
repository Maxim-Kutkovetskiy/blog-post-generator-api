"""
FastAPI приложение для автоматической генерации контента на основе актуальных новостей.

Приложение использует:
- Currents API для получения свежих новостей по заданной теме
- OpenAI API (GPT-4o-mini) для генерации заголовков, мета-описаний и статей

Версия: 1.1.0 (адаптирован для Koyeb)
"""

import os
import logging
from typing import Optional, List
from datetime import datetime

# ============================================================================
# ЗАГРУЗКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ
# ============================================================================

# Загружаем .env файл (только для локальной разработки)
# На Koyeb переменные окружения настраиваются в панели управления
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Загружены переменные из .env файла (локальная разработка)")
except ImportError:
    print("ℹ️ python-dotenv не установлен, используем системные переменные окружения")

# ============================================================================
# ИМПОРТЫ ФРЕЙМВОРКОВ И БИБЛИОТЕК
# ============================================================================

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import requests
from openai import OpenAI

# ============================================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ============================================================================
# КОНФИГУРАЦИЯ ПРИЛОЖЕНИЯ
# ============================================================================

class Settings:
    """
    Класс для хранения настроек приложения.
    Все настройки загружаются из переменных окружения.
    
    ИЗМЕНЕНИЕ: Убрана автоматическая валидация при инициализации.
    Теперь валидация происходит в startup_event, что позволяет
    корректно обрабатывать ошибки на этапе запуска сервера.
    """

    def __init__(self):
        # API ключи (могут быть пустыми при инициализации)
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.currents_api_key: str = os.getenv("CURRENTS_API_KEY", "")
        
        # Порт: Koyeb передаёт через переменную PORT
        self.port: int = int(os.getenv("PORT", 8000))
        
        # Настройки API
        self.currents_api_url: str = "https://api.currentsapi.services/v1/latest-news"
        self.openai_model: str = "gpt-4o-mini"
        self.max_news_count: int = 5
        self.news_language: str = "en"
        
        # Флаг валидации
        self._validated: bool = False

    def validate(self) -> None:
        """
        Проверяет, что все необходимые переменные окружения установлены.
        
        Raises:
            ValueError: Если отсутствуют обязательные переменные
        """
        if self._validated:
            return
            
        missing_keys = []

        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")

        if not self.currents_api_key:
            missing_keys.append("CURRENTS_API_KEY")

        if missing_keys:
            error_message = f"Отсутствуют обязательные переменные окружения: {', '.join(missing_keys)}"
            logger.error(f"❌ {error_message}")
            raise ValueError(error_message)

        self._validated = True
        logger.info("✅ Все переменные окружения успешно загружены")

    def is_configured(self) -> bool:
        """Проверяет, настроено ли приложение."""
        return bool(self.openai_api_key and self.currents_api_key)


# Создаём экземпляр настроек
# ИЗМЕНЕНИЕ: НЕ вызываем validate() здесь - это произойдёт в startup_event
settings = Settings()

# Глобальная переменная для OpenAI клиента
# ИЗМЕНЕНИЕ: Инициализируется в startup_event, а не при загрузке модуля
openai_client: Optional[OpenAI] = None

# ============================================================================
# PYDANTIC МОДЕЛИ (СХЕМЫ ДАННЫХ)
# ============================================================================

class TopicRequest(BaseModel):
    """
    Модель запроса для генерации контента.
    """
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
        description="Максимальное количество новостей для контекста"
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


class NewsArticle(BaseModel):
    """Модель для представления одной новости."""
    title: str = Field(..., description="Заголовок новости")
    description: Optional[str] = Field(None, description="Описание новости")
    url: Optional[str] = Field(None, description="Ссылка на источник")
    published: Optional[str] = Field(None, description="Дата публикации")


class GeneratedContent(BaseModel):
    """
    Модель ответа с сгенерированным контентом.
    """
    title: str = Field(..., description="Заголовок статьи")
    meta_description: str = Field(..., description="Мета-описание для SEO")
    post_content: str = Field(..., description="Основной текст статьи")
    news_sources: List[str] = Field(
        default_factory=list,
        description="Заголовки использованных новостей"
    )
    generated_at: str = Field(..., description="Время генерации")
    topic: str = Field(..., description="Исходная тема")


class HealthResponse(BaseModel):
    """Модель ответа для проверки состояния сервиса."""
    status: str = Field(..., description="Статус сервиса")
    timestamp: str = Field(..., description="Текущее время сервера")
    version: str = Field(default="1.1.0", description="Версия API")
    configured: bool = Field(default=False, description="Настроены ли API ключи")


class ErrorResponse(BaseModel):
    """Модель для ответов с ошибками."""
    error: str = Field(..., description="Тип ошибки")
    detail: str = Field(..., description="Подробное описание ошибки")
    timestamp: str = Field(..., description="Время возникновения ошибки")


# ============================================================================
# ИНИЦИАЛИЗАЦИЯ FASTAPI ПРИЛОЖЕНИЯ
# ============================================================================

app = FastAPI(
    title="Content Generation API",
    description="""
    ## API для автоматической генерации контента на основе актуальных новостей

    ### Возможности:
    * 📰 Получение актуальных новостей по заданной теме
    * ✍️ Генерация SEO-оптимизированных заголовков
    * 📝 Создание мета-описаний для поисковых систем
    * 📄 Генерация полноценных статей с использованием GPT-4

    ### Как использовать:
    1. Отправьте POST запрос на `/generate-post` с темой
    2. Получите готовый контент с заголовком, описанием и текстом статьи
    
    ### Версия: 1.1.0 (оптимизирован для Koyeb)
    """,
    version="1.1.0",
    contact={
        "name": "API Support",
        "email": "support@example.com"
    },
    license_info={
        "name": "MIT License"
    }
)

# CORS middleware
# ИЗМЕНЕНИЕ: Добавлен комментарий о безопасности
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ В продакшене укажите конкретные домены!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def get_current_timestamp() -> str:
    """Возвращает текущее время в формате ISO 8601."""
    return datetime.now().isoformat()


def get_openai_client() -> OpenAI:
    """
    Возвращает инициализированный OpenAI клиент.
    
    ИЗМЕНЕНИЕ: Добавлена функция для безопасного получения клиента
    с проверкой инициализации.
    
    Raises:
        HTTPException: Если клиент не инициализирован
    """
    global openai_client
    
    if openai_client is None:
        logger.error("❌ OpenAI клиент не инициализирован")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис не полностью инициализирован. Попробуйте позже."
        )
    
    return openai_client


def get_recent_news(
    topic: str,
    language: str = "en",
    max_count: int = 5
) -> tuple[str, List[str]]:
    """
    Получает последние новости по заданной теме из Currents API.

    Args:
        topic: Тема для поиска новостей
        language: Язык новостей (по умолчанию 'en')
        max_count: Максимальное количество новостей

    Returns:
        tuple: (строка с новостями для контекста, список заголовков)

    Raises:
        HTTPException: При ошибке получения данных от API
    """
    logger.info(f"🔍 Поиск новостей по теме: '{topic}' (язык: {language})")

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
            logger.error(f"❌ Ошибка Currents API: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Ошибка при получении новостей: {response.text}"
            )

        data = response.json()
        news_articles = data.get("news", [])

        if not news_articles:
            logger.warning(f"⚠️ Новости по теме '{topic}' не найдены")
            return "Свежих новостей по данной теме не найдено.", []

        news_articles = news_articles[:max_count]
        titles = [article.get("title", "Без заголовка") for article in news_articles]

        news_context = "\n".join([
            f"- {article.get('title', 'Без заголовка')}: {article.get('description', '')[:200]}"
            for article in news_articles
        ])

        logger.info(f"✅ Найдено {len(titles)} новостей по теме '{topic}'")
        return news_context, titles

    except requests.exceptions.Timeout:
        logger.error("❌ Таймаут при запросе к Currents API")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Превышено время ожидания ответа от новостного сервиса"
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Ошибка сети при запросе к Currents API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ошибка соединения с новостным сервисом: {str(e)}"
        )


def generate_title(topic: str, news_context: str) -> str:
    """Генерирует привлекательный заголовок для статьи."""
    logger.info(f"📝 Генерация заголовка для темы: '{topic}'")

    client = get_openai_client()

    prompt = f"""Придумайте привлекательный и точный заголовок для статьи на тему '{topic}'.

Учитывайте актуальные новости:
{news_context}

Требования к заголовку:
- Длина: 50-70 символов
- Должен быть интересным и привлекать внимание
- Должен ясно передавать суть темы
- Не используйте кликбейт
- Заголовок должен быть на русском языке

Верните только заголовок, без кавычек и дополнительного текста."""

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7,
        stop=["\n"]
    )

    title = response.choices[0].message.content.strip()
    title = title.strip('"\'«»')

    logger.info(f"✅ Заголовок сгенерирован: '{title}'")
    return title


def generate_meta_description(title: str, topic: str) -> str:
    """Генерирует SEO-оптимизированное мета-описание."""
    logger.info(f"📋 Генерация мета-описания для: '{title}'")

    client = get_openai_client()

    prompt = f"""Напишите мета-описание для статьи.

Заголовок статьи: {title}
Тема: {topic}

Требования:
- Длина: 150-160 символов (это критически важно для SEO)
- Должно быть информативным и привлекательным
- Содержать ключевые слова по теме
- Побуждать к прочтению статьи
- На русском языке

Верните только мета-описание, без кавычек."""

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.5
    )

    meta_description = response.choices[0].message.content.strip()
    meta_description = meta_description.strip('"\'«»')

    logger.info(f"✅ Мета-описание сгенерировано ({len(meta_description)} символов)")
    return meta_description


def generate_article_content(topic: str, title: str, news_context: str) -> str:
    """Генерирует полный текст статьи."""
    logger.info(f"📄 Генерация контента статьи для темы: '{topic}'")

    client = get_openai_client()

    prompt = f"""Напишите подробную, информативную статью на тему '{topic}'.

Заголовок статьи: {title}

Используйте следующие актуальные новости как контекст:
{news_context}

Требования к статье:
1. **Объём**: минимум 1500-2000 символов
2. **Структура**:
   - Введение (2-3 предложения, захватывающее внимание)
   - 3-4 подраздела с подзаголовками (используйте ## для подзаголовков)
   - Заключение с выводами
3. **Содержание**:
   - Анализ текущих трендов по теме
   - Конкретные примеры из актуальных новостей
   - Экспертная оценка ситуации
   - Прогнозы на будущее
4. **Стиль**:
   - Профессиональный, но доступный язык
   - Каждый абзац — 3-5 предложений
   - Используйте факты и статистику
   - Избегайте воды и общих фраз
5. **SEO**:
   - Естественное использование ключевых слов
   - Читабельная структура с подзаголовками

Статья должна быть на русском языке."""

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.7,
        presence_penalty=0.6,
        frequency_penalty=0.6
    )

    content = response.choices[0].message.content.strip()

    logger.info(f"✅ Статья сгенерирована ({len(content)} символов)")
    return content


def generate_content(topic: str, language: str = "en", max_news: int = 5) -> GeneratedContent:
    """
    Основная функция для генерации полного контента статьи.
    """
    logger.info(f"🚀 Начало генерации контента для темы: '{topic}'")

    try:
        # Шаг 1: Получаем актуальные новости
        news_context, news_titles = get_recent_news(topic, language, max_news)

        # Шаг 2: Генерируем заголовок
        title = generate_title(topic, news_context)

        # Шаг 3: Генерируем мета-описание
        meta_description = generate_meta_description(title, topic)

        # Шаг 4: Генерируем основной контент
        post_content = generate_article_content(topic, title, news_context)

        result = GeneratedContent(
            title=title,
            meta_description=meta_description,
            post_content=post_content,
            news_sources=news_titles,
            generated_at=get_current_timestamp(),
            topic=topic
        )

        logger.info(f"✅ Контент успешно сгенерирован для темы: '{topic}'")
        return result

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"❌ Ошибка при генерации контента: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка при генерации контента: {str(e)}"
        )


# ============================================================================
# API ЭНДПОИНТЫ
# ============================================================================

@app.get(
    "/",
    response_model=HealthResponse,
    summary="Корневой эндпоинт",
    description="Проверка работоспособности сервиса",
    tags=["Health Check"]
)
async def root() -> HealthResponse:
    """Корневой эндпоинт для проверки работоспособности."""
    return HealthResponse(
        status="running",
        timestamp=get_current_timestamp(),
        version="1.1.0",
        configured=settings.is_configured()
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка здоровья сервиса",
    description="Детальная проверка состояния сервиса",
    tags=["Health Check"]
)
async def health_check() -> HealthResponse:
    """Эндпоинт для health checks (Kubernetes/Docker/Koyeb)."""
    return HealthResponse(
        status="healthy",
        timestamp=get_current_timestamp(),
        version="1.1.0",
        configured=settings.is_configured()
    )


@app.get(
    "/heartbeat",
    summary="Heartbeat эндпоинт",
    description="Простая проверка доступности сервиса",
    tags=["Health Check"]
)
async def heartbeat() -> dict:
    """Простой heartbeat эндпоинт."""
    return {"status": "OK", "timestamp": get_current_timestamp()}


@app.post(
    "/generate-post",
    response_model=GeneratedContent,
    summary="Генерация контента",
    description="Генерирует статью на заданную тему с использованием актуальных новостей.",
    tags=["Content Generation"],
    responses={
        200: {"description": "Успешная генерация", "model": GeneratedContent},
        400: {"description": "Некорректный запрос", "model": ErrorResponse},
        500: {"description": "Внутренняя ошибка", "model": ErrorResponse},
        502: {"description": "Ошибка внешнего сервиса", "model": ErrorResponse},
        503: {"description": "Сервис недоступен", "model": ErrorResponse}
    }
)
async def generate_post_api(request: TopicRequest) -> GeneratedContent:
    """Генерирует контент для поста на основе заданной темы."""
    logger.info(f"📨 Получен запрос: тема='{request.topic}'")

    if not request.topic.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Тема не может быть пустой"
        )

    result = generate_content(
        topic=request.topic.strip(),
        language=request.language,
        max_news=request.max_news
    )

    return result


@app.get(
    "/news/{topic}",
    summary="Получить новости по теме",
    description="Возвращает список актуальных новостей",
    tags=["News"]
)
async def get_news(
    topic: str,
    language: str = "en",
    limit: int = 5
) -> dict:
    """Получает актуальные новости по теме."""
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

# ИЗМЕНЕНИЕ: Исправлены обработчики - теперь возвращают JSONResponse
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Обработчик HTTP исключений."""
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
    """Общий обработчик исключений."""
    logger.error(f"❌ Необработанное исключение: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "Произошла внутренняя ошибка сервера",
            "timestamp": get_current_timestamp()
        }
    )


# ============================================================================
# СОБЫТИЯ ЖИЗНЕННОГО ЦИКЛА
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Выполняется при запуске приложения.
    
    ИЗМЕНЕНИЕ: Добавлена инициализация OpenAI клиента и валидация настроек.
    Это позволяет корректно обрабатывать ошибки конфигурации.
    """
    global openai_client
    
    logger.info("🚀 Запуск Content Generation API...")
    logger.info(f"📌 Порт: {settings.port}")
    logger.info(f"🤖 Модель OpenAI: {settings.openai_model}")
    
    # Проверяем конфигурацию
    try:
        settings.validate()
        
        # Инициализируем OpenAI клиент только после успешной валидации
        openai_client = OpenAI(api_key=settings.openai_api_key)
        logger.info("✅ OpenAI клиент инициализирован")
        
    except ValueError as e:
        logger.error(f"❌ Ошибка конфигурации: {e}")
        logger.warning("⚠️ Приложение запущено, но генерация контента недоступна!")
        # Не падаем - позволяем health check работать
    
    logger.info("✅ Приложение запущено!")


@app.on_event("shutdown")
async def shutdown_event():
    """Выполняется при остановке приложения."""
    logger.info("🛑 Остановка Content Generation API...")
    logger.info("✅ Приложение остановлено!")


