# Базовый образ
FROM python:3.12-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh

# Добавляем uv в PATH
ENV PATH="/root/.local/bin:$PATH"

# Рабочая директория
WORKDIR /app

# Копируем только зависимости (для кеша)
COPY pyproject.toml uv.lock ./

# Установка зависимостей через uv
RUN uv sync --frozen

# Копируем весь проект
COPY . .

# Переменные окружения
ENV PYTHONPATH=/app

# Команда запуска
CMD ["uv", "run", "python", "main.py"]