ML Project — Применение, развёртывание и анализ систем машинного обучения

О проекте
Проект реализует полный цикл ML-инженерии: от прототипирования моделей в Jupyter Notebook до развёртывания REST API с автоматическими тестами и рекомендательной системы. В работе представлены четыре прикладные ML-задачи (Sentiment Analysis, ASR, Image Classification, Object Detection), API-слой на FastAPI с CI/CD пайплайном и RecSys на датасете MovieLens 100K.

Структура репозитория
ml-projects/
├── part1_ml_tasks/
│ └── Серия_1.ipynb # 4 ML-задачи (NLP, Audio, CV×2)
├── part2_fastapi/
│ ├── main.py # FastAPI приложение (Sentiment Analysis)
│ ├── test_main.py # Модульные тесты API
│ ├── requirements.txt # Зависимости проекта
│ └── .github/workflows/
│ └── test.yml # GitHub Actions CI/CD
├── part3_recsys/
│ └── Серия_3.ipynb # Рекомендательная система (MovieLens)
├── Отчет.pdf # Финальный отчёт по проекту
└── .gitignore # Исключения для Git

Стек технологий
- Python, FastAPI, Uvicorn, Pydantic v2 — HTTP API, схемы и валидация входа/выхода
- Hugging Face Transformers + PyTorch — Sentiment Analysis (`blanchefort/rubert-base-cased-sentiment`), ASR (`openai/whisper-tiny`)
- TensorFlow / Keras Applications** — классификация изображений (`EfficientNetB0`, веса ImageNet)
- Ultralytics YOLOv8 + OpenCV — детекция объектов
- pandas, NumPy, scikit-learn — подготовка данных и алгоритмы RecSys (TF-IDF, косинусное сходство)
- librosa — загрузка и ресемплинг аудио (.wav, .ogg) до 16 кГц
- pytest, FastAPI TestClient — unit/API-тестирование
- GitHub Actions — автоматический запуск `pytest` на push/PR

Запуск
Часть 1: ML-задачи и RecSys
Откройте `part1_ml_tasks/Серия_1.ipynb` и `part3_recsys/Серия_3.ipynb` и запускайте ячейки последовательно.

Часть 2: FastAPI API
cd part2_fastapi
pip install -r requirements.txt
uvicorn main:app --reload

API будет доступен по адресу: http://127.0.0.1:8000
Интерактивная документация: http://127.0.0.1:8000/docs

Тестирование 
cd part2_fastapi
pytest -v

Описание задач
Тема 1: Sentiment Analysis (NLP)
Классификация русскоязычного текста по эмоциональной окраске (POSITIVE / NEGATIVE / NEUTRAL) с использованием модели RuBERT.
Тема 2: Распознавание речи (ASR)
Транскрипция аудио в текст с помощью Whisper-tiny. Поддержка английского и русского языков, включая реальные голосовые сообщения из мессенджеров.
Тема 3: Классификация изображений (CV)
Определение объекта на изображении с помощью EfficientNetB0, предобученной на ImageNet (1000 классов).
Тема 4: Детекция объектов (CV)
Поиск и локализация объектов в кадре с помощью YOLOv8n в реальном времени.
Часть 2: REST API
Обёртка модели Sentiment Analysis в FastAPI с Pydantic-валидацией, Swagger UI и автоматическим тестированием через GitHub Actions.
Часть 3: Рекомендательная система
RecSys на датасете MovieLens 100K (100 000 оценок, 943 пользователя, 1682 фильма). Реализованы 4 подхода:
Popularity-based — эвристика по среднему рейтингу
Genre-based — подбор по жанровым тегам
Content-Based Filtering — TF-IDF + косинусное сходство
Collaborative Filtering (Item-Based) — сходство на основе паттернов оценок
