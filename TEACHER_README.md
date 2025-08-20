# Teacher-Student Training Module

Этот репозиторий теперь включает модуль для teacher-student обучения Stable Diffusion моделей. Модуль находится в папке `teacher/` и предоставляет полный pipeline для обучения student моделей с использованием предсчитанных outputs от teacher модели.

## 🚀 Быстрый старт

### 1. Создание sample датасета
```bash
cd teacher
python3 quick_start.py --create_sample
```

### 2. Добавление изображений
Поместите ваши тренировочные изображения в папку `sample_dataset/images/` и соответствующие подписи в `.txt` файлы.

### 3. Запуск полного pipeline
```bash
python3 quick_start.py --data_dir ./sample_dataset --max_train_steps 1000
```

## 📁 Структура модуля

```
teacher/
├── generate_teacher_outputs.py    # Генерация teacher outputs
├── teacher_student_dataset.py     # Датасет для teacher-student обучения
├── train_student.py              # Тренировка student модели
├── quick_start.py                # Скрипт для быстрого старта
├── test_teacher_student.py       # Тестирование функциональности
├── requirements.txt               # Зависимости
└── README.md                     # Подробная документация
```

## 🔧 Основные компоненты

### 1. Teacher Output Generator (`generate_teacher_outputs.py`)
- Прогоняет датасет через teacher модель
- Сохраняет latents, timesteps, text embeddings и teacher predictions
- Оптимизирован для работы на картах с ограниченной памятью

### 2. Teacher-Student Dataset (`teacher_student_dataset.py`)
- Загружает предсчитанные teacher outputs
- Предоставляет данные для тренировки student модели
- Поддерживает батчинг и перемешивание

### 3. Student Trainer (`train_student.py`)
- Обучает student модель подгоняться под teacher predictions
- Использует MSE loss между student и teacher outputs
- Поддерживает различные оптимизаторы и планировщики learning rate

## 💡 Преимущества подхода

1. **Экономия памяти**: Teacher модель не нужна во время тренировки
2. **Быстрое обучение**: Student учится напрямую у teacher
3. **Контролируемость**: Точные target'ы для обучения
4. **Масштабируемость**: Можно использовать разные teacher модели

## 🎯 Оптимизация для RTX 3060 12GB

Для карт с ограниченной памятью рекомендуется:

```bash
# Генерация teacher outputs
python3 generate_teacher_outputs.py \
    --model_path "runwayml/stable-diffusion-v1-5" \
    --train_data_dir "./my_dataset" \
    --output_dir "./teacher_outputs" \
    --resolution 512 512 \
    --use_fp16

# Тренировка student
python3 train_student.py \
    --model_path "runwayml/stable-diffusion-v1-5" \
    --teacher_outputs_dir "./teacher_outputs" \
    --output_dir "./student_outputs" \
    --batch_size 1 \
    --optimizer_type "AdamW8bit" \
    --use_fp16
```

## 📊 Примеры использования

### Базовое обучение
```bash
# Шаг 1: Генерация teacher outputs
python3 generate_teacher_outputs.py \
    --model_path "runwayml/stable-diffusion-v1-5" \
    --train_data_dir "./dataset" \
    --output_dir "./teacher_outputs" \
    --resolution 512 512

# Шаг 2: Тренировка student
python3 train_student.py \
    --model_path "runwayml/stable-diffusion-v1-5" \
    --teacher_outputs_dir "./teacher_outputs" \
    --output_dir "./student_outputs" \
    --max_train_steps 2000
```

### Продвинутое обучение
```bash
python3 train_student.py \
    --model_path "runwayml/stable-diffusion-v1-5" \
    --teacher_outputs_dir "./teacher_outputs" \
    --output_dir "./student_outputs" \
    --max_train_steps 5000 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --optimizer_type "AdamW8bit" \
    --lr_scheduler "cosine" \
    --weight_decay 1e-5
```

## 🔍 Тестирование

Для проверки работоспособности модуля:

```bash
cd teacher
python3 test_teacher_student.py
```

## 📋 Требования

- Python 3.8+
- PyTorch 2.0+
- Diffusers 0.21.0+
- Transformers 4.30.0+
- GPU с минимум 8GB VRAM (рекомендуется 12GB+)

## 📚 Документация

Подробная документация находится в файле `teacher/README.md`.

## 🤝 Интеграция с sd_scripts

Этот модуль полностью совместим с существующим кодом sd_scripts и использует те же библиотеки и подходы. Все функции sd_scripts сохранены и работают как прежде.

## 📝 Лицензия

Модуль распространяется под той же лицензией, что и основной репозиторий sd_scripts.

## 🆘 Поддержка

При возникновении проблем:

1. Проверьте, что все зависимости установлены
2. Убедитесь, что у вас достаточно GPU памяти
3. Используйте `--use_fp16` для экономии памяти
4. Начните с малого количества шагов тренировки

## 🚀 Будущие улучшения

- Поддержка SDXL и SD3 моделей
- Интеграция с существующими тренировочными скриптами
- Поддержка различных loss функций
- Автоматическая оптимизация гиперпараметров