# Teacher-Student LoRA Training

Этот проект реализует процесс обучения типа teacher-student для LoRA в sd_scripts. Принцип работы заключается в том, что сначала teacher модель обрабатывает датасет и сохраняет свои выходы, а затем student модель учится подгоняться под эти предсчитанные выходы.

## ✨ Возможности

- **Поддержка SD и SDXL моделей**: Автоматическое определение типа модели
- **LyCORIS поддержка**: Использование LyCORIS вместо стандартного LoRA
- **Экономия VRAM**: Teacher модель не держится в памяти во время обучения
- **Быстрое обучение**: Student учится напрямую у teacher
- **Оптимизация для RTX 3060**: Настроено для карт с 12GB памяти

## Принцип работы

1. **Подготовка teacher outputs**: Запускаете teacher модель на датасете и сохраняете (latents, timesteps, text_embeddings, eps_teacher) в файлы .npz
2. **Обучение student**: Student модель читает эти предсчитанные teacher outputs и учится повторять их поведение
3. **Экономия памяти**: Teacher модель не держится в памяти во время обучения student

## Преимущества

- **Экономия VRAM**: Не нужно держать teacher модель в памяти во время обучения
- **Быстрое обучение**: Student учится напрямую у teacher, а не с нуля
- **Консистентность**: Все примеры обрабатываются одинаково teacher моделью
- **Масштабируемость**: Можно использовать мощные teacher модели на GPU с большим объемом памяти
- **Гибкость**: Поддержка различных типов сетей (LoRA, LyCORIS)

## Требования

- Python 3.8+
- PyTorch 2.0+
- Diffusers
- Transformers
- sd_scripts (этот проект)
- GPU с минимум 12GB VRAM (для RTX 3060)
- **Для LyCORIS**: `pip install lycoris`

## Установка

1. Клонируйте репозиторий sd_scripts
2. Скопируйте файлы из этого проекта в соответствующие директории
3. Установите зависимости:

```bash
pip install -r requirements.txt

# Для LyCORIS поддержки
pip install lycoris
```

## Использование

### Быстрый запуск (все в одном)

```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --base_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir ./trained_model \
    --network_module lycoris.kohya \
    --network_dim 64 \
    --network_alpha 64
```

### Пошаговый запуск

#### Шаг 1: Подготовка teacher outputs

**Для SD модели:**
```bash
python finetune/prepare_teacher_outputs.py \
    --train_data_dir /path/to/your/images \
    --in_json /path/to/metadata.json \
    --teacher_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --output_dir /path/to/teacher_outputs \
    --mixed_precision fp16 \
    --max_resolution "512,512" \
    --min_bucket_reso 256 \
    --max_bucket_reso 1024 \
    --bucket_reso_steps 64
```

**Для SDXL модели:**
```bash
python finetune/prepare_teacher_outputs.py \
    --train_data_dir /path/to/your/images \
    --in_json /path/to/metadata.json \
    --teacher_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir /path/to/teacher_outputs \
    --mixed_precision fp16 \
    --max_resolution "1024,1024" \
    --min_bucket_reso 512 \
    --max_bucket_reso 2048 \
    --bucket_reso_steps 64
```

#### Шаг 2: Обучение student модели

**С LoRA:**
```bash
python train_network_teacher_student.py \
    --train_data_dir /path/to/your/images \
    --teacher_outputs_dir /path/to/teacher_outputs \
    --output_dir /path/to/trained_lora \
    --model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --train_batch_size 1 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --network_dim 64 \
    --network_alpha 64 \
    --network_module networks.lora \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 4
```

**С LyCORIS:**
```bash
python train_network_teacher_student.py \
    --train_data_dir /path/to/your/images \
    --teacher_outputs_dir /path/to/teacher_outputs \
    --output_dir /path/to/trained_lycoris \
    --model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --train_batch_size 1 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --network_dim 64 \
    --network_alpha 64 \
    --network_module lycoris.kohya \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 4
```

## Параметры

### Основные параметры

- `--train_data_dir`: Директория с изображениями для обучения
- `--in_json`: JSON файл с метаданными
- `--teacher_model_name_or_path`: Путь к teacher модели
- `--output_dir`: Директория для сохранения teacher outputs
- `--mixed_precision`: Точность вычислений (fp16 для экономии памяти)

### Параметры разрешения

**Для SD:**
- `--max_resolution`: "512,512" (по умолчанию)
- `--min_bucket_reso`: 256
- `--max_bucket_reso`: 1024

**Для SDXL:**
- `--max_resolution`: "1024,1024" (рекомендуется)
- `--min_bucket_reso`: 512
- `--max_bucket_reso`: 2048

### Параметры сети

- `--network_module`: Тип сети ("networks.lora" или "lycoris.kohya")
- `--network_dim`: Размерность сети (16-128)
- `--network_alpha`: Альфа параметр сети
- `--network_weights`: Путь к весам для LyCORIS (опционально)

## Структура файлов

```
finetune/
├── prepare_teacher_outputs.py    # Скрипт подготовки teacher outputs
library/
├── teacher_student_dataset.py    # Dataset для teacher-student обучения
train_network_teacher_student.py  # Основной скрипт обучения
run_teacher_student_training.py   # Автоматизированный pipeline
README_teacher_student.md         # Этот файл
```

## Формат teacher outputs

Каждый файл .npz содержит:
- `latents`: Латентные представления изображений
- `timesteps`: Временные шаги для диффузии
- `text_embeddings`: Текстовые эмбеддинги от teacher модели
- `eps_teacher`: Предсказания шума от teacher модели
- `caption`: Текстовая подпись к изображению
- `image_path`: Путь к исходному изображению
- `original_size`: Оригинальный размер изображения
- `is_sdxl`: Флаг SDXL модели

## Оптимизация для RTX 3060 12GB

### SD модели
```bash
--train_batch_size 1
--mixed_precision fp16
--gradient_accumulation_steps 4
--max_resolution "512,512"
--network_dim 32
```

### SDXL модели
```bash
--train_batch_size 1
--mixed_precision fp16
--gradient_accumulation_steps 8
--max_resolution "1024,1024"
--network_dim 64
--gradient_checkpointing
```

## Примеры использования

### SD + LoRA
```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "runwayml/stable-diffusion-v1-5" \
    --base_model "runwayml/stable-diffusion-v1-5" \
    --output_dir ./trained_lora \
    --network_module networks.lora \
    --network_dim 32 \
    --network_alpha 32
```

### SDXL + LyCORIS
```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --base_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir ./trained_lycoris \
    --network_module lycoris.kohya \
    --network_dim 64 \
    --network_alpha 64 \
    --max_resolution "1024,1024"
```

### Смешанные модели (teacher SDXL, student SD)
```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --base_model "runwayml/stable-diffusion-v1-5" \
    --output_dir ./trained_mixed \
    --network_module networks.lora \
    --network_dim 32
```

## Troubleshooting

### Ошибка "CUDA out of memory"
- Уменьшите `--train_batch_size` до 1
- Увеличьте `--gradient_accumulation_steps`
- Используйте `--mixed_precision fp16`
- Для SDXL: используйте `--gradient_checkpointing`

### Ошибка "LyCORIS not installed"
```bash
pip install lycoris
```

### Ошибка "Teacher output not found"
- Проверьте, что `--teacher_outputs_dir` содержит файлы .npz
- Убедитесь, что имена файлов совпадают с именами изображений
- Запустите сначала `prepare_teacher_outputs.py`

### Медленное обучение SDXL
- Увеличьте `--gradient_accumulation_steps`
- Уменьшите `--max_data_loader_n_workers`
- Используйте SSD для хранения teacher outputs
- Рассмотрите использование `--gradient_checkpointing`

## Дополнительные возможности

### Обучение text encoder
```bash
--train_text_encoder
```

### Использование v-parameterization
```bash
--v_parameterization
```

### Настройка LyCORIS
```bash
--network_module lycoris.kohya
--network_dim 64 --network_alpha 64
--network_dropout 0.1
```

### Продолжение обучения
```bash
--network_weights /path/to/existing/weights
```

## Тестирование

Запустите тесты для проверки функциональности:

```bash
python test_teacher_student.py
```

## Лицензия

Этот проект использует ту же лицензию, что и sd_scripts.

## Поддержка

При возникновении проблем:
1. Проверьте логи на наличие ошибок
2. Убедитесь, что все зависимости установлены
3. Проверьте соответствие версий PyTorch и Diffusers
4. Для LyCORIS: убедитесь, что установлен пакет `lycoris`
5. Создайте issue в репозитории проекта