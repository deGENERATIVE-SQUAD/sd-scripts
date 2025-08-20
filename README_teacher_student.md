# Teacher-Student LoRA Training

Этот проект реализует процесс обучения типа teacher-student для LoRA в sd_scripts. Принцип работы заключается в том, что сначала teacher модель обрабатывает датасет и сохраняет свои выходы, а затем student модель учится подгоняться под эти предсчитанные выходы.

## Принцип работы

1. **Подготовка teacher outputs**: Запускаете teacher модель на датасете и сохраняете (latents, timesteps, text_embeddings, eps_teacher) в файлы .npz
2. **Обучение student**: Student модель читает эти предсчитанные teacher outputs и учится повторять их поведение
3. **Экономия памяти**: Teacher модель не держится в памяти во время обучения student

## Преимущества

- **Экономия VRAM**: Не нужно держать teacher модель в памяти во время обучения
- **Быстрое обучение**: Student учится напрямую у teacher, а не с нуля
- **Консистентность**: Все примеры обрабатываются одинаково teacher моделью
- **Масштабируемость**: Можно использовать мощные teacher модели на GPU с большим объемом памяти

## Требования

- Python 3.8+
- PyTorch 2.0+
- Diffusers
- Transformers
- sd_scripts (этот проект)
- GPU с минимум 12GB VRAM (для RTX 3060)

## Установка

1. Клонируйте репозиторий sd_scripts
2. Скопируйте файлы из этого проекта в соответствующие директории
3. Установите зависимости:

```bash
pip install -r requirements.txt
```

## Использование

### Шаг 1: Подготовка teacher outputs

Сначала запустите скрипт для подготовки teacher outputs:

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

**Параметры:**
- `--train_data_dir`: Директория с изображениями для обучения
- `--in_json`: JSON файл с метаданными (как в обычном обучении)
- `--teacher_model_name_or_path`: Путь к teacher модели
- `--output_dir`: Директория для сохранения teacher outputs
- `--mixed_precision`: Точность вычислений (fp16 для экономии памяти)
- `--max_resolution`: Максимальное разрешение изображений
- `--min_bucket_reso`, `--max_bucket_reso`: Диапазон разрешений для bucketing
- `--bucket_reso_steps`: Шаг между разрешениями bucketing

### Шаг 2: Обучение student модели

После подготовки teacher outputs запустите обучение student модели:

```bash
python train_network_teacher_student.py \
    --train_data_dir /path/to/your/images \
    --teacher_outputs_dir /path/to/teacher_outputs \
    --output_dir /path/to/trained_lora \
    --model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --train_batch_size 1 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --network_dim 32 \
    --network_alpha 32 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 4 \
    --save_every_n_steps 1000 \
    --logging_steps 10
```

**Параметры:**
- `--train_data_dir`: Директория с изображениями (та же, что и для teacher)
- `--teacher_outputs_dir`: Директория с teacher outputs (из шага 1)
- `--output_dir`: Директория для сохранения обученной LoRA
- `--model_name_or_path`: Базовая модель для обучения LoRA
- `--train_batch_size`: Размер батча (рекомендуется 1 для экономии памяти)
- `--num_train_epochs`: Количество эпох обучения
- `--learning_rate`: Скорость обучения
- `--network_dim`: Размерность LoRA сети
- `--network_alpha`: Альфа параметр LoRA
- `--gradient_accumulation_steps`: Количество шагов накопления градиентов
- `--save_every_n_steps`: Сохранение модели каждые N шагов
- `--logging_steps`: Логирование каждые N шагов

## Структура файлов

```
finetune/
├── prepare_teacher_outputs.py    # Скрипт подготовки teacher outputs
library/
├── teacher_student_dataset.py    # Dataset для teacher-student обучения
train_network_teacher_student.py  # Основной скрипт обучения
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

## Оптимизация для RTX 3060 12GB

Для карты с ограниченной памятью:

1. **Используйте fp16**: `--mixed_precision fp16`
2. **Маленький batch size**: `--train_batch_size 1`
3. **Gradient accumulation**: `--gradient_accumulation_steps 4` или больше
4. **Оптимизируйте разрешение**: Используйте `--max_resolution "512,512"` или меньше
5. **Ограничьте LoRA размер**: `--network_dim 16` или `--network_dim 32`

## Пример полного workflow

```bash
# 1. Подготовка teacher outputs
python finetune/prepare_teacher_outputs.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --output_dir ./teacher_outputs \
    --mixed_precision fp16

# 2. Обучение student LoRA
python train_network_teacher_student.py \
    --train_data_dir ./dataset \
    --teacher_outputs_dir ./teacher_outputs \
    --output_dir ./trained_lora \
    --model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --train_batch_size 1 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --network_dim 32 \
    --network_alpha 32 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 4
```

## Troubleshooting

### Ошибка "CUDA out of memory"
- Уменьшите `--train_batch_size` до 1
- Увеличьте `--gradient_accumulation_steps`
- Используйте `--mixed_precision fp16`
- Уменьшите `--max_resolution`

### Ошибка "Teacher output not found"
- Проверьте, что `--teacher_outputs_dir` содержит файлы .npz
- Убедитесь, что имена файлов совпадают с именами изображений
- Запустите сначала `prepare_teacher_outputs.py`

### Медленное обучение
- Увеличьте `--gradient_accumulation_steps`
- Уменьшите `--max_data_loader_n_workers`
- Используйте SSD для хранения teacher outputs

## Дополнительные возможности

### Обучение text encoder
```bash
--train_text_encoder
```

### Использование v-parameterization
```bash
--v_parameterization
```

### Настройка LoRA параметров
```bash
--network_dim 64 --network_alpha 64  # Большая LoRA сеть
--network_dropout 0.1                 # Dropout для регуляризации
```

## Лицензия

Этот проект использует ту же лицензию, что и sd_scripts.

## Поддержка

При возникновении проблем:
1. Проверьте логи на наличие ошибок
2. Убедитесь, что все зависимости установлены
3. Проверьте соответствие версий PyTorch и Diffusers
4. Создайте issue в репозитории проекта