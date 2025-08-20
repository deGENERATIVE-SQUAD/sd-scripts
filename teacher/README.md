# SDXL Teacher-Student Training

Этот модуль реализует процесс обучения типа teacher-student для SDXL моделей, основанный на методе дистилляции DMD2 (https://arxiv.org/pdf/2405.14867).

## Принцип работы

1. **Генерация teacher outputs**: Датасет прогоняется через teacher модель, сохраняются:
   - `latents` - латентные представления изображений
   - `timesteps` - временные шаги
   - `text_embeddings` - текстовые эмбеддинги
   - `teacher_noise_pred` - предсказания шума от teacher модели

2. **Обучение student**: Student модель учится повторять teacher, а не учиться с нуля, используя предсчитанные outputs.

## Структура файлов

- `generate_teacher_outputs.py` - скрипт для генерации teacher outputs
- `train_student.py` - скрипт для обучения student модели
- `README.md` - данный файл с инструкциями

## Требования

- Python 3.8+
- PyTorch 2.0+
- Diffusers
- Accelerate
- SDXL модель для teacher
- Датасет изображений с подписями

## Использование

### Шаг 1: Генерация Teacher Outputs

```bash
python teacher/generate_teacher_outputs.py \
    --pretrained_model_name_or_path "path/to/teacher/model" \
    --train_data_dir "path/to/dataset" \
    --teacher_output_dir "./teacher_outputs" \
    --resolution 1024 \
    --max_token_length 225 \
    --train_batch_size 1 \
    --mixed_precision fp16 \
    --lowram \
    --num_inference_steps 1000
```

**Основные параметры:**
- `--pretrained_model_name_or_path` - путь к teacher модели (обязательно)
- `--train_data_dir` - путь к датасету (обязательно)
- `--teacher_output_dir` - директория для сохранения teacher outputs (обязательно)
- `--resolution` - разрешение изображений (по умолчанию 1024)
- `--max_token_length` - максимальная длина токенов (по умолчанию 225)
- `--train_batch_size` - размер батча (по умолчанию 1)
- `--mixed_precision` - тип смешанной точности (по умолчанию fp16)
- `--lowram` - режим низкого потребления памяти
- `--num_inference_steps` - количество шагов для генерации шума (по умолчанию 1000)

### Шаг 2: Обучение Student Модели

```bash
python teacher/train_student.py \
    --pretrained_model_name_or_path "path/to/student/base/model" \
    --train_data_dir "./teacher_outputs" \
    --output_dir "./student_output" \
    --output_name "student_model" \
    --max_train_epochs 10 \
    --max_train_steps 1000 \
    --learning_rate 1e-4 \
    --train_batch_size 1 \
    --mixed_precision fp16 \
    --lowram \
    --save_every_n_epochs 1 \
    --network_module "networks.lora" \
    --network_dim 32 \
    --network_alpha 1.0
```

**Основные параметры:**
- `--pretrained_model_name_or_path` - путь к базовой модели для student (обязательно)
- `--train_data_dir` - путь к директории с teacher outputs (обязательно)
- `--output_dir` - директория для сохранения обученной модели
- `--output_name` - имя выходных файлов
- `--max_train_epochs` - максимальное количество эпох
- `--max_train_steps` - максимальное количество шагов
- `--learning_rate` - скорость обучения
- `--network_module` - модуль сети (например, LoRA)
- `--network_dim` - размерность сети
- `--network_alpha` - альфа параметр сети

## Поддерживаемые функции

### Все оригинальные аргументы sd_scripts

Скрипты поддерживают все аргументы оригинального `sdxl_train_network.py`, включая:

- **Оптимизация памяти**: `--lowram`, `--highvram`, `--xformers`, `--sdpa`
- **Смешанная точность**: `--mixed_precision`, `--full_fp16`, `--full_bf16`
- **Сети**: `--network_module`, `--network_dim`, `--network_alpha`
- **Оптимизаторы**: `--optimizer_type`, `--learning_rate`, `--lr_scheduler_type`
- **Аугментация**: `--color_aug`, `--flip_aug`, `--face_crop_aug_range`
- **Кэширование**: `--cache_latents`, `--cache_text_encoder_outputs`

### Поддержка LyCORIS и других network modules

```bash
# LoRA
--network_module "networks.lora" --network_dim 32 --network_alpha 1.0

# LoHa
--network_module "networks.loha" --network_dim 32 --network_alpha 1.0

# LoKr
--network_module "networks.lokr" --network_dim 32 --network_alpha 1.0

# Custom network
--network_module "path/to/custom/network" --network_args "arg1" "arg2"
```

## Оптимизация для RTX 3060 12GB

Для карты RTX 3060 12GB рекомендуется использовать:

```bash
# Генерация teacher outputs
python teacher/generate_teacher_outputs.py \
    --pretrained_model_name_or_path "path/to/teacher" \
    --train_data_dir "path/to/dataset" \
    --teacher_output_dir "./teacher_outputs" \
    --train_batch_size 1 \
    --mixed_precision fp16 \
    --lowram \
    --xformers \
    --vae_batch_size 1

# Обучение student
python teacher/train_student.py \
    --pretrained_model_name_or_path "path/to/student/base" \
    --train_data_dir "./teacher_outputs" \
    --train_batch_size 1 \
    --mixed_precision fp16 \
    --lowram \
    --xformers \
    --gradient_accumulation_steps 4
```

## Структура Teacher Outputs

После генерации в `teacher_output_dir` будут созданы:

```
teacher_outputs/
├── dataset_info.json          # Информация о датасете
├── sample_000000.pt          # Sample 0
├── sample_000001.pt          # Sample 1
├── sample_000002.pt          # Sample 2
└── ...
```

Каждый `.pt` файл содержит:
- `latents` - латентные представления
- `noisy_latents` - зашумленные латенты
- `timesteps` - временные шаги
- `text_embeddings1` - эмбеддинги первого text encoder
- `text_embeddings2` - эмбеддинги второго text encoder
- `pool2` - pooled embeddings
- `teacher_noise_pred` - предсказания teacher
- `original_size` - оригинальные размеры
- `crop_top_left` - позиции кропа
- `target_size` - целевые размеры

## Примеры использования

### Простой пример с LoRA

```bash
# 1. Генерируем teacher outputs
python teacher/generate_teacher_outputs.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --train_data_dir "./my_dataset" \
    --teacher_output_dir "./teacher_outputs" \
    --resolution 1024 \
    --lowram \
    --xformers

# 2. Обучаем student с LoRA
python teacher/train_student.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --train_data_dir "./teacher_outputs" \
    --output_dir "./student_output" \
    --output_name "my_student" \
    --max_train_epochs 5 \
    --learning_rate 1e-4 \
    --network_module "networks.lora" \
    --network_dim 64 \
    --network_alpha 32 \
    --lowram \
    --xformers
```

### Пример с кастомной сетью

```bash
# Обучение с кастомной сетью
python teacher/train_student.py \
    --pretrained_model_name_or_path "path/to/base/model" \
    --train_data_dir "./teacher_outputs" \
    --network_module "path/to/custom/network.py" \
    --network_args "custom_arg1" "custom_arg2" \
    --max_train_epochs 10 \
    --learning_rate 5e-5
```

## Мониторинг обучения

### TensorBoard

```bash
# Запуск TensorBoard
tensorboard --logdir ./logs

# В скрипте обучения
--log_with tensorboard --logging_dir ./logs
```

### Weights & Biases

```bash
# В скрипте обучения
--log_with wandb --wandb_run_name "my_experiment" --wandb_api_key "your_key"
```

## Советы по оптимизации

1. **Память**: Используйте `--lowram` и `--xformers` для экономии памяти
2. **Батч-сайз**: Начните с `train_batch_size=1`, увеличивайте при возможности
3. **Смешанная точность**: Используйте `--mixed_precision fp16` для ускорения
4. **Кэширование**: Включите `--cache_text_encoder_outputs` для ускорения
5. **Gradient accumulation**: Используйте `--gradient_accumulation_steps` для увеличения эффективного батч-сайза

## Устранение неполадок

### Ошибки памяти

```bash
# Уменьшите batch size
--train_batch_size 1

# Включите lowram режим
--lowram

# Используйте xformers
--xformers

# Уменьшите resolution
--resolution 512
```

### Медленное обучение

```bash
# Включите mixed precision
--mixed_precision fp16

# Кэшируйте text encoder outputs
--cache_text_encoder_outputs

# Используйте gradient accumulation
--gradient_accumulation_steps 4
```

### Проблемы с загрузкой модели

```bash
# Проверьте путь к модели
--pretrained_model_name_or_path "correct/path/to/model"

# Используйте disable_mmap для больших моделей
--disable_mmap_load_safetensors
```

## Лицензия

Этот код основан на sd_scripts и следует тем же лицензионным условиям.