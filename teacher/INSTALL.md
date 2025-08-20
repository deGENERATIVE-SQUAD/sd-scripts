# Установка и настройка SDXL Teacher-Student Training

## Требования

### Системные требования
- **ОС**: Linux (рекомендуется Ubuntu 20.04+), Windows 10/11, macOS
- **Python**: 3.8 или выше
- **GPU**: NVIDIA GPU с поддержкой CUDA (рекомендуется RTX 3060 12GB или лучше)
- **RAM**: Минимум 16GB, рекомендуется 32GB+
- **Дисковое пространство**: Минимум 50GB свободного места

### Программные требования
- **CUDA**: 11.8 или выше
- **PyTorch**: 2.0 или выше
- **Git**: для клонирования репозитория

## Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts
```

### 2. Установка зависимостей

#### Автоматическая установка (рекомендуется)

```bash
# Установка всех зависимостей
pip install -r requirements.txt

# Установка дополнительных зависимостей для teacher-student
pip install accelerate diffusers transformers
```

#### Ручная установка

```bash
# Основные зависимости
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install accelerate diffusers transformers
pip install safetensors pillow opencv-python
pip install xformers  # для оптимизации памяти
pip install lion-pytorch  # опционально, для Lion оптимизатора
```

### 3. Проверка установки

```bash
# Запуск теста совместимости
python teacher/test_compatibility.py
```

Если все тесты прошли успешно, вы увидите:
```
🎉 All tests passed! The teacher-student module is ready to use.
```

## Настройка

### 1. Подготовка датасета

Создайте структуру датасета:

```
my_dataset/
├── image1.jpg
├── image1.caption
├── image2.jpg
├── image2.caption
├── image3.jpg
└── image3.caption
```

**Формат подписей**: `.caption` файлы должны содержать текстовые описания изображений.

### 2. Подготовка моделей

#### Teacher модель
- Может быть любая SDXL модель в формате Diffusers или checkpoint
- Рекомендуется использовать качественную модель (например, `stabilityai/stable-diffusion-xl-base-1.0`)

#### Student базовая модель
- Обычно та же модель, что и teacher
- Может быть другой SDXL моделью для transfer learning

### 3. Создание конфигурации

#### Автоматическая настройка (рекомендуется)

```bash
python teacher/quick_start.py
```

Следуйте интерактивным инструкциям для настройки всех параметров.

#### Ручная настройка

Создайте файл `my_config.toml`:

```toml
[teacher_generation]
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
train_data_dir = "./my_dataset"
teacher_output_dir = "./teacher_outputs"
resolution = 1024
max_token_length = 225
train_batch_size = 1
mixed_precision = "fp16"
lowram = true
xformers = true

[student_training]
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
train_data_dir = "./teacher_outputs"
output_dir = "./student_output"
output_name = "my_student"
max_train_epochs = 10
learning_rate = 1e-4
network_module = "networks.lora"
network_dim = 64
network_alpha = 32
```

## Использование

### Быстрый старт

```bash
# Интерактивная настройка и запуск
python teacher/quick_start.py --run
```

### Пошаговое выполнение

#### Шаг 1: Генерация teacher outputs

```bash
python teacher/generate_teacher_outputs.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --train_data_dir "./my_dataset" \
    --teacher_output_dir "./teacher_outputs" \
    --resolution 1024 \
    --lowram \
    --xformers
```

#### Шаг 2: Обучение student модели

```bash
python teacher/train_student.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --train_data_dir "./teacher_outputs" \
    --output_dir "./student_output" \
    --output_name "my_student" \
    --max_train_epochs 10 \
    --learning_rate 1e-4 \
    --network_module "networks.lora" \
    --network_dim 64 \
    --network_alpha 32 \
    --lowram \
    --xformers
```

### Полный pipeline

```bash
python teacher/run_teacher_student.py \
    --teacher_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --dataset_path "./my_dataset" \
    --student_base_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --max_epochs 10 \
    --learning_rate 1e-4 \
    --network_module "networks.lora" \
    --network_dim 64 \
    --network_alpha 32 \
    --lowram \
    --xformers
```

## Оптимизация для RTX 3060 12GB

### Рекомендуемые настройки

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

### Ключевые параметры для экономии памяти

- `--lowram`: Включает режим низкого потребления памяти
- `--xformers`: Использует memory efficient attention
- `--train_batch_size 1`: Минимальный размер батча
- `--mixed_precision fp16`: Смешанная точность для экономии памяти
- `--gradient_accumulation_steps 4`: Эффективный батч-сайз 4

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

### Ошибки импорта

```bash
# Убедитесь, что вы в корневой директории sd_scripts
pwd  # должно показать путь к sd_scripts

# Проверьте установку зависимостей
pip list | grep torch
pip list | grep diffusers
pip list | grep accelerate
```

## Мониторинг и логирование

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

### Логи консоли

```bash
# Уровень логирования
--console_log_level INFO

# Сохранение логов в файл
--console_log_file "./training.log"
```

## Примеры использования

### Обучение LoRA с teacher-student

```bash
# 1. Генерируем teacher outputs
python teacher/generate_teacher_outputs.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --train_data_dir "./my_dataset" \
    --teacher_output_dir "./teacher_outputs" \
    --resolution 1024 \
    --lowram \
    --xformers

# 2. Обучаем LoRA с teacher-student
python teacher/train_student.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --train_data_dir "./teacher_outputs" \
    --output_dir "./student_output" \
    --output_name "my_lora" \
    --max_train_epochs 5 \
    --learning_rate 1e-4 \
    --network_module "networks.lora" \
    --network_dim 64 \
    --network_alpha 32 \
    --lowram \
    --xformers
```

### Обучение с кастомной сетью

```bash
python teacher/train_student.py \
    --pretrained_model_name_or_path "path/to/base/model" \
    --train_data_dir "./teacher_outputs" \
    --network_module "path/to/custom/network.py" \
    --network_args "custom_arg1" "custom_arg2" \
    --max_train_epochs 10 \
    --learning_rate 5e-5
```

## Поддержка

### Документация
- [README.md](README.md) - Основная документация
- [example_config.toml](example_config.toml) - Пример конфигурации
- [test_compatibility.py](test_compatibility.py) - Тест совместимости

### Полезные ссылки
- [SD Scripts](https://github.com/kohya-ss/sd-scripts) - Основной репозиторий
- [DMD2 Paper](https://arxiv.org/pdf/2405.14867) - Исходная статья метода
- [SDXL Documentation](https://huggingface.co/docs/diffusers/using-diffusers/sdxl) - Документация SDXL

### Сообщество
- GitHub Issues: для багов и feature requests
- Discussions: для вопросов и обсуждений
- Discord: для живого общения (если доступно)