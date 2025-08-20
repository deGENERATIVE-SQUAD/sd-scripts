# SDXL Teacher-Student Training - Инструкции по использованию

## 🚀 Быстрый старт

### 1. Проверка совместимости
```bash
python teacher/test_compatibility.py
```

### 2. Интерактивная настройка
```bash
python teacher/quick_start.py
```

### 3. Запуск обучения
```bash
python teacher/quick_start.py --run
```

## 📋 Пошаговое руководство

### Шаг 1: Подготовка датасета

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

**Формат подписей**: `.caption` файлы содержат текстовые описания изображений.

### Шаг 2: Выбор моделей

#### Teacher модель
- Рекомендуется: `stabilityai/stable-diffusion-xl-base-1.0`
- Может быть любая SDXL модель (Diffusers или checkpoint)
- Определяет качество обучения

#### Student базовая модель
- Обычно та же модель, что и teacher
- Может быть другой SDXL моделью для transfer learning

### Шаг 3: Генерация teacher outputs

```bash
python teacher/generate_teacher_outputs.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --train_data_dir "./my_dataset" \
    --teacher_output_dir "./teacher_outputs" \
    --resolution 1024 \
    --max_token_length 225 \
    --train_batch_size 1 \
    --mixed_precision fp16 \
    --lowram \
    --xformers
```

**Обязательные параметры:**
- `--pretrained_model_name_or_path`: Путь к teacher модели
- `--train_data_dir`: Путь к датасету
- `--teacher_output_dir`: Директория для сохранения outputs

**Рекомендуемые параметры:**
- `--lowram`: Оптимизация памяти
- `--xformers`: Memory efficient attention
- `--mixed_precision fp16`: Смешанная точность

### Шаг 4: Обучение student модели

```bash
python teacher/train_student.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --train_data_dir "./teacher_outputs" \
    --output_dir "./student_output" \
    --output_name "my_student" \
    --max_train_epochs 10 \
    --max_train_steps 1000 \
    --learning_rate 1e-4 \
    --train_batch_size 1 \
    --mixed_precision fp16 \
    --lowram \
    --xformers \
    --save_every_n_epochs 1 \
    --network_module "networks.lora" \
    --network_dim 64 \
    --network_alpha 32
```

**Обязательные параметры:**
- `--pretrained_model_name_or_path`: Путь к базовой модели для student
- `--train_data_dir`: Путь к директории с teacher outputs
- `--output_dir`: Директория для сохранения обученной модели

## 🔧 Настройка network modules

### LoRA
```bash
--network_module "networks.lora" \
--network_dim 64 \
--network_alpha 32
```

### LoHa
```bash
--network_module "networks.loha" \
--network_dim 64 \
--network_alpha 32
```

### LoKr
```bash
--network_module "networks.lokr" \
--network_dim 64 \
--network_alpha 32
```

### Кастомная сеть
```bash
--network_module "path/to/custom/network.py" \
--network_args "arg1" "arg2"
```

## ⚡ Оптимизация для RTX 3060 12GB

### Рекомендуемые настройки
```bash
# Генерация teacher outputs
--train_batch_size 1 \
--mixed_precision fp16 \
--lowram \
--xformers \
--vae_batch_size 1

# Обучение student
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

## 📊 Мониторинг обучения

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

### Консольные логи
```bash
# Уровень логирования
--console_log_level INFO

# Сохранение логов в файл
--console_log_file "./training.log"
```

## 🔄 Полный pipeline

### Автоматический запуск
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

### Пошаговый запуск
```bash
# Шаг 1: Генерация teacher outputs
python teacher/generate_teacher_outputs.py [параметры]

# Шаг 2: Обучение student
python teacher/train_student.py [параметры]
```

## 📁 Структура outputs

### Teacher outputs
```
teacher_outputs/
├── dataset_info.json          # Информация о датасете
├── sample_000000.pt          # Sample 0
├── sample_000001.pt          # Sample 1
├── sample_000002.pt          # Sample 2
└── ...
```

### Student outputs
```
student_output/
├── my_student-epoch01-step00001000.safetensors  # UNet
├── my_student-te1-epoch01-step00001000.safetensors  # Text Encoder 1
└── my_student-te2-epoch01-step00001000.safetensors  # Text Encoder 2
```

## 🎛️ Дополнительные параметры

### Оптимизация памяти
```bash
--highvram                    # Высокое потребление памяти
--cpu_offload_checkpointing   # Offload чекпоинтов на CPU
--gradient_checkpointing      # Gradient checkpointing
```

### Аугментация
```bash
--color_aug                   # Цветовая аугментация
--flip_aug                    # Горизонтальное отражение
--face_crop_aug_range "0,0"  # Кроп лица
--random_crop                 # Случайный кроп
```

### Кэширование
```bash
--cache_latents               # Кэширование латентов
--cache_text_encoder_outputs  # Кэширование text encoder outputs
--cache_text_encoder_outputs_to_disk  # Кэширование на диск
```

### Оптимизаторы
```bash
--optimizer_type "AdamW"      # Тип оптимизатора
--learning_rate 1e-4          # Скорость обучения
--lr_scheduler_type "cosine"  # Тип scheduler
--weight_decay 0.01           # Weight decay
```

### Loss functions
```bash
--loss_type "l2"              # L2 loss (по умолчанию)
--loss_type "l1"              # L1 loss
--loss_type "huber"           # Huber loss
--loss_type "smooth_l1"       # Smooth L1 loss
```

## 🆘 Устранение неполадок

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

# Включите gradient checkpointing
--gradient_checkpointing
```

### Медленное обучение
```bash
# Включите mixed precision
--mixed_precision fp16

# Кэшируйте text encoder outputs
--cache_text_encoder_outputs

# Используйте gradient accumulation
--gradient_accumulation_steps 4

# Включите xformers
--xformers
```

### Проблемы с загрузкой модели
```bash
# Проверьте путь к модели
--pretrained_model_name_or_path "correct/path/to/model"

# Используйте disable_mmap для больших моделей
--disable_mmap_load_safetensors

# Проверьте формат модели (Diffusers или checkpoint)
```

### Ошибки импорта
```bash
# Убедитесь, что вы в корневой директории sd_scripts
pwd  # должно показать путь к sd_scripts

# Проверьте установку зависимостей
pip list | grep torch
pip list | grep diffusers
pip list | grep accelerate

# Запустите тест совместимости
python teacher/test_compatibility.py
```

## 📚 Примеры конфигураций

### Минимальная конфигурация
```toml
[teacher_generation]
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
train_data_dir = "./my_dataset"
teacher_output_dir = "./teacher_outputs"

[student_training]
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
train_data_dir = "./teacher_outputs"
output_dir = "./student_output"
network_module = "networks.lora"
network_dim = 32
network_alpha = 1.0
```

### Полная конфигурация
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
num_inference_steps = 1000
cache_text_encoder_outputs = true

[student_training]
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
train_data_dir = "./teacher_outputs"
output_dir = "./student_output"
output_name = "my_student"
max_train_epochs = 10
max_train_steps = 1000
learning_rate = 1e-4
train_batch_size = 1
mixed_precision = "fp16"
lowram = true
xformers = true
save_every_n_epochs = 1
network_module = "networks.lora"
network_dim = 64
network_alpha = 32
optimizer_type = "AdamW"
lr_scheduler_type = "cosine"
loss_type = "l2"
gradient_accumulation_steps = 4
log_with = "tensorboard"
logging_dir = "./logs"
```

## 🎯 Лучшие практики

1. **Начните с малого**: Используйте небольшой датасет для тестирования
2. **Мониторьте память**: Следите за использованием GPU памяти
3. **Экспериментируйте**: Пробуйте разные network modules и параметры
4. **Сохраняйте чекпоинты**: Регулярно сохраняйте прогресс обучения
5. **Используйте логирование**: TensorBoard или W&B для мониторинга
6. **Оптимизируйте память**: Используйте lowram и xformers для RTX 3060

## 🔮 Расширенные возможности

### Кастомные loss functions
```python
# В train_student.py можно модифицировать compute_loss метод
def compute_loss(self, student_pred, teacher_pred, loss_type="l2"):
    if loss_type == "custom":
        # Ваша кастомная loss function
        loss = custom_loss_function(student_pred, teacher_pred)
    else:
        # Стандартные loss functions
        loss = super().compute_loss(student_pred, teacher_pred, loss_type)
    return loss
```

### Кастомные network modules
```python
# Создайте свой network module
class CustomNetwork(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Ваша архитектура
    
    def forward(self, x):
        # Ваш forward pass
        return output

# Используйте в обучении
--network_module "path/to/custom_network.py"
```

---

**🎓 Teacher-Student Training** - эффективный способ обучения SDXL моделей!