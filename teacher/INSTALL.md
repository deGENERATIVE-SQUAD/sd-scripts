# Установка Teacher-Student модуля

Этот документ содержит пошаговые инструкции по установке и настройке teacher-student модуля для Stable Diffusion.

## Предварительные требования

### Системные требования
- **OS**: Linux (рекомендуется Ubuntu 20.04+), Windows 10+, macOS 10.15+
- **Python**: 3.8 или выше
- **RAM**: Минимум 16GB (рекомендуется 32GB+)
- **GPU**: NVIDIA GPU с минимум 8GB VRAM (рекомендуется 12GB+)
- **Диск**: Минимум 10GB свободного места

### Поддерживаемые GPU
- **RTX 3060 12GB**: Полная поддержка, оптимизировано
- **RTX 3070/3080**: Полная поддержка
- **RTX 3090/4090**: Полная поддержка, можно увеличить batch size
- **GTX 1660/1060**: Ограниченная поддержка (только FP16)

## Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-repo/sd_scripts.git
cd sd_scripts
```

### 2. Создание виртуального окружения (рекомендуется)

```bash
# Создание виртуального окружения
python3 -m venv teacher_env

# Активация в Linux/macOS
source teacher_env/bin/activate

# Активация в Windows
teacher_env\Scripts\activate
```

### 3. Установка зависимостей

```bash
# Переход в папку teacher
cd teacher

# Установка базовых зависимостей
pip install -r requirements.txt

# Или установка по частям
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate
pip install safetensors tqdm pillow numpy
```

### 4. Установка CUDA (если не установлен)

```bash
# Проверка версии CUDA
nvidia-smi

# Установка PyTorch с поддержкой CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5. Установка дополнительных оптимизаторов (опционально)

```bash
# 8-bit оптимизаторы для экономии памяти
pip install bitsandbytes

# Flash Attention для ускорения (если поддерживается)
pip install flash-attn --no-build-isolation
```

## Проверка установки

### 1. Проверка GPU

```bash
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

### 2. Проверка зависимостей

```bash
python3 -c "
import diffusers, transformers, torch
print('All dependencies imported successfully!')
print(f'Diffusers version: {diffusers.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'PyTorch version: {torch.__version__}')
"
```

### 3. Запуск тестов

```bash
cd teacher
python3 test_teacher_student.py
```

## Настройка для разных GPU

### RTX 3060 12GB (рекомендуемые настройки)

```bash
# Генерация teacher outputs
python3 generate_teacher_outputs.py \
    --model_path "runwayml/stable-diffusion-v1-5" \
    --train_data_dir "./dataset" \
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

### RTX 3080+ (высокопроизводительные настройки)

```bash
# Можно увеличить batch size
python3 train_student.py \
    --model_path "runwayml/stable-diffusion-v1-5" \
    --teacher_outputs_dir "./teacher_outputs" \
    --output_dir "./student_outputs" \
    --batch_size 2 \
    --optimizer_type "AdamW" \
    --lr_scheduler "cosine"
```

### GTX 1660/1060 (ограниченная память)

```bash
# Обязательно использовать FP16
python3 generate_teacher_outputs.py \
    --model_path "runwayml/stable-diffusion-v1-5" \
    --train_data_dir "./dataset" \
    --output_dir "./teacher_outputs" \
    --resolution 512 512 \
    --use_fp16

python3 train_student.py \
    --model_path "runwayml/stable-diffusion-v1-5" \
    --teacher_outputs_dir "./teacher_outputs" \
    --output_dir "./student_outputs" \
    --batch_size 1 \
    --optimizer_type "AdamW8bit" \
    --use_fp16
```

## Устранение проблем

### Ошибка "CUDA out of memory"

```bash
# Решения:
# 1. Уменьшить batch size
--batch_size 1

# 2. Включить FP16
--use_fp16

# 3. Использовать 8-bit оптимизатор
--optimizer_type "AdamW8bit"

# 4. Уменьшить разрешение
--resolution 512 512
```

### Ошибка "Module not found"

```bash
# Проверить виртуальное окружение
which python3
pip list

# Переустановить зависимости
pip install --force-reinstall -r requirements.txt
```

### Медленная загрузка моделей

```bash
# Использовать локальные модели вместо HuggingFace
--model_path "./local_model"  # вместо "runwayml/stable-diffusion-v1-5"

# Предварительно скачать модели
python3 -c "
from diffusers import StableDiffusionPipeline
pipeline = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')
pipeline.save_pretrained('./local_model')
"
```

## Оптимизация производительности

### 1. Использование xformers (если поддерживается)

```bash
pip install xformers
```

### 2. Настройка переменных окружения

```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### 3. Оптимизация PyTorch

```bash
# В начале Python скрипта
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

## Обновление

### Обновление модуля

```bash
cd sd_scripts
git pull origin main
cd teacher
pip install -r requirements.txt --upgrade
```

### Обновление зависимостей

```bash
pip install --upgrade torch torchvision torchaudio
pip install --upgrade diffusers transformers accelerate
```

## Поддержка

При возникновении проблем:

1. Проверьте системные требования
2. Убедитесь, что все зависимости установлены
3. Проверьте логи ошибок
4. Создайте issue в репозитории с описанием проблемы

## Полезные ссылки

- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [CUDA Installation](https://docs.nvidia.com/cuda/)