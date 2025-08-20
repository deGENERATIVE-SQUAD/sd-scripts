# 🚀 Quick Start Guide

## Быстрый старт для Teacher-Student LoRA обучения

### 1. Подготовка окружения

Убедитесь, что у вас установлены все зависимости:

```bash
pip install torch torchvision diffusers transformers accelerate
```

### 2. Структура проекта

```
your_project/
├── dataset/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── metadata.json
├── finetune/
│   └── prepare_teacher_outputs.py
├── library/
│   └── teacher_student_dataset.py
├── train_network_teacher_student.py
└── run_teacher_student_training.py
```

### 3. Быстрый запуск (все в одном)

```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "runwayml/stable-diffusion-v1-5" \
    --base_model "runwayml/stable-diffusion-v1-5" \
    --output_dir ./trained_lora \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --network_dim 32 \
    --network_alpha 32
```

### 4. Пошаговый запуск

#### Шаг 1: Подготовка teacher outputs

```bash
python finetune/prepare_teacher_outputs.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --output_dir ./teacher_outputs \
    --mixed_precision fp16
```

#### Шаг 2: Обучение student LoRA

```bash
python train_network_teacher_student.py \
    --train_data_dir ./dataset \
    --teacher_outputs_dir ./teacher_outputs \
    --output_dir ./trained_lora \
    --model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --train_batch_size 1 \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --network_dim 32 \
    --network_alpha 32 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 4
```

### 5. Оптимизация для RTX 3060 12GB

```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "runwayml/stable-diffusion-v1-5" \
    --base_model "runwayml/stable-diffusion-v1-5" \
    --output_dir ./trained_lora \
    --train_batch_size 1 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --network_dim 16 \
    --network_alpha 16 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 8
```

### 6. Тестирование системы

```bash
python test_teacher_student.py
```

### 7. Пример metadata.json

```json
{
    "image1.jpg": {
        "caption": "A beautiful landscape with mountains",
        "resolution": [512, 512]
    },
    "image2.jpg": {
        "caption": "Portrait of a person",
        "resolution": [512, 512]
    }
}
```

### 8. Проверка результатов

После завершения обучения проверьте:

- `./teacher_outputs/` - файлы .npz с teacher outputs
- `./trained_lora/` - обученная LoRA модель
- Логи обучения в консоли

### 9. Использование обученной LoRA

```python
from diffusers import StableDiffusionPipeline
import torch

# Загрузите базовую модель
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# Загрузите обученную LoRA
pipe.load_lora_weights("./trained_lora")

# Генерация изображения
image = pipe("Your prompt here").images[0]
image.save("generated_image.png")
```

### 10. Troubleshooting

#### Ошибка CUDA out of memory:
- Уменьшите `--train_batch_size` до 1
- Увеличьте `--gradient_accumulation_steps`
- Используйте `--mixed_precision fp16`
- Уменьшите `--network_dim`

#### Медленное обучение:
- Увеличьте `--gradient_accumulation_steps`
- Уменьшите `--max_data_loader_n_workers`
- Используйте SSD для хранения данных

### 11. Полезные команды

#### Только подготовка teacher outputs:
```bash
python run_teacher_student_training.py \
    --skip_training \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "runwayml/stable-diffusion-v1-5" \
    --base_model "runwayml/stable-diffusion-v1-5" \
    --output_dir ./output
```

#### Только обучение (используя существующие teacher outputs):
```bash
python run_teacher_student_training.py \
    --skip_teacher_prep \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "runwayml/stable-diffusion-v1-5" \
    --base_model "runwayml/stable-diffusion-v1-5" \
    --output_dir ./trained_lora \
    --teacher_outputs_dir ./existing_teacher_outputs
```

### 12. Мониторинг обучения

Во время обучения вы увидите:
- Прогресс по эпохам
- Loss значения
- Скорость обучения
- Время выполнения

### 13. Сохранение и восстановление

Модель автоматически сохраняется каждые 1000 шагов. Для восстановления с определенного шага используйте соответствующие аргументы в скрипте обучения.

---

**🎯 Готово!** Теперь у вас есть полностью функциональная система teacher-student обучения для LoRA, оптимизированная для RTX 3060 12GB.