# Teacher-Student Training for Stable Diffusion

Этот модуль реализует процесс обучения типа teacher-student для Stable Diffusion моделей. Принцип работы заключается в том, что сначала teacher модель прогоняет датасет и сохраняет свои предсказания, а затем student модель учится повторять эти предсказания.

## Преимущества подхода

1. **Экономия памяти**: Teacher модель не нужно держать в памяти во время тренировки student
2. **Быстрая тренировка**: Student учится напрямую у teacher, а не с нуля
3. **Контролируемое обучение**: Student точно знает, что должен предсказать
4. **Масштабируемость**: Можно использовать разные teacher модели для разных задач

## Структура файлов

```
teacher/
├── generate_teacher_outputs.py    # Генерация teacher outputs
├── teacher_student_dataset.py     # Датасет для teacher-student обучения
├── train_student.py              # Тренировка student модели
└── README.md                     # Этот файл
```

## Процесс работы

### Шаг 1: Генерация Teacher Outputs

Сначала нужно пройти датасет через teacher модель и сохранить:
- `latents`: Закодированные изображения в латентном пространстве
- `t`: Таймстепы для диффузии
- `text_embeddings`: Выходы text encoder'а
- `eps_teacher`: Предсказания teacher модели

```bash
python teacher/generate_teacher_outputs.py \
    --model_path /path/to/teacher/model \
    --train_data_dir /path/to/training/images \
    --output_dir ./teacher_outputs \
    --resolution 512 512 \
    --use_fp16
```

**Параметры:**
- `--model_path`: Путь к teacher модели
- `--train_data_dir`: Папка с тренировочными изображениями
- `--output_dir`: Папка для сохранения teacher outputs
- `--resolution`: Разрешение изображений (ширина высота)
- `--use_fp16`: Использовать FP16 для экономии памяти
- `--vae_path`: Путь к VAE (опционально)
- `--caption_extension`: Расширение файлов с подписями (по умолчанию .txt)
- `--metadata_file`: Файл метаданных (опционально)

### Шаг 2: Тренировка Student Модели

После генерации teacher outputs можно тренировать student модель:

```bash
python teacher/train_student.py \
    --model_path /path/to/student/model \
    --teacher_outputs_dir ./teacher_outputs \
    --output_dir ./student_outputs \
    --max_train_steps 1000 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --resolution 512 512
```

**Параметры:**
- `--model_path`: Путь к student модели
- `--teacher_outputs_dir`: Папка с teacher outputs
- `--output_dir`: Папка для сохранения результатов тренировки
- `--max_train_steps`: Максимальное количество шагов тренировки
- `--batch_size`: Размер батча
- `--learning_rate`: Скорость обучения
- `--optimizer_type`: Тип оптимизатора (AdamW, AdamW8bit)
- `--lr_scheduler`: Планировщик learning rate (constant, linear, cosine)
- `--weight_decay`: Weight decay для оптимизатора
- `--max_grad_norm`: Максимальная норма градиента для clipping
- `--logging_steps`: Логирование каждые N шагов
- `--save_steps`: Сохранение чекпоинта каждые N шагов

## Требования к системе

- **GPU**: Минимум 12GB VRAM (RTX 3060 или лучше)
- **RAM**: Минимум 16GB
- **Python**: 3.8+
- **PyTorch**: 2.0+

## Установка зависимостей

```bash
pip install torch torchvision torchaudio
pip install diffusers transformers
pip install safetensors tqdm pillow numpy
pip install bitsandbytes  # для AdamW8bit оптимизатора
```

## Примеры использования

### Пример 1: Базовое teacher-student обучение

```bash
# 1. Генерируем teacher outputs
python teacher/generate_teacher_outputs.py \
    --model_path "runwayml/stable-diffusion-v1-5" \
    --train_data_dir "./my_dataset" \
    --output_dir "./teacher_outputs" \
    --resolution 512 512 \
    --use_fp16

# 2. Тренируем student
python teacher/train_student.py \
    --model_path "runwayml/stable-diffusion-v1-5" \
    --teacher_outputs_dir "./teacher_outputs" \
    --output_dir "./student_outputs" \
    --max_train_steps 2000 \
    --batch_size 1 \
    --learning_rate 5e-5
```

### Пример 2: Обучение с кастомными параметрами

```bash
# Генерация teacher outputs с кастомным VAE
python teacher/generate_teacher_outputs.py \
    --model_path "runwayml/stable-diffusion-v1-5" \
    --vae_path "./custom_vae" \
    --train_data_dir "./my_dataset" \
    --output_dir "./teacher_outputs" \
    --resolution 768 768 \
    --use_fp16

# Тренировка с продвинутыми настройками
python teacher/train_student.py \
    --model_path "runwayml/stable-diffusion-v1-5" \
    --teacher_outputs_dir "./teacher_outputs" \
    --output_dir "./student_outputs" \
    --max_train_steps 5000 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --optimizer_type "AdamW8bit" \
    --lr_scheduler "cosine" \
    --weight_decay 1e-5 \
    --max_grad_norm 0.5 \
    --logging_steps 5 \
    --save_steps 200
```

## Структура выходных данных

### Teacher Outputs

```
teacher_outputs/
├── metadata.json                    # Метаданные датасета
├── image_000000_latents.npy        # Латенты изображения
├── image_000000_t.npy             # Таймстеп
├── image_000000_text_embeddings.npy # Text embeddings
├── image_000000_eps_teacher.npy   # Teacher предсказания
├── image_000001_latents.npy
└── ...
```

### Student Outputs

```
student_outputs/
├── checkpoints/
│   ├── unet_step_00000100.safetensors
│   ├── training_state_step_00000100.json
│   ├── unet_step_00000200.safetensors
│   └── ...
└── training_log.txt
```

## Оптимизация для RTX 3060 12GB

Для карты с 12GB VRAM рекомендуется:

1. **Использовать FP16**: `--use_fp16`
2. **Batch size = 1**: `--batch_size 1`
3. **Использовать AdamW8bit**: `--optimizer_type "AdamW8bit"`
4. **Периодически очищать память**: Автоматически каждые 100 шагов

## Troubleshooting

### Ошибка "CUDA out of memory"

1. Уменьшите batch size до 1
2. Включите FP16: `--use_fp16`
3. Используйте AdamW8bit оптимизатор
4. Уменьшите разрешение изображений

### Ошибка "File not found"

1. Проверьте пути к файлам
2. Убедитесь, что teacher outputs были сгенерированы
3. Проверьте права доступа к папкам

### Медленная тренировка

1. Увеличьте batch size (если позволяет память)
2. Используйте более быстрый оптимизатор
3. Уменьшите количество логирований

## Дополнительные возможности

### Кастомные loss функции

Можно модифицировать `compute_loss` в `TeacherStudentTrainer` для использования других loss функций:

```python
def compute_loss(self, batch):
    # L1 loss вместо MSE
    loss = F.l1_loss(noise_pred, eps_teacher)
    
    # Или комбинация loss функций
    mse_loss = F.mse_loss(noise_pred, eps_teacher)
    l1_loss = F.l1_loss(noise_pred, eps_teacher)
    loss = 0.7 * mse_loss + 0.3 * l1_loss
    
    return loss
```

### Кастомные оптимизаторы

Добавьте новые оптимизаторы в `setup_training`:

```python
elif self.args.optimizer_type == "Lion":
    from lion_pytorch import Lion
    self.optimizer = Lion(
        self.unet.parameters(),
        lr=self.args.learning_rate,
        weight_decay=self.args.weight_decay
    )
```

## Лицензия

Этот код распространяется под той же лицензией, что и основной репозиторий sd_scripts.