# SDXL Teacher-Student Training - Обзор

## 🎯 Что это такое?

Модуль **SDXL Teacher-Student Training** реализует процесс обучения типа teacher-student для SDXL моделей, основанный на методе дистилляции **DMD2** (https://arxiv.org/pdf/2405.14867).

## 🔄 Принцип работы

### 1. **Teacher Phase** (Генерация целей обучения)
- Датасет прогоняется через **teacher модель**
- Сохраняются предсчитанные outputs:
  - `latents` - латентные представления изображений
  - `timesteps` - временные шаги
  - `text_embeddings` - текстовые эмбеддинги
  - `teacher_noise_pred` - предсказания шума от teacher

### 2. **Student Phase** (Обучение)
- **Student модель** учится повторять teacher
- Вместо обучения с нуля - подгоняется под предсчитанные targets
- Результат: быстрая конвергенция и лучшее качество

## 📁 Структура модуля

```
teacher/
├── generate_teacher_outputs.py    # Генерация teacher outputs
├── train_student.py               # Обучение student модели
├── run_teacher_student.py         # Полный pipeline
├── quick_start.py                 # Быстрый старт
├── test_compatibility.py          # Тест совместимости
├── example_config.toml            # Пример конфигурации
├── README.md                      # Основная документация
├── INSTALL.md                     # Инструкции по установке
└── OVERVIEW.md                    # Этот файл
```

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

## 💡 Ключевые преимущества

- **🎓 Эффективность**: Student учится быстрее и лучше
- **💾 Память**: Teacher не держится в памяти во время обучения
- **🔧 Гибкость**: Поддержка всех network modules (LoRA, LoHa, LoKr, кастомные)
- **⚡ Оптимизация**: Оптимизировано для RTX 3060 12GB
- **📚 Совместимость**: Все аргументы оригинального sd_scripts

## 🎛️ Поддерживаемые функции

### Network Modules
- ✅ **LoRA** (`networks.lora`)
- ✅ **LoHa** (`networks.loha`) 
- ✅ **LoKr** (`networks.lokr`)
- ✅ **Кастомные сети** (любые Python модули)

### Оптимизация памяти
- ✅ **LowRAM режим** (`--lowram`)
- ✅ **XFormers** (`--xformers`)
- ✅ **SDPA** (`--sdpa`)
- ✅ **Mixed Precision** (`--mixed_precision fp16`)

### Все оригинальные аргументы
- ✅ **Аугментация**: `--color_aug`, `--flip_aug`, `--face_crop_aug_range`
- ✅ **Кэширование**: `--cache_latents`, `--cache_text_encoder_outputs`
- ✅ **Оптимизаторы**: `--optimizer_type`, `--learning_rate`, `--lr_scheduler_type`
- ✅ **Логирование**: `--log_with`, `--logging_dir`

## 🎯 Оптимизация для RTX 3060 12GB

### Рекомендуемые настройки
```bash
--train_batch_size 1          # Минимальный батч
--mixed_precision fp16        # Смешанная точность
--lowram                      # Режим низкой памяти
--xformers                    # Memory efficient attention
--gradient_accumulation_steps 4  # Эффективный батч-сайз
```

## 📊 Примеры использования

### Простой LoRA
```bash
# 1. Генерируем teacher outputs
python teacher/generate_teacher_outputs.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --train_data_dir "./my_dataset" \
    --teacher_output_dir "./teacher_outputs" \
    --lowram --xformers

# 2. Обучаем LoRA
python teacher/train_student.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --train_data_dir "./teacher_outputs" \
    --network_module "networks.lora" \
    --network_dim 64 --network_alpha 32 \
    --lowram --xformers
```

### Кастомная сеть
```bash
python teacher/train_student.py \
    --pretrained_model_name_or_path "path/to/base/model" \
    --train_data_dir "./teacher_outputs" \
    --network_module "path/to/custom/network.py" \
    --network_args "arg1" "arg2"
```

## 🔧 Технические детали

### Формат Teacher Outputs
Каждый sample сохраняется как `.pt` файл с ключами:
- `latents`: Тензор [B, 4, H, W]
- `noisy_latents`: Тензор [B, 4, H, W] 
- `timesteps`: Тензор [B]
- `text_embeddings1/2`: Тензоры [B, L, D]
- `pool2`: Тензор [B, D]
- `teacher_noise_pred`: Тензор [B, 4, H, W]
- `original_size`, `crop_top_left`, `target_size`: Метаданные

### Loss Function
По умолчанию используется **L2 loss** между student и teacher predictions:
```python
loss = F.mse_loss(student_pred, teacher_pred)
```

Поддерживаются: `l1`, `l2`, `huber`, `smooth_l1`

## 📈 Мониторинг

### TensorBoard
```bash
tensorboard --logdir ./logs
```

### Weights & Biases
```bash
--log_with wandb --wandb_run_name "my_experiment"
```

### Консольные логи
```bash
--console_log_level INFO --console_log_file "./training.log"
```

## 🆘 Устранение неполадок

### Ошибки памяти
```bash
--train_batch_size 1 --lowram --xformers --resolution 512
```

### Медленное обучение
```bash
--mixed_precision fp16 --cache_text_encoder_outputs --gradient_accumulation_steps 4
```

### Проблемы импорта
```bash
# Убедитесь, что вы в корне sd_scripts
pwd
python teacher/test_compatibility.py
```

## 📚 Документация

- **[README.md](README.md)** - Полная документация
- **[INSTALL.md](INSTALL.md)** - Установка и настройка
- **[example_config.toml](example_config.toml)** - Примеры конфигурации

## 🌟 Особенности реализации

### Memory Management
- Teacher модель загружается только для генерации outputs
- Student обучение происходит без teacher в памяти
- Автоматическое управление памятью GPU/CPU

### Dataset Handling
- Поддержка всех форматов sd_scripts
- Автоматическое кэширование text encoder outputs
- Bucket resolution для оптимизации

### Training Loop
- Gradient accumulation для эффективного батч-сайза
- Автоматическое сохранение чекпоинтов
- Поддержка всех оптимизаторов и scheduler'ов

## 🔮 Будущие улучшения

- [ ] Поддержка SD 1.5/2.1
- [ ] Интеграция с ComfyUI
- [ ] Web UI для настройки
- [ ] Автоматическая оптимизация гиперпараметров
- [ ] Поддержка multi-GPU обучения

## 📞 Поддержка

- **GitHub Issues**: Баги и feature requests
- **Documentation**: Подробные инструкции
- **Examples**: Готовые примеры использования

---

**🎓 Teacher-Student Training** - революционный подход к обучению SDXL моделей!