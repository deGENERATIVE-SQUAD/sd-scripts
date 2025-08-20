# 🚀 Teacher-Student Training System - Обзор

## ✨ Что это такое?

Полнофункциональная система для teacher-student обучения LoRA/LyCORIS моделей в sd_scripts, которая позволяет:

- **Экономить VRAM**: Teacher модель не держится в памяти во время обучения
- **Ускорить обучение**: Student учится напрямую у teacher
- **Поддерживать SD и SDXL**: Автоматическое определение типа модели
- **Использовать LyCORIS**: Расширенные алгоритмы адаптации

## 🎯 Основные возможности

### ✅ Поддерживаемые модели
- **SD 1.x/2.x**: Стандартные Stable Diffusion модели
- **SDXL**: Stable Diffusion XL модели
- **Смешанные**: Teacher SDXL + Student SD (и наоборот)

### ✅ Поддерживаемые сети
- **LoRA**: Стандартный Low-Rank Adaptation
- **LyCORIS**: Расширенные алгоритмы (LoHa, LoKr, LoCon, IA³, DyLoRA, GLoRA)

### ✅ Автоматизация
- Автоматическое определение типа модели
- Автоматическая настройка параметров
- Полный pipeline от подготовки до обучения

## 📁 Структура проекта

```
├── finetune/
│   └── prepare_teacher_outputs.py    # Подготовка teacher outputs
├── library/
│   └── teacher_student_dataset.py    # Dataset для teacher-student
├── train_network_teacher_student.py  # Основной trainer
├── run_teacher_student_training.py   # Автоматизированный pipeline
├── test_teacher_student.py           # Тестирование системы
├── README_teacher_student.md         # Подробная документация
├── QUICKSTART.md                     # Быстрый старт
├── example_configs.md                # Примеры конфигураций
├── INSTALL_LYCORIS.md                # Установка LyCORIS
└── OVERVIEW.md                       # Этот файл
```

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install torch torchvision diffusers transformers accelerate
pip install lycoris  # Для LyCORIS поддержки
```

### 2. Запуск полного pipeline
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

## 💡 Примеры использования

### SD + LoRA
```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --teacher_model "runwayml/stable-diffusion-v1-5" \
    --base_model "runwayml/stable-diffusion-v1-5" \
    --output_dir ./trained_lora \
    --network_module networks.lora \
    --network_dim 32
```

### SDXL + LyCORIS
```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --teacher_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --base_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir ./trained_lycoris \
    --network_module lycoris.kohya \
    --network_dim 64 \
    --network_args "algo=loha,conv_dim=16,conv_alpha=8"
```

### Смешанные модели
```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --teacher_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --base_model "runwayml/stable-diffusion-v1-5" \
    --output_dir ./trained_mixed \
    --network_module networks.lora \
    --network_dim 32
```

## 🔧 Оптимизация для RTX 3060 12GB

### SD модели
```bash
--train_batch_size 1
--mixed_precision fp16
--gradient_accumulation_steps 4
--max_resolution "512,512"
--network_dim 16-32
```

### SDXL модели
```bash
--train_batch_size 1
--mixed_precision fp16
--gradient_accumulation_steps 8-16
--max_resolution "1024,1024"
--network_dim 32-64
--gradient_checkpointing
```

## 📊 Сравнение методов

| Метод | Скорость | Качество | Память | Гибкость |
|-------|----------|----------|---------|----------|
| **LoRA** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **LyCORIS LoHa** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **LyCORIS LoKr** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **LyCORIS LoCon** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🎨 Поддерживаемые алгоритмы LyCORIS

- **LoHa**: Hadamard Product - быстрое обучение
- **LoKr**: Kronecker Product - высокое качество
- **LoCon**: Convolution - для изображений
- **IA³**: Intrinsic Dimension - экономия памяти
- **DyLoRA**: Dynamic LoRA - адаптивные размеры
- **GLoRA**: Generalized LoRA - универсальный

## 📈 Workflow

```
1. Подготовка данных
   ↓
2. Запуск teacher модели
   ↓
3. Сохранение outputs (.npz)
   ↓
4. Обучение student модели
   ↓
5. Готовая LoRA/LyCORIS
```

## 🧪 Тестирование

```bash
python test_teacher_student.py
```

Проверяет:
- ✅ Создание teacher outputs
- ✅ Загрузку dataset (SD & SDXL)
- ✅ Создание сетей (LoRA & LyCORIS)
- ✅ Определение типа модели

## 📚 Документация

- **[README_teacher_student.md](README_teacher_student.md)**: Полная документация
- **[QUICKSTART.md](QUICKSTART.md)**: Быстрый старт
- **[example_configs.md](example_configs.md)**: Примеры конфигураций
- **[INSTALL_LYCORIS.md](INSTALL_LYCORIS.md)**: Установка LyCORIS

## 🆘 Troubleshooting

### Частые проблемы
1. **CUDA out of memory**: Уменьшите batch size, увеличьте gradient accumulation
2. **LyCORIS not installed**: `pip install lycoris`
3. **Teacher output not found**: Проверьте пути и запустите prepare_teacher_outputs.py

### Оптимизация памяти
- Используйте `--mixed_precision fp16`
- Уменьшите `--network_dim`
- Увеличьте `--gradient_accumulation_steps`
- Для SDXL: используйте `--gradient_checkpointing`

## 🌟 Преимущества системы

1. **Экономия VRAM**: Teacher модель не держится в памяти
2. **Быстрое обучение**: Student учится напрямую у teacher
3. **Гибкость**: Поддержка SD/SDXL + LoRA/LyCORIS
4. **Автоматизация**: Полный pipeline в одном скрипте
5. **Оптимизация**: Настроено для RTX 3060 12GB
6. **Масштабируемость**: Можно использовать мощные teacher модели

## 🚀 Следующие шаги

1. **Установите зависимости**: `pip install lycoris`
2. **Подготовьте датасет**: Создайте папку с изображениями и metadata.json
3. **Запустите pipeline**: Используйте `run_teacher_student_training.py`
4. **Экспериментируйте**: Попробуйте разные алгоритмы LyCORIS
5. **Оптимизируйте**: Настройте параметры под вашу GPU

## 📞 Поддержка

При возникновении проблем:
1. Проверьте логи на наличие ошибок
2. Убедитесь, что все зависимости установлены
3. Проверьте соответствие версий PyTorch и Diffusers
4. Создайте issue в репозитории проекта

---

**🎯 Готово к использованию!** Система полностью функциональна и готова для teacher-student обучения LoRA/LyCORIS моделей.