# Установка LyCORIS

## Что такое LyCORIS?

LyCORIS (Low-Rank Adaptation with Orthogonal Constraint and Improved Stability) - это расширенная библиотека для обучения адаптивных сетей в Stable Diffusion. Она предоставляет более гибкие и эффективные методы адаптации по сравнению со стандартным LoRA.

## Возможности LyCORIS

- **LoHa**: Low-Rank Adaptation with Hadamard Product
- **LoKr**: Low-Rank Adaptation with Kronecker Product
- **LoCon**: Low-Rank Adaptation with Convolution
- **IA³**: Intrinsic Dimension Adaptation
- **DyLoRA**: Dynamic LoRA
- **GLoRA**: Generalized LoRA

## Установка

### Способ 1: Через pip (рекомендуется)

```bash
pip install lycoris
```

### Способ 2: Из исходного кода

```bash
git clone https://github.com/KohakuBlueleaf/LyCORIS.git
cd LyCORIS
pip install -e .
```

### Способ 3: Для Windows

```bash
pip install lycoris --index-url https://download.pytorch.org/whl/cu118
```

## Проверка установки

```python
import lycoris
print(f"LyCORIS version: {lycoris.__version__}")
```

## Использование в teacher-student обучении

### Базовое использование

```bash
python train_network_teacher_student.py \
    --network_module lycoris.kohya \
    --network_dim 64 \
    --network_alpha 64
```

### Продвинутые настройки

```bash
python train_network_teacher_student.py \
    --network_module lycoris.kohya \
    --network_dim 64 \
    --network_alpha 64 \
    --network_args "algo=loha,conv_dim=16,conv_alpha=8"
```

## Типы алгоритмов

### LoHa (Low-Rank Adaptation with Hadamard Product)
```bash
--network_args "algo=loha,conv_dim=16,conv_alpha=8"
```

### LoKr (Low-Rank Adaptation with Kronecker Product)
```bash
--network_args "algo=lokr,conv_dim=16,conv_alpha=8"
```

### LoCon (Low-Rank Adaptation with Convolution)
```bash
--network_args "algo=locon,conv_dim=16,conv_alpha=8"
```

### IA³ (Intrinsic Dimension Adaptation)
```bash
--network_args "algo=ia3,conv_dim=16,conv_alpha=8"
```

### DyLoRA (Dynamic LoRA)
```bash
--network_args "algo=dylora,conv_dim=16,conv_alpha=8"
```

### GLoRA (Generalized LoRA)
```bash
--network_args "algo=glora,conv_dim=16,conv_alpha=8"
```

## Примеры конфигураций

### SD + LoHa
```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "runwayml/stable-diffusion-v1-5" \
    --base_model "runwayml/stable-diffusion-v1-5" \
    --output_dir ./trained_loha \
    --network_module lycoris.kohya \
    --network_dim 32 \
    --network_alpha 32 \
    --network_args "algo=loha,conv_dim=16,conv_alpha=8"
```

### SDXL + LoKr
```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --base_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir ./trained_lokr \
    --network_module lycoris.kohya \
    --network_dim 64 \
    --network_alpha 64 \
    --network_args "algo=lokr,conv_dim=32,conv_alpha=16" \
    --max_resolution "1024,1024"
```

### SDXL + LoCon с dropout
```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --base_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir ./trained_locon \
    --network_module lycoris.kohya \
    --network_dim 64 \
    --network_alpha 64 \
    --network_dropout 0.1 \
    --network_args "algo=locon,conv_dim=32,conv_alpha=16" \
    --max_resolution "1024,1024"
```

## Параметры network_args

### Общие параметры
- `algo`: Тип алгоритма (loha, lokr, locon, ia3, dylora, glora)
- `conv_dim`: Размерность для сверточных слоев
- `conv_alpha`: Альфа параметр для сверточных слоев
- `dropout`: Dropout для регуляризации
- `use_scale_shift_norm`: Использовать scale-shift normalization

### Специфичные для алгоритмов
- **LoHa**: `use_wscale`, `use_conv`
- **LoKr**: `use_wscale`, `use_conv`
- **LoCon**: `use_wscale`, `use_conv`
- **IA³**: `use_wscale`, `use_conv`
- **DyLoRA**: `use_wscale`, `use_conv`
- **GLoRA**: `use_wscale`, `use_conv`

## Оптимизация для RTX 3060 12GB

### SD + LoHa
```bash
--network_dim 16
--network_alpha 16
--network_args "algo=loha,conv_dim=8,conv_alpha=4"
--train_batch_size 1
--gradient_accumulation_steps 8
```

### SDXL + LoKr
```bash
--network_dim 32
--network_alpha 32
--network_args "algo=lokr,conv_dim=16,conv_alpha=8"
--train_batch_size 1
--gradient_accumulation_steps 16
--max_resolution "1024,1024"
```

## Продолжение обучения

### С существующими весами LyCORIS
```bash
python train_network_teacher_student.py \
    --network_weights ./existing_lycoris.safetensors \
    --network_module lycoris.kohya \
    --network_dim 64 \
    --network_alpha 64
```

## Troubleshooting

### Ошибка "LyCORIS not installed"
```bash
pip install lycoris
```

### Ошибка совместимости версий
```bash
pip install lycoris==1.8.0
```

### Проблемы с CUDA
```bash
pip install lycoris --index-url https://download.pytorch.org/whl/cu118
```

### Проблемы с зависимостями
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install lycoris
```

## Сравнение с LoRA

| Характеристика | LoRA | LyCORIS |
|----------------|------|---------|
| Скорость обучения | Быстро | Быстро |
| Качество | Хорошо | Лучше |
| Гибкость | Ограниченная | Высокая |
| Память | Экономичная | Экономичная |
| Поддерживаемые алгоритмы | 1 | 6+ |

## Рекомендации

1. **Для начала**: Используйте LoHa или LoKr
2. **Для качества**: Попробуйте LoCon или GLoRA
3. **Для экономии памяти**: Используйте IA³
4. **Для экспериментов**: DyLoRA с динамическими размерами

## Дополнительные ресурсы

- [Официальный репозиторий LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)
- [Документация LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS/wiki)
- [Примеры использования](https://github.com/KohakuBlueleaf/LyCORIS/tree/main/examples)