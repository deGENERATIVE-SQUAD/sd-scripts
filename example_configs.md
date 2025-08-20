# Примеры конфигураций

## SD + LoRA (базовый)

```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "runwayml/stable-diffusion-v1-5" \
    --base_model "runwayml/stable-diffusion-v1-5" \
    --output_dir ./trained_lora \
    --network_module networks.lora \
    --network_dim 32 \
    --network_alpha 32 \
    --train_batch_size 1 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 4
```

## SDXL + LoRA

```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --base_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir ./trained_sdxl_lora \
    --network_module networks.lora \
    --network_dim 64 \
    --network_alpha 64 \
    --train_batch_size 1 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 8 \
    --max_resolution "1024,1024" \
    --min_bucket_reso 512 \
    --max_bucket_reso 2048
```

## SDXL + LyCORIS

```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --base_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir ./trained_sdxl_lycoris \
    --network_module lycoris.kohya \
    --network_dim 64 \
    --network_alpha 64 \
    --train_batch_size 1 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 8 \
    --max_resolution "1024,1024" \
    --min_bucket_reso 512 \
    --max_bucket_reso 2048
```

## Смешанные модели (teacher SDXL, student SD)

```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --base_model "runwayml/stable-diffusion-v1-5" \
    --output_dir ./trained_mixed \
    --network_module networks.lora \
    --network_dim 32 \
    --network_alpha 32 \
    --train_batch_size 1 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 4
```

## Оптимизация для RTX 3060 12GB

### SD модель
```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "runwayml/stable-diffusion-v1-5" \
    --base_model "runwayml/stable-diffusion-v1-5" \
    --output_dir ./trained_lora \
    --network_module networks.lora \
    --network_dim 16 \
    --network_alpha 16 \
    --train_batch_size 1 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 8 \
    --max_resolution "512,512"
```

### SDXL модель
```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --base_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir ./trained_sdxl \
    --network_module networks.lora \
    --network_dim 32 \
    --network_alpha 32 \
    --train_batch_size 1 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 16 \
    --max_resolution "1024,1024" \
    --min_bucket_reso 512 \
    --max_bucket_reso 2048
```

## Продолжение обучения

### С существующими весами LoRA
```bash
python train_network_teacher_student.py \
    --train_data_dir ./dataset \
    --teacher_outputs_dir ./teacher_outputs \
    --output_dir ./continued_training \
    --model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --network_module networks.lora \
    --network_dim 32 \
    --network_alpha 32 \
    --network_weights ./existing_lora.safetensors \
    --train_batch_size 1 \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 4
```

### С существующими весами LyCORIS
```bash
python train_network_teacher_student.py \
    --train_data_dir ./dataset \
    --teacher_outputs_dir ./teacher_outputs \
    --output_dir ./continued_training \
    --model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --network_module lycoris.kohya \
    --network_dim 64 \
    --network_alpha 64 \
    --network_weights ./existing_lycoris.safetensors \
    --train_batch_size 1 \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 8 \
    --max_resolution "1024,1024"
```

## Только подготовка teacher outputs

```bash
python run_teacher_student_training.py \
    --skip_training \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --base_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir ./output \
    --max_resolution "1024,1024" \
    --min_bucket_reso 512 \
    --max_bucket_reso 2048
```

## Только обучение (используя существующие teacher outputs)

```bash
python run_teacher_student_training.py \
    --skip_teacher_prep \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --base_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir ./trained_model \
    --teacher_outputs_dir ./existing_teacher_outputs \
    --network_module lycoris.kohya \
    --network_dim 64 \
    --network_alpha 64
```

## Продвинутые настройки

### С dropout и регуляризацией
```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --base_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir ./trained_advanced \
    --network_module lycoris.kohya \
    --network_dim 64 \
    --network_alpha 64 \
    --network_dropout 0.1 \
    --train_batch_size 1 \
    --num_train_epochs 15 \
    --learning_rate 1e-4 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 8 \
    --max_resolution "1024,1024" \
    --max_grad_norm 1.0
```

### С различными learning rates для компонентов
```bash
python run_teacher_student_training.py \
    --train_data_dir ./dataset \
    --in_json ./dataset/metadata.json \
    --teacher_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --base_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir ./trained_lr_tuned \
    --network_module lycoris.kohya \
    --network_dim 64 \
    --network_alpha 64 \
    --train_batch_size 1 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --text_encoder_lr_scale 0.5 \
    --unet_lr_scale 1.0 \
    --vae_lr_scale 0.1 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 8 \
    --max_resolution "1024,1024"
```

## Конфигурационные файлы

### SD + LoRA (config_sd_lora.toml)
```toml
[training]
train_data_dir = "./dataset"
in_json = "./dataset/metadata.json"
teacher_model = "runwayml/stable-diffusion-v1-5"
base_model = "runwayml/stable-diffusion-v1-5"
output_dir = "./trained_lora"
network_module = "networks.lora"
network_dim = 32
network_alpha = 32
train_batch_size = 1
num_train_epochs = 10
learning_rate = 1e-4
mixed_precision = "fp16"
gradient_accumulation_steps = 4
max_resolution = "512,512"
min_bucket_reso = 256
max_bucket_reso = 1024
```

### SDXL + LyCORIS (config_sdxl_lycoris.toml)
```toml
[training]
train_data_dir = "./dataset"
in_json = "./dataset/metadata.json"
teacher_model = "stabilityai/stable-diffusion-xl-base-1.0"
base_model = "stabilityai/stable-diffusion-xl-base-1.0"
output_dir = "./trained_sdxl_lycoris"
network_module = "lycoris.kohya"
network_dim = 64
network_alpha = 64
train_batch_size = 1
num_train_epochs = 10
learning_rate = 1e-4
mixed_precision = "fp16"
gradient_accumulation_steps = 8
max_resolution = "1024,1024"
min_bucket_reso = 512
max_bucket_reso = 2048
network_dropout = 0.1
text_encoder_lr_scale = 0.5
unet_lr_scale = 1.0
vae_lr_scale = 0.1
```

## Рекомендации по параметрам

### Размеры сети
- **SD + LoRA**: 16-32 (для экономии памяти), 64-128 (для качества)
- **SDXL + LoRA**: 32-64 (для экономии памяти), 128-256 (для качества)
- **SDXL + LyCORIS**: 64-128 (для экономии памяти), 256-512 (для качества)

### Learning rates
- **SD**: 1e-4 (стандарт), 5e-5 (продолжение обучения)
- **SDXL**: 1e-4 (стандарт), 5e-5 (продолжение обучения), 1e-5 (тонкая настройка)

### Разрешения
- **SD**: 512x512 (стандарт), 768x768 (высокое качество)
- **SDXL**: 1024x1024 (стандарт), 1280x1280 (высокое качество)

### Gradient accumulation
- **SD**: 4-8 шагов
- **SDXL**: 8-16 шагов (для экономии памяти)