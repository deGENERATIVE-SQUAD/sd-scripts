# SDXL Teacher-Student Training - Тестирование

## 🧪 Обзор тестирования

Этот документ описывает, как протестировать модуль teacher-student training и убедиться, что он работает корректно.

## 🚀 Быстрое тестирование

### 1. Тест совместимости
```bash
python teacher/test_compatibility.py
```

Этот скрипт проверяет:
- ✅ Импорт всех необходимых модулей
- ✅ Доступность CUDA
- ✅ Загрузку SDXL моделей
- ✅ Доступность strategy модулей
- ✅ Импорт teacher-student классов
- ✅ Функции оптимизации памяти

### 2. Демонстрация возможностей
```bash
python teacher/demo.py
```

Этот скрипт показывает:
- 📦 Импорт основных классов
- 🔧 Создание экземпляров классов
- 📊 Функциональность датасета
- 🌟 Ключевые возможности
- ⚡ Советы по оптимизации
- 🚀 Следующие шаги

## 🔍 Детальное тестирование

### Тест 1: Проверка импортов

```bash
cd /path/to/sd-scripts
python -c "
from teacher.generate_teacher_outputs import TeacherOutputGenerator
from teacher.train_student import StudentTrainer, TeacherStudentDataset
print('✓ All imports successful')
"
```

**Ожидаемый результат**: `✓ All imports successful`

### Тест 2: Проверка создания классов

```bash
python -c "
from teacher.generate_teacher_outputs import TeacherOutputGenerator
from teacher.train_student import StudentTrainer

teacher_gen = TeacherOutputGenerator()
student_trainer = StudentTrainer()

print(f'Teacher VAE scale: {teacher_gen.vae_scale_factor}')
print(f'Student VAE scale: {student_trainer.vae_scale_factor}')
print(f'Teacher is SDXL: {teacher_gen.is_sdxl}')
print(f'Student is SDXL: {student_trainer.is_sdxl}')
"
```

**Ожидаемый результат**:
```
Teacher VAE scale: 0.13025
Student VAE scale: 0.13025
Teacher is SDXL: True
Student is SDXL: True
```

### Тест 3: Проверка аргументов

```bash
python teacher/generate_teacher_outputs.py --help
python teacher/train_student.py --help
python teacher/run_teacher_student.py --help
```

**Ожидаемый результат**: Отображение справки по аргументам командной строки

## 🧪 Интеграционные тесты

### Тест 4: Проверка совместимости с sd_scripts

```bash
# Убедитесь, что вы в корне sd_scripts
pwd

# Проверьте наличие основных файлов
ls -la sdxl_train_network.py
ls -la library/sdxl_train_util.py
ls -la library/sdxl_model_util.py

# Проверьте импорт основных модулей
python -c "
import library.sdxl_train_util
import library.sdxl_model_util
import library.train_util
print('✓ SD Scripts modules imported successfully')
"
```

### Тест 5: Проверка стратегий

```bash
python -c "
from library.strategy_sdxl import SdxlTokenizeStrategy
from library.strategy_sd import SdSdxlLatentsCachingStrategy

print('✓ Strategy modules available')
"
```

## 🔧 Тестирование функциональности

### Тест 6: Проверка парсера аргументов

```bash
python -c "
from teacher.generate_teacher_outputs import setup_parser
from teacher.train_student import setup_parser as setup_parser_student

parser1 = setup_parser()
parser2 = setup_parser_student()

print(f'Teacher parser args: {len(parser1._actions)}')
print(f'Student parser args: {len(parser2._actions)}')
print('✓ Argument parsers created successfully')
"
```

### Тест 7: Проверка конфигурации

```bash
# Проверьте пример конфигурации
python -c "
import toml
config = toml.load('teacher/example_config.toml')
print('✓ Configuration file loaded successfully')
print(f'Sections: {list(config.keys())}')
"
```

## 🐛 Тестирование обработки ошибок

### Тест 8: Проверка отсутствующих файлов

```bash
# Попробуйте запустить с несуществующим датасетом
python teacher/generate_teacher_outputs.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --train_data_dir "./nonexistent_dataset" \
    --teacher_output_dir "./test_outputs" \
    2>&1 | grep -i "error\|not found"
```

**Ожидаемый результат**: Сообщение об ошибке о том, что датасет не найден

### Тест 9: Проверка неверных аргументов

```bash
# Попробуйте запустить с неверными аргументами
python teacher/train_student.py \
    --pretrained_model_name_or_path "invalid_model" \
    --train_data_dir "./nonexistent" \
    2>&1 | grep -i "error\|not found"
```

## 📊 Тестирование производительности

### Тест 10: Проверка использования памяти

```bash
# Установите nvidia-ml-py для мониторинга GPU
pip install nvidia-ml-py

# Создайте простой тест памяти
python -c "
import torch
import pynvml

if torch.cuda.is_available():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f'GPU Memory: {info.total / 1024**3:.1f} GB total')
    print(f'GPU Memory: {info.free / 1024**3:.1f} GB free')
    print('✓ GPU memory monitoring available')
else:
    print('⚠ CUDA not available')
"
```

## 🎯 Тестирование конкретных сценариев

### Тест 11: LoRA конфигурация

```bash
python -c "
from teacher.train_student import setup_parser
parser = setup_parser()

# Проверьте LoRA аргументы
lora_args = [action.dest for action in parser._actions if 'network' in action.dest]
print(f'LoRA arguments: {lora_args}')

# Проверьте наличие обязательных аргументов
required_args = ['network_module', 'network_dim', 'network_alpha']
for arg in required_args:
    if arg in lora_args:
        print(f'✓ {arg} available')
    else:
        print(f'✗ {arg} missing')
"
```

### Тест 12: Оптимизация памяти

```bash
python -c "
from teacher.generate_teacher_outputs import setup_parser
parser = setup_parser()

# Проверьте аргументы оптимизации памяти
memory_args = [action.dest for action in parser._actions if any(x in action.dest for x in ['lowram', 'highvram', 'xformers', 'sdpa'])]
print(f'Memory optimization arguments: {memory_args}')

for arg in ['lowram', 'xformers']:
    if arg in memory_args:
        print(f'✓ {arg} available')
    else:
        print(f'✗ {arg} missing')
"
```

## 🔍 Тестирование совместимости

### Тест 13: Проверка версий

```bash
python -c "
import torch
import diffusers
import accelerate

print(f'PyTorch: {torch.__version__}')
print(f'Diffusers: {diffusers.__version__}')
print(f'Accelerate: {accelerate.__version__}')

# Проверьте минимальные версии
torch_version = torch.__version__.split('+')[0]
major, minor = map(int, torch_version.split('.')[:2])

if major >= 2:
    print('✓ PyTorch 2.0+ available')
else:
    print('✗ PyTorch 2.0+ required')
"
```

### Тест 14: Проверка CUDA

```bash
python -c "
import torch

if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    
    # Проверьте совместимость версий
    if torch.version.cuda >= '11.8':
        print('✓ CUDA 11.8+ available')
    else:
        print('⚠ CUDA 11.8+ recommended')
else:
    print('✗ CUDA not available')
"
```

## 📝 Создание тестовых данных

### Тест 15: Создание минимального датасета

```bash
# Создайте тестовый датасет
mkdir -p test_dataset
echo "A beautiful landscape" > test_dataset/image1.caption
echo "A cute cat" > test_dataset/image2.caption

# Создайте пустые изображения (для тестирования структуры)
convert -size 512x512 xc:white test_dataset/image1.jpg
convert -size 512x512 xc:black test_dataset/image2.jpg

echo "✓ Test dataset created"
ls -la test_dataset/
```

## 🚨 Устранение проблем тестирования

### Проблема: Импорт не работает

**Решение**:
```bash
# Убедитесь, что вы в корне sd_scripts
pwd

# Добавьте путь в PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Попробуйте импорт снова
python -c "from teacher.generate_teacher_outputs import TeacherOutputGenerator"
```

### Проблема: Модули не найдены

**Решение**:
```bash
# Проверьте установку зависимостей
pip list | grep -E "(torch|diffusers|accelerate)"

# Установите недостающие
pip install torch diffusers accelerate
```

### Проблема: CUDA недоступна

**Решение**:
```bash
# Проверьте установку CUDA
nvidia-smi

# Проверьте PyTorch с CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Переустановите PyTorch с CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Результаты тестирования

После выполнения всех тестов вы должны увидеть:

```
🧪 TESTING SUMMARY
==================
✓ Import tests: PASSED
✓ Class creation: PASSED
✓ Argument parsing: PASSED
✓ Configuration loading: PASSED
✓ Strategy modules: PASSED
✓ Memory optimization: PASSED
✓ CUDA compatibility: PASSED
✓ Error handling: PASSED

🎉 All tests passed! Module is ready for use.
```

## 🎯 Следующие шаги после тестирования

1. **Запустите демо**: `python teacher/demo.py`
2. **Попробуйте quick start**: `python teacher/quick_start.py`
3. **Подготовьте датасет** для реального обучения
4. **Запустите обучение** с небольшим датасетом
5. **Мониторьте результаты** с TensorBoard

---

**🧪 Тестирование** - ключ к успешному использованию модуля!