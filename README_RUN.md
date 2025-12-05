# Инструкция по запуску лабораторной работы

## Проблема с драйверами Vulkan

Если программа не запускается с ошибкой "failed to create instance" или segfault, это означает проблему с драйверами Vulkan.

## Решение

### Вариант 1: Использовать скрипт запуска (рекомендуется)

```bash
cd /home/ivan/CG
./run.sh
```

Скрипт автоматически попробует различные драйверы.

### Вариант 2: Ручной запуск с указанием драйвера

**Для NVIDIA:**
```bash
cd /home/ivan/CG
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
./build/Lab1_3DGraphics
```

**Для Intel:**
```bash
cd /home/ivan/CG
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
./build/Lab1_3DGraphics
```

**Для software renderer (медленно, но должно работать):**
```bash
cd /home/ivan/CG
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
./build/Lab1_3DGraphics
```

### Вариант 3: Перезагрузка системы

После установки драйверов NVIDIA может потребоваться перезагрузка:

```bash
sudo reboot
```

После перезагрузки попробуйте запустить программу снова.

## Проверка драйверов

Проверьте, что Vulkan работает:

```bash
vulkaninfo --summary
```

Если команда выполняется без ошибок, драйверы установлены правильно.

## Если ничего не помогает

Установите mesa-vulkan-drivers (software renderer):

```bash
sudo apt-get install mesa-vulkan-drivers
```

Затем запустите с software renderer (см. Вариант 2).

