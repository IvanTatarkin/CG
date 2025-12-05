#!/bin/bash

# Скрипт для запуска лабораторной работы
# Исправляет проблемы с путями к драйверам Vulkan

# Убираем проблемные переменные окружения Vulkan SDK
unset VK_ICD_FILENAMES
unset VK_LAYER_PATH

echo "Попытка запуска с различными драйверами Vulkan..."
echo ""

# Пробуем использовать NVIDIA драйвер
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    echo "Пробуем NVIDIA драйвер..."
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    cd "$(dirname "$0")"
    if ./build/Lab1_3DGraphics 2>&1; then
        exit 0
    fi
fi

# Пробуем Intel драйвер
if [ -f /usr/share/vulkan/icd.d/intel_icd.x86_64.json ]; then
    echo "Пробуем Intel драйвер..."
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
    cd "$(dirname "$0")"
    if ./build/Lab1_3DGraphics 2>&1; then
        exit 0
    fi
fi

# Пробуем software renderer
if [ -f /usr/share/vulkan/icd.d/lvp_icd.x86_64.json ]; then
    echo "Пробуем software renderer (медленнее, но должен работать)..."
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
    cd "$(dirname "$0")"
    if ./build/Lab1_3DGraphics 2>&1; then
        exit 0
    fi
fi

echo ""
echo "Не удалось запустить программу ни с одним драйвером."
echo "Попробуйте:"
echo "1. Перезагрузить компьютер после установки драйверов NVIDIA"
echo "2. Проверить драйверы: vulkaninfo --summary"
echo "3. Установить драйверы: sudo apt-get install mesa-vulkan-drivers"

