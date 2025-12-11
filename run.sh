#!/bin/bash

# Скрипт для сборки и запуска лабораторной работы
# Исправляет проблемы с путями к драйверам Vulkan

# Сохраняем, что задал пользователь
USER_ICD="$VK_ICD_FILENAMES"

# Убираем проблемные переменные окружения Vulkan SDK,
# но если пользователь заранее задал VK_ICD_FILENAMES — оставляем.
if [ -z "$VK_ICD_FILENAMES" ]; then
    unset VK_ICD_FILENAMES
else
    echo "Используем VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
fi
unset VK_LAYER_PATH

# Собираем проект (и шейдеры)
cd "$(dirname "$0")"
./compile_shaders.sh || { echo "Не удалось собрать шейдеры"; exit 1; }

# Чистим старый кэш, чтобы избежать несоответствия путей
rm -rf build

cmake -S . -B build >/dev/null || { echo "cmake configure failed"; exit 1; }
cmake --build build --config Release -j >/dev/null || { echo "cmake build failed"; exit 1; }

echo "Попытка запуска с различными драйверами Vulkan..."
echo ""

cd "$(dirname "$0")"

# Если хотите форсировать LLVM/soft (чтобы не висло при поиске GPU), раскомментируйте:
# export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json

# Иначе пробуем по очереди
try_icd() {
    local icd="$1"
    local name="$2"
    if [ -f "$icd" ]; then
        echo "Пробуем $name драйвер..."
        export VK_ICD_FILENAMES="$icd"
        ./build/Lab1_3DGraphics 2>&1 && exit 0
    fi
}

if [ -n "$USER_ICD" ]; then
    echo "Запуск с пользовательским ICD: $USER_ICD"
    export VK_ICD_FILENAMES="$USER_ICD"
    ./build/Lab1_3DGraphics 2>&1 && exit 0
else
    try_icd /usr/share/vulkan/icd.d/nvidia_icd.json "NVIDIA"
    try_icd /usr/share/vulkan/icd.d/intel_icd.x86_64.json "Intel"
    try_icd /usr/share/vulkan/icd.d/lvp_icd.x86_64.json "software renderer (LLVM)"
fi

echo ""
echo "Не удалось запустить программу ни с одним драйвером."
echo "Попробуйте:"
echo "1. Перезагрузить компьютер после установки драйверов NVIDIA"
echo "2. Проверить драйверы: vulkaninfo --summary"
echo "3. Установить драйверы: sudo apt-get install mesa-vulkan-drivers"

