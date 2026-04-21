# Лабораторная работа №6
## Вариант 17. Сегментация текста для казахского алфавита

Для варианта 17 в Microsoft Word набрана фраза `сәулем мен сені сүйемін` шрифтом `Arial`, размер `52`, после чего изображение строки сохранено в монохромный файл `bmp`.

Основные артефакты:

- исходная строка из Microsoft Word в `bmp`: [source_bmp](lab6/source_bmp)
- координаты обрамляющих прямоугольников: [segments_variant17.csv](lab6/segments_variant17.csv)
- вырезанные символы: [segmented_symbols_png](lab6/segmented_symbols_png)
- профили строки: [profiles_png](lab6/profiles_png)
- профили символов алфавита: [alphabet_profiles_x_png](lab6/alphabet_profiles_x_png), [alphabet_profiles_y_png](lab6/alphabet_profiles_y_png)

### Исходная строка

![phrase](lab6/preview_png/phrase_preview.png)

### Горизонтальный и вертикальный профили строки

Горизонтальный профиль показывает распределение черных пикселей по строкам, а вертикальный — по столбцам. Именно вертикальный профиль используется для разбиения строки на отдельные символы.

![horizontal](lab6/profiles_png/horizontal_profile.png)

![vertical](lab6/profiles_png/vertical_profile.png)

### Результат сегментации

Сегментация выполнена по вертикальному профилю с порогом `2` пикселя. На выходе получено `20` обрамляющих прямоугольников

#### Прямоугольники на строке

![boxes](lab6/preview_png/segmentation_boxes.png)

#### Вырезанные символы

![segments](lab6/preview_png/segments_overview.png)

### Фрагмент CSV с координатами сегментов

Первые 8 сегментов из [segments_variant17.csv](lab6/segments_variant17.csv):

| № | Символ | Код | `left` | `top` | `right` | `bottom` | `width` | `height` |
|--:|:------:|:---:|------:|-----:|-------:|--------:|--------:|---------:|
| 1 | `с` | `u0441` | 0 | 7 | 17 | 27 | 17 | 20 |
| 2 | `ә` | `u04d9` | 19 | 7 | 38 | 27 | 19 | 20 |
| 3 | `у` | `u0443` | 40 | 7 | 55 | 34 | 15 | 27 |
| 4 | `л` | `u043b` | 58 | 7 | 78 | 27 | 20 | 20 |
| 5 | `е` | `u0435` | 81 | 7 | 99 | 27 | 18 | 20 |
| 6 | `м` | `u043c` | 103 | 7 | 124 | 27 | 21 | 20 |
| 7 | `м` | `u043c` | 140 | 7 | 161 | 27 | 21 | 20 |
| 8 | `е` | `u0435` | 165 | 7 | 183 | 27 | 18 | 20 |

### Профили символов выбранного алфавита

Для всех букв казахского алфавита построены отдельные профили `X` и `Y`. Ниже приведены несколько характерных примеров.

| Символ | Изображение | Профиль X | Профиль Y |
|:--:|:--:|:--:|:--:|
| `ә` | ![u04d9](lab6/alphabet_symbols_png/u04d9.png) | ![u04d9x](lab6/alphabet_profiles_x_png/u04d9_x.png) | ![u04d9y](lab6/alphabet_profiles_y_png/u04d9_y.png) |
| `ү` | ![u04af](lab6/alphabet_symbols_png/u04af.png) | ![u04afx](lab6/alphabet_profiles_x_png/u04af_x.png) | ![u04afy](lab6/alphabet_profiles_y_png/u04af_y.png) |
| `і` | ![u0456](lab6/alphabet_symbols_png/u0456.png) | ![u0456x](lab6/alphabet_profiles_x_png/u0456_x.png) | ![u0456y](lab6/alphabet_profiles_y_png/u0456_y.png) |
| `ң` | ![u04a3](lab6/alphabet_symbols_png/u04a3.png) | ![u04a3x](lab6/alphabet_profiles_x_png/u04a3_x.png) | ![u04a3y](lab6/alphabet_profiles_y_png/u04a3_y.png) |

#### Обзор части алфавита

![alphabet](lab6/preview_png/alphabet_overview.png)

### Вывод

Для варианта 17 строка `сәулем мен сені сүйемін` подготовлена в Microsoft Word и сохранена в монохромный `bmp`, после чего построены горизонтальный и вертикальный профили и выполнена сегментация символов по профилям.
