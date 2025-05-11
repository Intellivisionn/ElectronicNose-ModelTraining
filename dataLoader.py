from enums import Label
from transformer import transform

def loadData() -> list:
    datafiles = {
        Label.APPLE.value: 'Data\\kokot_apple_20250423_122409.json',
        Label.BANANA.value: 'Data\\kokot_banana_20250418_160651.json',
        Label.BLOOD_ORANGE.value: 'Data\\kokot_blood_orange_20250424_143025.json',
        Label.BLUEBERRY.value: 'Data\\kokot_blueberry_20250420_145344.json',
        Label.COCONUT.value: 'Data\\kokot_coconut_20250417_174342.json',
        Label.GRAPE.value: 'Data\\kokot_grape_20250422_144509.json',
        Label.LAVENDER.value: 'Data\\kokot_lavender_20250415_130548.json',
        Label.MANGO.value: 'Data\\kokot_mango_20250419_205843.json',
        Label.MELON.value: 'Data\\kokot_melon_20250422_101805.json',
        Label.PINEAPPLE.value: 'Data\\kokot_pineapple_20250423_191451.json',
        Label.STRAWBERRY.value: 'Data\\kokot_strawberry _20250417_151214.json',
        Label.VANILLA.value: 'Data\\kokot_vanilla_20250414_160354.json'
    }
    full_data = []

    for label, file_path in datafiles.items():
        transformed_data = transform(file_path, label)
        full_data.extend(transformed_data)

    return full_data