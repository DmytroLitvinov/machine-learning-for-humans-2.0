def analyze_dataframe(df):
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    binary_columns = []
    multicategory_columns = []

    for col in categorical_columns:
        unique_values = df[col].nunique()
        if unique_values == 2:
            binary_columns.append(col)
        elif unique_values > 2:
            multicategory_columns.append(col)

    # Виведення результатів
    print(f"Кількість числових колонок: {len(numeric_columns)}")
    print(f"Кількість категоріальних колонок: {len(categorical_columns)}")
    print(f" - Бінарних: {len(binary_columns)}")
    print(f" - Мультикатегоріальних: {len(multicategory_columns)}")

    print("Числові колонки:", numeric_columns)
    print("Категоріальні колонки:", categorical_columns)
    print("Бінарні колонки:", binary_columns)
    print("Мультикатегоріальні колонки:", multicategory_columns)