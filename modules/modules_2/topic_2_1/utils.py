import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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
    print(f"Кількість рядків: {df.shape[0]}")
    print(f"Кількість колонок: {df.shape[1]}")
    print(f"Кількість числових колонок: {len(numeric_columns)}")
    print(f"Кількість категоріальних колонок: {len(categorical_columns)}")
    print(f" - Бінарних: {len(binary_columns)}")
    print(f" - Мультикатегоріальних: {len(multicategory_columns)}")

    print("Числові колонки:", numeric_columns)
    print("Категоріальні колонки:", categorical_columns)
    print("Бінарні колонки:", binary_columns)
    print("Мультикатегоріальні колонки:", multicategory_columns)

def get_importance_df(model, X_train):
    result = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    return result


def render_feature_importance(importance_df, n_features=10):
    plt.title('Feature Importance')
    sns.barplot(data=importance_df.head(n_features), x='importance', y='feature')